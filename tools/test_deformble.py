#!/usr/bin/env python3
# Copyright (c) OpenMMLab.
"""
Visualize all sampling locations from the 12 DeformableFeatureAggregation
modules in SparseDrive (6 detection + 6 map) for the first N test frames.

This script DOES NOT modify SparseDrive model source code. It only registers
forward hooks at runtime. Model outputs are left untouched.

Recommended location:
    tools/test_deformble.py

Example:
    python tools/test_deformble.py \
        projects/configs/sparsedrive_small_stage2.py \
        path/to/sparsedrive_stage2.pth \
        --num-frames 50 \
        --out-dir output/deformble_plot

Optional:
    --save-npz
        Also save normalized (u, v) coordinates and valid masks for later
        quantitative analysis.
"""

import argparse
import importlib
import os
import os.path as osp
import re
import warnings

import mmcv
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, DataContainer as DC
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import build_dataset, replace_ImageToTensor
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description="SparseDrive deformable sampling-point visualizer"
    )
    parser.add_argument("config", help="test config file")
    parser.add_argument("checkpoint", help="model checkpoint")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=50,
        help="number of sequential test frames to run (default: 50)",
    )
    parser.add_argument(
        "--out-dir",
        default="output/deformble_plot",
        help="directory for PNG visualizations",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU id used for inference",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=2,
        help="test dataloader workers (default: 2)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="output figure dpi",
    )
    parser.add_argument(
        "--det-size",
        type=float,
        default=0.8,
        help="scatter marker size for detection sampling points",
    )
    parser.add_argument(
        "--map-size",
        type=float,
        default=3.0,
        help="scatter marker size for map sampling points",
    )
    parser.add_argument(
        "--det-alpha",
        type=float,
        default=0.12,
        help="scatter alpha for detection sampling points",
    )
    parser.add_argument(
        "--map-alpha",
        type=float,
        default=0.45,
        help="scatter alpha for map sampling points",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="also save uv/valid arrays for every frame",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.03,
        help="minimum alpha for low-weight valid points",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=0.85,
        help="maximum alpha for high-weight points",
    )
    parser.add_argument(
        "--weight-percentile",
        type=float,
        default=99.0,
        help=(
            "percentile used as the high end when mapping aggregation weight "
            "to scatter alpha (default: 99)"
        ),
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, e.g. key=value",
    )
    return parser.parse_args()


def import_plugin(cfg, config_path):
    """Match the plugin import behavior used by tools/test.py."""
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    if not getattr(cfg, "plugin", False):
        return

    if hasattr(cfg, "plugin_dir"):
        module_path = cfg.plugin_dir.rstrip("/").replace("/", ".")
    else:
        module_path = osp.dirname(config_path).replace("/", ".").strip(".")

    print("[plugin] importing:", module_path)
    importlib.import_module(module_path)


def prepare_test_cfg(cfg):
    """Prepare test dataset config while forcing batch size == 1."""
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        # We explicitly run one frame at a time because temporal memory and
        # per-frame plotting are both easier to interpret this way.
        cfg.data.test.pop("samples_per_gpu", None)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
            ds_cfg.pop("samples_per_gpu", None)
            # Kept for compatibility if a config had >1 samples/GPU before.
            if "pipeline" in ds_cfg:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    else:
        raise TypeError("cfg.data.test must be a dict or list")


def unwrap_data(value):
    """Unwrap MMDetection DataContainer/list wrappers without changing data."""
    if isinstance(value, DC):
        value = value.data

    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
        if isinstance(value, DC):
            value = value.data

    return value


def extract_display_images(data, cfg):
    """
    Return transformed model-input camera images as RGB float images in [0, 1].

    The sampling coordinates in metas['projection_mat']/image_wh correspond to
    these resized/cropped test images, so plotting on these tensors gives the
    correct visual alignment.
    """
    img = unwrap_data(data["img"])

    # Some MMCV collate paths produce one additional list wrapper.
    if isinstance(img, (list, tuple)):
        if len(img) != 1:
            raise RuntimeError(
                "Unexpected test-time augmentation list. "
                "This script expects the normal SparseDrive test pipeline."
            )
        img = img[0]

    if not torch.is_tensor(img):
        raise TypeError("Could not unwrap data['img'] to a torch.Tensor")

    # Expected [B, num_cams, 3, H, W], with B forced to 1.
    if img.ndim == 4:
        img = img.unsqueeze(0)
    if img.ndim != 5:
        raise RuntimeError("Unexpected image shape: {}".format(tuple(img.shape)))
    if img.shape[0] != 1:
        raise RuntimeError(
            "This visualizer requires batch size 1, got {}".format(img.shape[0])
        )

    imgs = img[0].detach().float().cpu().permute(0, 2, 3, 1).numpy()

    norm_cfg = cfg.get(
        "img_norm_cfg",
        dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
        ),
    )
    mean = np.asarray(norm_cfg["mean"], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray(norm_cfg["std"], dtype=np.float32).reshape(1, 1, 1, 3)

    imgs = imgs * std + mean

    # mmcv.imnormalize(..., to_rgb=True) converts BGR input to RGB before
    # normalization, so the inverse is already RGB. If to_rgb=False, convert
    # BGR -> RGB for matplotlib.
    if not norm_cfg.get("to_rgb", True):
        imgs = imgs[..., ::-1]

    imgs = np.clip(imgs / 255.0, 0.0, 1.0)
    return imgs


def get_camera_names(dataset, frame_idx, num_cams):
    """
    Camera iteration in NuScenes3DDataset follows info["cams"].items().
    Reuse exactly that dictionary order when available.
    """
    try:
        cams = list(dataset.data_infos[frame_idx]["cams"].keys())
        if len(cams) == num_cams:
            return cams
    except Exception:
        pass
    return ["CAM_{}".format(i) for i in range(num_cams)]


def numeric_name_key(name):
    nums = [int(x) for x in re.findall(r"\d+", name)]
    return nums if nums else [0]


class DeformableSamplingRecorder:
    """
    Runtime-only observer for DeformableFeatureAggregation.

    Important:
      * No model output is replaced.
      * No parameter is changed.
      * kps_generator output is captured by a forward hook, so the exact
        keypoints used by the original forward pass are recorded.
      * A second forward hook projects those exact 3D keypoints to image
        coordinates and stores only detached CPU arrays.
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.enabled = False
        self.records = []
        self.layer_specs = []

    def discover(self):
        det = []
        map_ = []

        for name, module in self.model.named_modules():
            if module.__class__.__name__ != "DeformableFeatureAggregation":
                continue

            if "det_head" in name:
                det.append((name, module))
            elif "map_head" in name:
                map_.append((name, module))

        # named_modules() normally already follows ModuleList order. Sorting
        # numeric indices makes the ordering explicit and robust.
        det.sort(key=lambda x: numeric_name_key(x[0]))
        map_.sort(key=lambda x: numeric_name_key(x[0]))

        if len(det) != 6 or len(map_) != 6:
            found = [x[0] for x in det + map_]
            raise RuntimeError(
                "Expected 6 detection + 6 map deformable modules, "
                "but found det={} map={}.\nFound:\n{}".format(
                    len(det), len(map_), "\n".join(found)
                )
            )

        self.layer_specs = []
        for idx, (name, module) in enumerate(det, 1):
            self.layer_specs.append(
                dict(branch="det", layer=idx, name=name, module=module)
            )
        for idx, (name, module) in enumerate(map_, 1):
            self.layer_specs.append(
                dict(branch="map", layer=idx, name=name, module=module)
            )

        print("[recorder] deformable modules:")
        for spec in self.layer_specs:
            print(
                "  {:>3s}-L{}  {}".format(
                    spec["branch"].upper(), spec["layer"], spec["name"]
                )
            )

    @staticmethod
    def _project_with_validity(key_points, metas):
        """
        key_points:     [B, N, P, 3]
        projection_mat: [B, C, 4, 4]
        image_wh:       [B, C, 2]

        Returns
        -------
        uv:    [B, C, N, P, 2], normalized to [0,1] image coordinates
        valid: [B, C, N, P], requiring positive depth and in-image uv
        """
        projection_mat = metas["projection_mat"]
        image_wh = metas.get("image_wh", None)

        pts_h = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])],
            dim=-1,
        )

        # [B,C,1,1,4,4] @ [B,1,N,P,4,1]
        projected = torch.matmul(
            projection_mat[:, :, None, None],
            pts_h[:, None, ..., None],
        ).squeeze(-1)

        depth = projected[..., 2]
        safe_depth = torch.clamp(depth, min=1e-5)
        uv = projected[..., :2] / safe_depth[..., None]

        if image_wh is not None:
            uv = uv / image_wh[:, :, None, None]

        finite = torch.isfinite(uv).all(dim=-1) & torch.isfinite(depth)
        valid = (
            finite
            & (depth > 1e-5)
            & (uv[..., 0] >= 0.0)
            & (uv[..., 0] <= 1.0)
            & (uv[..., 1] >= 0.0)
            & (uv[..., 1] <= 1.0)
        )
        return uv, valid

    def register(self):
        self.discover()

        for spec in self.layer_specs:
            module = spec["module"]
            state = {}

            def make_kps_hook(local_state):
                def kps_hook(_kps_module, _inputs, output):
                    if not self.enabled:
                        return
                    key_points = output[0] if isinstance(output, tuple) else output
                    # Keep only a detached reference until the parent DAF hook
                    # runs a moment later.
                    local_state["key_points"] = key_points.detach()
                return kps_hook

            def make_daf_hook(local_spec, local_state):
                def daf_hook(_module, inputs, _output):
                    if not self.enabled:
                        local_state.pop("key_points", None)
                        return

                    if "key_points" not in local_state:
                        raise RuntimeError(
                            "DAF hook fired without captured keypoints for "
                            + local_spec["name"]
                        )

                    if len(inputs) < 5:
                        raise RuntimeError(
                            "Unexpected DeformableFeatureAggregation inputs "
                            "for {}".format(local_spec["name"])
                        )

                    metas = inputs[4]
                    key_points = local_state.pop("key_points")

                    # inputs:
                    #   0 instance_feature [B,N,256]
                    #   1 anchor
                    #   2 anchor_embed      [B,N,256]
                    #   3 feature_maps
                    #   4 metas
                    instance_feature = inputs[0]
                    anchor_embed = inputs[2]

                    with torch.no_grad():
                        uv, valid = self._project_with_validity(
                            key_points, metas
                        )

                        # Recompute exactly the same aggregation weights used
                        # by DAF. In model.eval() attn_drop is disabled, so
                        # this is deterministic and does not modify inference.
                        #
                        # raw_weights:
                        # [B,N,C,L,P,G]
                        raw_weights = _module._get_weights(
                            instance_feature,
                            anchor_embed,
                            metas,
                        )

                        # Collapse group and FPN level so every real image-space
                        # sampling location (camera, query, point) has one
                        # scalar weight:
                        #
                        # [B,N,C,L,P,G]
                        #   mean over G
                        # -> [B,N,C,L,P]
                        #   sum over L
                        # -> [B,N,C,P]
                        #
                        # For every query, sum over C and P remains 1.
                        spatial_weight = (
                            raw_weights.mean(dim=-1).sum(dim=3)
                        )

                        # Match uv layout [B,C,N,P,...].
                        spatial_weight = spatial_weight.permute(
                            0, 2, 1, 3
                        )

                    if uv.shape[0] != 1:
                        raise RuntimeError(
                            "Recorder expects B=1, got uv shape {}".format(
                                tuple(uv.shape)
                            )
                        )

                    # Drop batch dimension. Shapes:
                    # uv      [C,N,P,2]
                    # valid   [C,N,P]
                    # weight  [C,N,P]
                    self.records.append(
                        dict(
                            branch=local_spec["branch"],
                            layer=local_spec["layer"],
                            name=local_spec["name"],
                            uv=uv[0].detach().float().cpu().numpy(),
                            valid=valid[0].detach().cpu().numpy(),
                            weight=spatial_weight[0]
                            .detach()
                            .float()
                            .cpu()
                            .numpy(),
                            num_queries=int(key_points.shape[1]),
                            num_points=int(key_points.shape[2]),
                        )
                    )
                return daf_hook

            self.handles.append(
                module.kps_generator.register_forward_hook(
                    make_kps_hook(state)
                )
            )
            self.handles.append(
                module.register_forward_hook(
                    make_daf_hook(spec, state)
                )
            )

    def begin_frame(self):
        self.records = []
        self.enabled = True

    def end_frame(self):
        self.enabled = False

        # Sort into Det L1..L6, Map L1..L6 regardless of execution append order.
        branch_order = {"det": 0, "map": 1}
        self.records.sort(
            key=lambda r: (branch_order[r["branch"]], r["layer"])
        )

        if len(self.records) != 12:
            warnings.warn(
                "Expected 12 DAF records for this frame, got {}.".format(
                    len(self.records)
                )
            )
        return self.records

    def remove(self):
        self.enabled = False
        for h in self.handles:
            h.remove()
        self.handles = []


def make_branch_colors():
    """Only two visual categories: Detection and Map."""
    return {
        "det": "tab:blue",
        "map": "tab:red",
    }



def weight_to_alpha(weights, valid_mask, alpha_min, alpha_max, percentile):
    """
    Map aggregation weights to visible alpha values.

    Weights are genuine model aggregation weights. Only the visualization
    mapping is normalized:
      - 0 maps to alpha_min
      - the requested positive-weight percentile maps to alpha_max
      - values above it are clipped to alpha_max

    A sqrt transform makes mid/low weights easier to see while preserving
    monotonicity: larger weight always means larger/equal alpha.
    """
    weights = np.asarray(weights, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    alpha = np.zeros_like(weights, dtype=np.float32)
    values = weights[valid_mask]

    if values.size == 0:
        return alpha

    positive = values[values > 0]
    if positive.size == 0:
        alpha[valid_mask] = alpha_min
        return alpha

    percentile = float(np.clip(percentile, 50.0, 100.0))
    high = float(np.percentile(positive, percentile))
    if not np.isfinite(high) or high <= 0:
        high = float(positive.max())
    if high <= 0:
        alpha[valid_mask] = alpha_min
        return alpha

    normalized = np.clip(weights / high, 0.0, 1.0)
    normalized = np.sqrt(normalized)

    alpha[valid_mask] = (
        alpha_min
        + normalized[valid_mask] * (alpha_max - alpha_min)
    )
    return alpha


def make_rgba(color, alpha_values):
    """Create one RGBA color per scatter point."""
    rgba_base = np.asarray(
        matplotlib.colors.to_rgba(color),
        dtype=np.float32,
    )
    rgba = np.repeat(
        rgba_base[None, :],
        len(alpha_values),
        axis=0,
    )
    rgba[:, 3] = np.asarray(alpha_values, dtype=np.float32)
    return rgba


def plot_frame(
    frame_idx,
    images,
    records,
    camera_names,
    out_file,
    det_size,
    map_size,
    det_alpha,
    map_alpha,
    dpi,
    alpha_min,
    alpha_max,
    weight_percentile,
):
    """
    Visualize all 12 deformable layers as two categories.

    Color:
        blue = Detection
        red x = Map

    Alpha:
        driven by the real aggregation weight of each (camera, query, point)
        after averaging 8 feature groups and summing the 4 FPN levels.

    The legacy det_alpha/map_alpha arguments are retained for CLI backwards
    compatibility but are no longer used as a constant alpha.
    """
    num_cams = images.shape[0]
    if num_cams != 6:
        raise RuntimeError("Expected 6 cameras, got {}".format(num_cams))

    if not (0.0 <= alpha_min <= 1.0):
        raise ValueError("--alpha-min must be in [0, 1]")
    if not (0.0 <= alpha_max <= 1.0):
        raise ValueError("--alpha-max must be in [0, 1]")
    if alpha_min > alpha_max:
        raise ValueError("--alpha-min cannot exceed --alpha-max")

    colors = make_branch_colors()

    fig, axes = plt.subplots(
        2, 3,
        figsize=(19, 8.8),
        squeeze=False,
    )
    axes = axes.reshape(-1)

    det_records = [r for r in records if r["branch"] == "det"]
    map_records = [r for r in records if r["branch"] == "map"]

    total_valid_det = 0
    total_valid_map = 0

    # Compute branch-wise robust high thresholds across the whole frame.
    # This keeps alpha comparable across all six layers/cameras in the same
    # branch, while avoiding one extreme point washing out the visualization.
    branch_high = {}
    for branch, branch_records in [
        ("det", det_records),
        ("map", map_records),
    ]:
        vals = []
        for rec in branch_records:
            v = rec["weight"][rec["valid"]]
            if v.size:
                vals.append(v)
        if vals:
            vals = np.concatenate(vals)
            positive = vals[vals > 0]
            if positive.size:
                branch_high[branch] = float(
                    np.percentile(
                        positive,
                        np.clip(weight_percentile, 50.0, 100.0),
                    )
                )
            else:
                branch_high[branch] = 1.0
        else:
            branch_high[branch] = 1.0

    def alpha_from_branch_scale(weights, valid, branch):
        high = branch_high[branch]
        weights = np.asarray(weights, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        alpha = np.zeros_like(weights, dtype=np.float32)

        if not valid.any():
            return alpha

        if not np.isfinite(high) or high <= 0:
            alpha[valid] = alpha_min
            return alpha

        norm = np.clip(weights / high, 0.0, 1.0)
        norm = np.sqrt(norm)
        alpha[valid] = alpha_min + norm[valid] * (
            alpha_max - alpha_min
        )
        return alpha

    for cam_idx, ax in enumerate(axes):
        image = images[cam_idx]
        h, w = image.shape[:2]
        ax.imshow(image)

        cam_det_count = 0
        cam_map_count = 0

        # Detection first.
        for rec in det_records:
            valid = rec["valid"][cam_idx]
            pts = rec["uv"][cam_idx][valid]
            weights = rec["weight"][cam_idx]
            alphas = alpha_from_branch_scale(
                weights, valid, "det"
            )[valid]

            cam_det_count += len(pts)
            total_valid_det += len(pts)

            if len(pts) == 0:
                continue

            rgba = make_rgba(colors["det"], alphas)

            ax.scatter(
                pts[:, 0] * w,
                pts[:, 1] * h,
                s=det_size,
                marker="o",
                c=rgba,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )

        # Map on top.
        for rec in map_records:
            valid = rec["valid"][cam_idx]
            pts = rec["uv"][cam_idx][valid]
            weights = rec["weight"][cam_idx]
            alphas = alpha_from_branch_scale(
                weights, valid, "map"
            )[valid]

            cam_map_count += len(pts)
            total_valid_map += len(pts)

            if len(pts) == 0:
                continue

            rgba = make_rgba(colors["map"], alphas)

            ax.scatter(
                pts[:, 0] * w,
                pts[:, 1] * h,
                s=map_size,
                marker="x",
                c=rgba,
                linewidths=0.35,
                rasterized=True,
                zorder=3,
            )

        cam_name = (
            camera_names[cam_idx]
            if cam_idx < len(camera_names)
            else "CAM_{}".format(cam_idx)
        )

        ax.set_title(
            "{} | det={:,}, map={:,}".format(
                cam_name,
                cam_det_count,
                cam_map_count,
            ),
            fontsize=9,
        )
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=colors["det"],
            markeredgecolor="none",
            alpha=0.85,
            label="Detection (alpha = aggregation weight)",
        ),
        Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            markersize=7,
            color=colors["map"],
            markeredgewidth=1.2,
            alpha=0.9,
            label="Map (alpha = aggregation weight)",
        ),
    ]

    fig.suptitle(
        "SparseDrive deformable sampling | frame {:03d}\n"
        "alpha encodes aggregation weight "
        "(group-mean, FPN-level-sum; p{:g} -> alpha_max)".format(
            frame_idx,
            weight_percentile,
        ),
        fontsize=13,
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout(rect=[0.0, 0.075, 1.0, 0.94])
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return total_valid_det, total_valid_map


def save_frame_npz(frame_idx, records, camera_names, out_dir):
    payload = {
        "camera_names": np.asarray(camera_names),
    }
    for rec in records:
        prefix = "{}_L{}".format(rec["branch"], rec["layer"])
        payload[prefix + "_uv"] = rec["uv"].astype(np.float16)
        payload[prefix + "_valid"] = rec["valid"].astype(np.bool_)
        payload[prefix + "_weight"] = rec["weight"].astype(np.float16)

    np.savez_compressed(
        osp.join(out_dir, "frame_{:03d}.npz".format(frame_idx)),
        **payload
    )


def main():
    args = parse_args()

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SparseDrive inference.")

    torch.cuda.set_device(args.gpu_id)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    import_plugin(cfg, args.config)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # Same outer-detector override as the repository's tools/test.py.
    cfg.model.pretrained = None
    prepare_test_cfg(cfg)

    if cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs",
            osp.splitext(osp.basename(args.config))[0],
        )
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    if isinstance(cfg.data.test, dict):
        cfg.data.test.work_dir = cfg.work_dir

    dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader_origin(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    cfg.model.train_cfg = None
    model = build_detector(
        cfg.model,
        test_cfg=cfg.get("test_cfg"),
    )

    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location="cpu",
    )

    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    # Register hooks BEFORE wrapping in MMDataParallel. They remain attached
    # to the underlying module and observe the actual forward pass.
    recorder = DeformableSamplingRecorder(model)
    recorder.register()

    model = model.cuda(args.gpu_id)
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.eval()

    mmcv.mkdir_or_exist(args.out_dir)
    raw_dir = osp.join(args.out_dir, "raw")
    if args.save_npz:
        mmcv.mkdir_or_exist(raw_dir)

    max_frames = min(args.num_frames, len(dataset))
    print(
        "[run] visualizing {} frame(s) -> {}".format(
            max_frames, osp.abspath(args.out_dir)
        )
    )

    try:
        for frame_idx, data in enumerate(data_loader):
            if frame_idx >= max_frames:
                break

            # Extract a CPU copy before the data is scattered by MMDataParallel.
            images = extract_display_images(data, cfg)
            camera_names = get_camera_names(
                dataset, frame_idx, images.shape[0]
            )

            recorder.begin_frame()
            with torch.no_grad():
                # Match mmdet single_gpu_test's inference call. SparseDrive's
                # own forward ignores return_loss/rescale for computation.
                _ = model(
                    return_loss=False,
                    rescale=True,
                    **data
                )
            records = recorder.end_frame()

            out_file = osp.join(
                args.out_dir,
                "frame_{:03d}.png".format(frame_idx),
            )
            valid_det, valid_map = plot_frame(
                frame_idx=frame_idx,
                images=images,
                records=records,
                camera_names=camera_names,
                out_file=out_file,
                det_size=args.det_size,
                map_size=args.map_size,
                det_alpha=args.det_alpha,
                map_alpha=args.map_alpha,
                dpi=args.dpi,
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
                weight_percentile=args.weight_percentile,
            )

            if args.save_npz:
                save_frame_npz(
                    frame_idx,
                    records,
                    camera_names,
                    raw_dir,
                )

            det_desc = [
                "{}q×{}p".format(r["num_queries"], r["num_points"])
                for r in records if r["branch"] == "det"
            ]
            map_desc = [
                "{}q×{}p".format(r["num_queries"], r["num_points"])
                for r in records if r["branch"] == "map"
            ]

            print(
                "[{}/{}] saved {} | det={} | map={}".format(
                    frame_idx + 1,
                    max_frames,
                    out_file,
                    (det_desc[0] if det_desc else "N/A")
                    + ", valid={:,}".format(valid_det),
                    (map_desc[0] if map_desc else "N/A")
                    + ", valid={:,}".format(valid_map),
                )
            )
    finally:
        recorder.remove()

    print("[done] sampling plots:", osp.abspath(args.out_dir))
    if args.save_npz:
        print("[done] raw sampling arrays:", osp.abspath(raw_dir))


if __name__ == "__main__":
    # Keep the behavior compatible with the repository's tools/test.py.
    try:
        torch.multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()