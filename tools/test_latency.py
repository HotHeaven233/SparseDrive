#!/usr/bin/env python3
# Copyright (c) OpenMMLab.
"""
SparseDrive component latency benchmark.

Recommended location:
    tools/test_latency.py

What is measured (single GPU, batch size = 1):
    - resnet_ms              : img_backbone (ResNet)
    - fpn_ms                 : img_neck (FPN)
    - extract_other_ms       : GridMask / reshape / feature_maps_format / other
                               work inside SparseDrive.extract_feat
    - extract_feat_total_ms  : complete SparseDrive.extract_feat
    - det_ms                 : SparseDriveHead.det_head
    - map_ms                 : SparseDriveHead.map_head
    - motion_plan_ms         : SparseDriveHead.motion_plan_head
    - head_other_ms          : small orchestration overhead inside head.forward
    - head_total_ms          : complete SparseDriveHead.forward
    - postprocess_ms         : SparseDriveHead.post_process GPU elapsed time
    - total_ms               : synchronized end-to-end model-call wall time

The dataloader iteration itself is NOT included in total_ms. The MMDataParallel
scatter that happens after the timer starts IS included in total_ms.

Timing implementation:
    * CUDA Events are used for GPU component timings.
    * total_ms uses:
          torch.cuda.synchronize()
          perf_counter()
          model(...)
          torch.cuda.synchronize()
      so it reflects real synchronized per-frame inference latency.

No SparseDrive source file or checkpoint is modified. Forward hooks and
temporary method wrappers are removed/restored at the end.

Example:
    python tools/test_latency.py \
        projects/configs/sparsedrive_small_stage2.py \
        ckpt/sparsedrive_stage2.pth \
        --num-frames 100 \
        --out-dir output/latency

By default 10 sequential warm-up frames are run before the 100 measured frames.
Use --warmup-frames 0 if you want exactly 100 total forward passes.
"""

import argparse
import csv
import importlib
import json
import os
import os.path as osp
import statistics
import time
from collections import defaultdict

import mmcv
import numpy as np
import torch

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector


COMPONENT_COLUMNS = [
    "resnet_ms",
    "fpn_ms",
    "extract_other_ms",
    "extract_feat_total_ms",
    "det_ms",
    "map_ms",
    "motion_plan_ms",
    "head_other_ms",
    "head_total_ms",
    "postprocess_ms",
    "total_ms",
    "fps",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="SparseDrive per-component latency benchmark"
    )
    parser.add_argument("config", help="test config file")
    parser.add_argument("checkpoint", help="model checkpoint")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="number of MEASURED sequential frames (default: 100)",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="sequential warm-up frames before measurement (default: 10)",
    )
    parser.add_argument(
        "--out-dir",
        default="output/latency",
        help="output directory (default: output/latency)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU id (default: 0)",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=2,
        help="test dataloader workers (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed (default: 0)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="set deterministic cuDNN behavior",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, e.g. key=value",
    )
    return parser.parse_args()


def import_plugin(cfg, config_path):
    """Use the same plugin-loading convention as SparseDrive tools/test.py."""
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
    """
    Force normal test mode and batch size 1.

    Batch size 1 is important because SparseDrive has temporal instance memory
    and because the requested latency is per frame.
    """
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        cfg.data.test.pop("samples_per_gpu", None)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
            ds_cfg.pop("samples_per_gpu", None)
    else:
        raise TypeError("cfg.data.test must be a dict or list")


class ComponentTimer:
    """
    Runtime-only CUDA-event recorder.

    Module forward pre/post hooks create CUDA event pairs. Method wrappers use
    the same event-pair API. Events are converted to milliseconds only after
    the whole frame has synchronized, so no extra synchronization is inserted
    between SparseDrive components.
    """

    def __init__(self):
        self.enabled = False
        self.event_pairs = defaultdict(list)
        self.handles = []
        self.original_methods = []

    @staticmethod
    def _event():
        return torch.cuda.Event(enable_timing=True)

    def begin_frame(self):
        self.event_pairs = defaultdict(list)
        self.enabled = True

    def end_frame(self):
        self.enabled = False

        # Caller has already torch.cuda.synchronize()'d after model forward.
        result = {}
        for name, pairs in self.event_pairs.items():
            total = 0.0
            for start_event, end_event in pairs:
                total += float(start_event.elapsed_time(end_event))
            result[name] = total
        return result

    def add_event_pair(self, name, start_event, end_event):
        if self.enabled:
            self.event_pairs[name].append((start_event, end_event))

    def register_module(self, name, module):
        """Time one nn.Module's complete forward execution."""
        if module is None:
            return

        state = {"stack": []}

        def pre_hook(_module, _inputs):
            if not self.enabled:
                return
            start = self._event()
            start.record()
            state["stack"].append(start)

        def post_hook(_module, _inputs, _output):
            if not self.enabled:
                return
            if not state["stack"]:
                raise RuntimeError(
                    "Latency timer post-hook without matching pre-hook: "
                    + name
                )
            start = state["stack"].pop()
            end = self._event()
            end.record()
            self.add_event_pair(name, start, end)

        self.handles.append(module.register_forward_pre_hook(pre_hook))
        self.handles.append(module.register_forward_hook(post_hook))

    def wrap_method(self, obj, method_name, timer_name):
        """
        Temporarily replace an instance method with an output-preserving wrapper.

        The original bound method is restored by remove().
        """
        original = getattr(obj, method_name)
        self.original_methods.append((obj, method_name, original))

        def timed_method(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)

            start = self._event()
            start.record()
            output = original(*args, **kwargs)
            end = self._event()
            end.record()
            self.add_event_pair(timer_name, start, end)
            return output

        setattr(obj, method_name, timed_method)

    def remove(self):
        self.enabled = False

        for handle in self.handles:
            handle.remove()
        self.handles = []

        # Restore methods in reverse order in case wrappers were nested.
        for obj, method_name, original in reversed(self.original_methods):
            setattr(obj, method_name, original)
        self.original_methods = []


def install_timers(raw_model, timer):
    """
    Install all requested component timers.

    SparseDrive structure used here:
        raw_model.img_backbone
        raw_model.img_neck
        raw_model.head
            .det_head
            .map_head
            .motion_plan_head
            .post_process(...)
        raw_model.extract_feat(...)
    """
    timer.register_module("resnet_ms", raw_model.img_backbone)

    neck = getattr(raw_model, "img_neck", None)
    timer.register_module("fpn_ms", neck)

    head = raw_model.head
    timer.register_module("head_total_ms", head)

    if hasattr(head, "det_head"):
        timer.register_module("det_ms", head.det_head)

    if hasattr(head, "map_head"):
        timer.register_module("map_ms", head.map_head)

    if hasattr(head, "motion_plan_head"):
        timer.register_module(
            "motion_plan_ms",
            head.motion_plan_head,
        )

    # extract_feat is a method rather than a child module.
    timer.wrap_method(
        raw_model,
        "extract_feat",
        "extract_feat_total_ms",
    )

    # post_process is also a normal method and is called from simple_test.
    timer.wrap_method(
        head,
        "post_process",
        "postprocess_ms",
    )


def derive_times(component_times, total_ms):
    """
    Fill absent modules with zero and derive orchestration/formatting overhead.

    These derived CUDA quantities are computed from nested event intervals:
        extract_other =
            extract_feat_total - resnet - fpn

        head_other =
            head_total - det - map - motion_plan

    Clamp at zero to absorb tiny CUDA-event resolution/numerical differences.
    """
    resnet = float(component_times.get("resnet_ms", 0.0))
    fpn = float(component_times.get("fpn_ms", 0.0))
    extract_total = float(
        component_times.get("extract_feat_total_ms", 0.0)
    )

    det = float(component_times.get("det_ms", 0.0))
    map_ = float(component_times.get("map_ms", 0.0))
    motion = float(component_times.get("motion_plan_ms", 0.0))
    head_total = float(component_times.get("head_total_ms", 0.0))
    post = float(component_times.get("postprocess_ms", 0.0))

    extract_other = max(0.0, extract_total - resnet - fpn)
    head_other = max(0.0, head_total - det - map_ - motion)

    return {
        "resnet_ms": resnet,
        "fpn_ms": fpn,
        "extract_other_ms": extract_other,
        "extract_feat_total_ms": extract_total,
        "det_ms": det,
        "map_ms": map_,
        "motion_plan_ms": motion,
        "head_other_ms": head_other,
        "head_total_ms": head_total,
        "postprocess_ms": post,
        "total_ms": float(total_ms),
        "fps": 1000.0 / float(total_ms) if total_ms > 0 else 0.0,
    }


def percentile(values, q):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def build_summary(rows):
    summary = {}

    for key in COMPONENT_COLUMNS:
        values = [float(row[key]) for row in rows]
        arr = np.asarray(values, dtype=np.float64)

        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "p50": percentile(arr, 50),
            "p90": percentile(arr, 90),
            "p95": percentile(arr, 95),
            "max": float(arr.max()),
        }

    return summary


def write_per_frame_csv(rows, out_file):
    fieldnames = [
        "measurement_idx",
        "dataset_frame_idx",
    ] + COMPONENT_COLUMNS

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(summary, out_file):
    fieldnames = [
        "component",
        "mean",
        "std",
        "min",
        "p50",
        "p90",
        "p95",
        "max",
    ]

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for component in COMPONENT_COLUMNS:
            row = {"component": component}
            row.update(summary[component])
            writer.writerow(row)


def write_summary_txt(
    summary,
    rows,
    args,
    out_file,
    device_name,
):
    avg_total = summary["total_ms"]["mean"]
    synchronized_fps = 1000.0 / avg_total if avg_total > 0 else 0.0

    # Exclusive-ish top-level breakdown for readability. Do NOT sum nested
    # totals such as resnet + extract_feat_total or det + head_total.
    breakdown_keys = [
        "resnet_ms",
        "fpn_ms",
        "extract_other_ms",
        "det_ms",
        "map_ms",
        "motion_plan_ms",
        "head_other_ms",
        "postprocess_ms",
    ]

    lines = []
    lines.append("SparseDrive Latency Benchmark")
    lines.append("=" * 72)
    lines.append("config            : {}".format(args.config))
    lines.append("checkpoint        : {}".format(args.checkpoint))
    lines.append("GPU               : {}".format(device_name))
    lines.append("measured frames   : {}".format(len(rows)))
    lines.append("warm-up frames    : {}".format(args.warmup_frames))
    lines.append("batch size        : 1")
    lines.append("")
    lines.append(
        "total_ms definition: synchronized model(...) wall time; dataloader "
        "iteration excluded"
    )
    lines.append(
        "component definition: CUDA-event GPU elapsed time; hooks add no "
        "inter-component synchronization"
    )
    lines.append("")
    lines.append("Average component latency")
    lines.append("-" * 72)

    for key in breakdown_keys:
        stat = summary[key]
        pct = (
            100.0 * stat["mean"] / avg_total
            if avg_total > 0
            else 0.0
        )
        lines.append(
            "{:<24s} {:>9.3f} ms   {:>6.2f}% of total".format(
                key,
                stat["mean"],
                pct,
            )
        )

    lines.append("-" * 72)
    lines.append(
        "{:<24s} {:>9.3f} ms".format(
            "extract_feat_total_ms",
            summary["extract_feat_total_ms"]["mean"],
        )
    )
    lines.append(
        "{:<24s} {:>9.3f} ms".format(
            "head_total_ms",
            summary["head_total_ms"]["mean"],
        )
    )
    lines.append(
        "{:<24s} {:>9.3f} ms".format(
            "TOTAL",
            avg_total,
        )
    )
    lines.append(
        "{:<24s} {:>9.3f} FPS".format(
            "throughput",
            synchronized_fps,
        )
    )

    lines.append("")
    lines.append("Total latency distribution")
    lines.append("-" * 72)
    total = summary["total_ms"]
    for label in ["mean", "std", "min", "p50", "p90", "p95", "max"]:
        lines.append(
            "{:<8s}: {:>9.3f} ms".format(
                label,
                total[label],
            )
        )

    lines.append("")
    lines.append(
        "Note: extract_feat_total_ms and head_total_ms are nested totals. "
        "Do not add them again to their child component times."
    )
    lines.append(
        "Note: total_ms is synchronized wall latency while component values "
        "are CUDA-event GPU times, so their sum need not equal total_ms exactly."
    )
    lines.append("")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_json(summary, rows, args, out_file, device_name):
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": device_name,
        "num_measured_frames": len(rows),
        "warmup_frames": args.warmup_frames,
        "batch_size": 1,
        "summary": summary,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be > 0")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SparseDrive latency testing.")

    torch.cuda.set_device(args.gpu_id)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    import_plugin(cfg, args.config)

    if cfg.get("cudnn_benchmark", False) and not args.deterministic:
        torch.backends.cudnn.benchmark = True

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

    required_dataset_frames = args.warmup_frames + args.num_frames
    if len(dataset) < required_dataset_frames:
        raise RuntimeError(
            "Dataset has only {} frames, but warmup + measured requires {}."
            .format(len(dataset), required_dataset_frames)
        )

    data_loader = build_dataloader_origin(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    cfg.model.train_cfg = None
    raw_model = build_detector(
        cfg.model,
        test_cfg=cfg.get("test_cfg"),
    )

    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(raw_model)

    checkpoint = load_checkpoint(
        raw_model,
        args.checkpoint,
        map_location="cpu",
    )

    if "CLASSES" in checkpoint.get("meta", {}):
        raw_model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        raw_model.CLASSES = dataset.CLASSES

    # Install runtime-only timers before wrapping with MMDataParallel.
    timer = ComponentTimer()
    install_timers(raw_model, timer)

    raw_model = raw_model.cuda(args.gpu_id)
    model = MMDataParallel(
        raw_model,
        device_ids=[args.gpu_id],
    )
    model.eval()

    mmcv.mkdir_or_exist(args.out_dir)

    per_frame_csv = osp.join(
        args.out_dir,
        "latency_per_frame.csv",
    )
    summary_csv = osp.join(
        args.out_dir,
        "latency_summary.csv",
    )
    summary_txt = osp.join(
        args.out_dir,
        "latency_summary.txt",
    )
    summary_json = osp.join(
        args.out_dir,
        "latency_summary.json",
    )

    device_name = torch.cuda.get_device_name(args.gpu_id)

    print("[device]", device_name)
    print("[warmup] {} sequential frame(s)".format(args.warmup_frames))
    print("[measure] {} sequential frame(s)".format(args.num_frames))
    print("[output]", osp.abspath(args.out_dir))

    rows = []
    measurement_idx = 0

    try:
        for dataset_frame_idx, data in enumerate(data_loader):
            if dataset_frame_idx >= required_dataset_frames:
                break

            if dataset_frame_idx < args.warmup_frames:
                # Warm-up remains sequential so SparseDrive temporal instance
                # memory is in a realistic state when measurement begins.
                with torch.no_grad():
                    _ = model(
                        return_loss=False,
                        rescale=True,
                        **data
                    )

                # Synchronize warm-up iterations so compilation/caches are
                # completed before the measured interval begins.
                torch.cuda.synchronize(args.gpu_id)

                print(
                    "[warmup {}/{}] dataset frame {}".format(
                        dataset_frame_idx + 1,
                        args.warmup_frames,
                        dataset_frame_idx,
                    )
                )
                continue

            # -----------------------------------------------------------------
            # Measure one complete frame.
            # Dataloader fetch time happened before this point and is excluded.
            # -----------------------------------------------------------------
            torch.cuda.synchronize(args.gpu_id)

            timer.begin_frame()
            wall_start = time.perf_counter()

            with torch.no_grad():
                _ = model(
                    return_loss=False,
                    rescale=True,
                    **data
                )

            torch.cuda.synchronize(args.gpu_id)
            wall_end = time.perf_counter()

            component_times = timer.end_frame()
            total_ms = (wall_end - wall_start) * 1000.0
            measured = derive_times(component_times, total_ms)

            row = {
                "measurement_idx": measurement_idx,
                "dataset_frame_idx": dataset_frame_idx,
            }
            row.update(measured)
            rows.append(row)

            print(
                "[{}/{}] frame={} | "
                "ResNet={:.2f} FPN={:.2f} Det={:.2f} "
                "Map={:.2f} MotionPlan={:.2f} Post={:.2f} "
                "| Total={:.2f} ms ({:.2f} FPS)".format(
                    measurement_idx + 1,
                    args.num_frames,
                    dataset_frame_idx,
                    measured["resnet_ms"],
                    measured["fpn_ms"],
                    measured["det_ms"],
                    measured["map_ms"],
                    measured["motion_plan_ms"],
                    measured["postprocess_ms"],
                    measured["total_ms"],
                    measured["fps"],
                )
            )

            measurement_idx += 1

            if measurement_idx >= args.num_frames:
                break

    finally:
        timer.remove()

    if len(rows) != args.num_frames:
        raise RuntimeError(
            "Expected {} measured frames, got {}."
            .format(args.num_frames, len(rows))
        )

    summary = build_summary(rows)

    write_per_frame_csv(rows, per_frame_csv)
    write_summary_csv(summary, summary_csv)
    write_summary_txt(
        summary,
        rows,
        args,
        summary_txt,
        device_name,
    )
    write_json(
        summary,
        rows,
        args,
        summary_json,
        device_name,
    )

    avg = summary["total_ms"]["mean"]
    avg_fps = 1000.0 / avg if avg > 0 else 0.0

    print("")
    print("=" * 72)
    print("Average latency over {} measured frames".format(len(rows)))
    print("=" * 72)
    print("ResNet           : {:9.3f} ms".format(
        summary["resnet_ms"]["mean"]
    ))
    print("FPN              : {:9.3f} ms".format(
        summary["fpn_ms"]["mean"]
    ))
    print("Extract other    : {:9.3f} ms".format(
        summary["extract_other_ms"]["mean"]
    ))
    print("Detection        : {:9.3f} ms".format(
        summary["det_ms"]["mean"]
    ))
    print("Map              : {:9.3f} ms".format(
        summary["map_ms"]["mean"]
    ))
    print("Motion + Plan    : {:9.3f} ms".format(
        summary["motion_plan_ms"]["mean"]
    ))
    print("Head other       : {:9.3f} ms".format(
        summary["head_other_ms"]["mean"]
    ))
    print("Post-process     : {:9.3f} ms".format(
        summary["postprocess_ms"]["mean"]
    ))
    print("-" * 72)
    print("Extract total    : {:9.3f} ms".format(
        summary["extract_feat_total_ms"]["mean"]
    ))
    print("Head total       : {:9.3f} ms".format(
        summary["head_total_ms"]["mean"]
    ))
    print("TOTAL            : {:9.3f} ms".format(avg))
    print("FPS              : {:9.3f}".format(avg_fps))
    print("=" * 72)
    print("[saved]", per_frame_csv)
    print("[saved]", summary_csv)
    print("[saved]", summary_txt)
    print("[saved]", summary_json)


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()