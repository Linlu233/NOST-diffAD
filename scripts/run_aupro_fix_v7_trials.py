#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from run_aupro_fix_trials import Task, Trial, completed, is_running, running_command_text, shell_join, train_command
from run_downloaded_official_experiments import downloaded_experiments


DEFAULT_TARGETS = {
    ("mvtec_ad_2", "can"),
    ("mvtec_ad_2", "wallplugs"),
    ("mvtec_loco", "pushpins"),
    ("mvtec_loco", "screw_bag"),
}

METRIC_KEYS = [
    "image_auroc",
    "image_ap",
    "pixel_auroc",
    "au_pro",
    "pixel_ap",
    "f1_max",
    "iou",
    "dice",
    "selection_score",
    "selection_score_pro",
    "inference_speed_fps",
]


def post_overrides(
    *,
    mask_resize: str = "nearest",
    foreground: bool = False,
    norm: str = "none",
    smooth: int = 0,
) -> dict[str, object]:
    return {
        "data.mask_resize_mode": mask_resize,
        "eval.apply_foreground_mask": str(foreground).lower(),
        "eval.score_map_normalization": norm,
        "eval.score_map_smoothing_kernel": smooth,
    }


def proto_overrides(
    *,
    size: int,
    mode: str,
    monitor: str,
    topk: float,
    mask: bool,
    mask_resize: str = "nearest",
    foreground: bool = False,
    norm: str = "none",
    smooth: int = 0,
) -> dict[str, object]:
    return {
        "energy.alpha": 0.0,
        "energy.beta": 1.0,
        "energy.gamma": 0.0,
        "energy.eta": 0.0,
        "data.image_size": size,
        "train.batch_size": 1,
        "energy.topk_ratio": topk,
        "energy.upsample_mode": mode,
        "train.early_stopping.monitor": monitor,
        "graph.use_mask_topology": str(mask).lower(),
        "graph.beta_m": 1.0 if mask else 0.0,
        **post_overrides(mask_resize=mask_resize, foreground=foreground, norm=norm, smooth=smooth),
    }


def score_overrides(
    *,
    alpha: float,
    size: int,
    mode: str,
    monitor: str,
    topk: float,
    mask: bool,
    mask_resize: str = "nearest",
    gamma: float = 0.0,
    eta: float = 0.0,
    foreground: bool = False,
    norm: str = "none",
    smooth: int = 0,
) -> dict[str, object]:
    return {
        "energy.alpha": alpha,
        "energy.beta": 1.0,
        "energy.gamma": gamma,
        "energy.eta": eta,
        "data.image_size": size,
        "train.batch_size": 1,
        "energy.topk_ratio": topk,
        "energy.upsample_mode": mode,
        "train.early_stopping.monitor": monitor,
        "graph.use_mask_topology": str(mask).lower(),
        "graph.beta_m": 1.0 if mask else 0.0,
        **post_overrides(mask_resize=mask_resize, foreground=foreground, norm=norm, smooth=smooth),
    }


def can_trials() -> list[Trial]:
    return [
        Trial("nearest_proto672_nearest_aupro_nomask_tk002", proto_overrides(size=672, mode="nearest", monitor="au_pro", topk=0.002, mask=False)),
        Trial("nearest_proto672_nearest_pap_nomask_tk002", proto_overrides(size=672, mode="nearest", monitor="pixel_ap", topk=0.002, mask=False)),
        Trial("max_proto672_nearest_aupro_nomask_tk001", proto_overrides(size=672, mode="nearest", monitor="au_pro", topk=0.001, mask=False, mask_resize="max")),
        Trial("max_proto784_nearest_aupro_nomask_tk001", proto_overrides(size=784, mode="nearest", monitor="au_pro", topk=0.001, mask=False, mask_resize="max")),
        Trial("nearest_proto784_nearest_pap_nomask_tk001", proto_overrides(size=784, mode="nearest", monitor="pixel_ap", topk=0.001, mask=False)),
        Trial("nearest_proto448_bicubic_pro_nomask_tk20", proto_overrides(size=448, mode="bicubic", monitor="selection_score_pro", topk=0.2, mask=False)),
    ]


def wallplugs_trials() -> list[Trial]:
    return [
        Trial("nearest_proto560_nearest_fg_pro_mask_tk005", proto_overrides(size=560, mode="nearest", monitor="selection_score_pro", topk=0.005, mask=True, foreground=True)),
        Trial("nearest_proto672_nearest_fg_pap_mask_tk002", proto_overrides(size=672, mode="nearest", monitor="pixel_ap", topk=0.002, mask=True, foreground=True)),
        Trial("nearest_proto672_bilinear_fg_minmax_s3_aupro_mask_tk005", proto_overrides(size=672, mode="bilinear", monitor="au_pro", topk=0.005, mask=True, foreground=True, norm="image_minmax", smooth=3)),
        Trial("max_proto560_bicubic_fg_minmax_s3_aupro_nomask_tk005", proto_overrides(size=560, mode="bicubic", monitor="au_pro", topk=0.005, mask=False, mask_resize="max", foreground=True, norm="image_minmax", smooth=3)),
        Trial("nearest_score005_proto560_nearest_fg_pro_mask_tk005", score_overrides(alpha=0.05, size=560, mode="nearest", monitor="selection_score_pro", topk=0.005, mask=True, foreground=True)),
        Trial("nearest_score005_proto672_nearest_fg_pap_mask_tk002", score_overrides(alpha=0.05, size=672, mode="nearest", monitor="pixel_ap", topk=0.002, mask=True, foreground=True)),
    ]


def pushpins_trials() -> list[Trial]:
    return [
        Trial("nearest_proto560_bilinear_pro_mask_tk05", proto_overrides(size=560, mode="bilinear", monitor="selection_score_pro", topk=0.05, mask=True)),
        Trial("nearest_proto560_bicubic_aupro_mask_tk20", proto_overrides(size=560, mode="bicubic", monitor="au_pro", topk=0.2, mask=True)),
        Trial("nearest_proto672_nearest_pap_mask_tk005", proto_overrides(size=672, mode="nearest", monitor="pixel_ap", topk=0.005, mask=True)),
        Trial("nearest_proto784_area_pap_mask_tk005", proto_overrides(size=784, mode="area", monitor="pixel_ap", topk=0.005, mask=True)),
        Trial("nearest_score005_proto560_bicubic_aupro_mask_tk20", score_overrides(alpha=0.05, size=560, mode="bicubic", monitor="au_pro", topk=0.2, mask=True)),
    ]


def screw_bag_trials() -> list[Trial]:
    return [
        Trial("nearest_proto448_bicubic_aupro_nomask_tk005", proto_overrides(size=448, mode="bicubic", monitor="au_pro", topk=0.005, mask=False)),
        Trial("nearest_proto560_bilinear_fg_pro_mask_tk05", proto_overrides(size=560, mode="bilinear", monitor="selection_score_pro", topk=0.05, mask=True, foreground=True)),
        Trial("nearest_proto672_nearest_aupro_nomask_tk002", proto_overrides(size=672, mode="nearest", monitor="au_pro", topk=0.002, mask=False)),
        Trial("nearest_proto784_area_pro_mask_tk005", proto_overrides(size=784, mode="area", monitor="selection_score_pro", topk=0.005, mask=True)),
        Trial("nearest_score005_proto560_bilinear_fg_pap_mask_tk05", score_overrides(alpha=0.05, size=560, mode="bilinear", monitor="pixel_ap", topk=0.05, mask=True, foreground=True)),
    ]


def v7_trials(dataset: str, category: str) -> list[Trial]:
    if dataset == "mvtec_ad_2" and category == "can":
        return can_trials()
    if dataset == "mvtec_ad_2" and category == "wallplugs":
        return wallplugs_trials()
    if dataset == "mvtec_loco" and category == "pushpins":
        return pushpins_trials()
    if dataset == "mvtec_loco" and category == "screw_bag":
        return screw_bag_trials()
    return []


def parse_target(value: str) -> tuple[str, str]:
    if "/" not in value:
        raise argparse.ArgumentTypeError("target must be formatted as dataset/category")
    dataset, category = value.split("/", 1)
    return dataset, category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--targets", nargs="*", type=parse_target, default=sorted(DEFAULT_TARGETS))
    parser.add_argument("--output-root", default="outputs/aupro_fix_v7")
    parser.add_argument("--log-root", default="outputs/logs/aupro_fix_v7_trials")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=32)
    parser.add_argument("--min-epochs", type=int, default=45)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--skip-running", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_tasks(args: argparse.Namespace) -> list[Task]:
    targets = set(args.targets or [])
    experiments = [
        experiment
        for experiment in downloaded_experiments()
        if (experiment.dataset, experiment.category) in targets
    ]
    tasks: list[Task] = []
    for experiment in experiments:
        for trial in v7_trials(experiment.dataset, experiment.category):
            result_dir = Path(args.output_root) / "results" / experiment.dataset / trial.name / experiment.category
            checkpoint_dir = Path(args.output_root) / "checkpoints" / experiment.dataset / trial.name / experiment.category
            tasks.append(
                Task(
                    experiment=experiment,
                    trial=trial,
                    result_file=result_dir / f"{experiment.category}_train_metrics.json",
                    result_dir=result_dir,
                    checkpoint_dir=checkpoint_dir,
                    log_file=Path(args.log_root) / experiment.dataset / trial.name / f"{experiment.category}.log",
                )
            )
    if not tasks:
        raise RuntimeError(f"No v7 tasks matched targets={sorted(targets)}")
    return tasks


def launch(args: argparse.Namespace, task: Task) -> tuple[subprocess.Popen, object]:
    task.log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = task.log_file.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command = train_command(args, task)
    print(f"[aupro-v7] START {task.name} log={task.log_file}", flush=True)
    print("+ " + shell_join(command), file=handle, flush=True)
    return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env), handle


def heavy_task(task: Task) -> bool:
    try:
        return int(task.trial.overrides.get("data.image_size", 0)) >= 784
    except (TypeError, ValueError):
        return False


def metric_value(metrics: dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def summarize(tasks: list[Task], output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        if not completed(task.result_file):
            continue
        payload = json.loads(task.result_file.read_text(encoding="utf-8"))
        metrics = payload.get("best_eval") or payload.get("latest_eval") or {}
        row = {
            "dataset": task.experiment.dataset,
            "category": task.experiment.category,
            "trial": task.trial.name,
            "path": str(task.result_file),
        }
        for key in METRIC_KEYS:
            row[key] = metric_value(metrics, key)
        rows.append(row)

    summary_path = output_root / "all_tune_results.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "category", "trial", *METRIC_KEYS, "path"])
        writer.writeheader()
        writer.writerows(rows)

    best_rows: list[dict[str, Any]] = []
    selectors = {"sel": "selection_score_pro", "aupro": "au_pro", "pap": "pixel_ap"}
    groups = sorted({(row["dataset"], row["category"]) for row in rows})
    for dataset, category in groups:
        candidates = [row for row in rows if row["dataset"] == dataset and row["category"] == category]
        for selected_by, key in selectors.items():
            best = max(candidates, key=lambda row: metric_value(row, key))
            best_rows.append({"selected_by": selected_by, **best})

    best_path = output_root / "best_tune_summary.csv"
    with best_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "category", "selected_by", "trial", *METRIC_KEYS, "path"])
        writer.writeheader()
        writer.writerows(best_rows)
    print(f"[aupro-v7] wrote {summary_path}", flush=True)
    print(f"[aupro-v7] wrote {best_path}", flush=True)


def run_tasks(args: argparse.Namespace) -> None:
    tasks = build_tasks(args)
    command_text = running_command_text() if args.skip_running else ""
    pending = [
        task
        for task in tasks
        if not completed(task.result_file)
        and not (args.skip_running and is_running(task, command_text))
    ]
    print(
        f"[aupro-v7] total={len(tasks)} skipped={len(tasks) - len(pending)} "
        f"pending={len(pending)} max_parallel={args.max_parallel}",
        flush=True,
    )
    if args.dry_run:
        for task in pending:
            print("+ " + shell_join(train_command(args, task)), flush=True)
        return

    running: list[tuple[Task, subprocess.Popen, object]] = []
    attempts: dict[str, int] = {}
    while pending or running:
        while pending and len(running) < max(1, int(args.max_parallel)):
            if any(heavy_task(task) for task, _, _ in running):
                break
            if heavy_task(pending[0]) and running:
                break
            task = pending.pop(0)
            if completed(task.result_file) or (args.skip_running and is_running(task)):
                print(f"[aupro-v7] SKIP {task.name}", flush=True)
                continue
            process, handle = launch(args, task)
            running.append((task, process, handle))

        next_running: list[tuple[Task, subprocess.Popen, object]] = []
        for task, process, handle in running:
            code = process.poll()
            if code is None:
                next_running.append((task, process, handle))
                continue
            handle.close()
            if code != 0:
                attempt = attempts.get(task.name, 0)
                if attempt < max(0, int(args.retries)):
                    attempts[task.name] = attempt + 1
                    print(f"[aupro-v7] RETRY {task.name} exit={code}", flush=True)
                    pending.insert(0, task)
                    continue
                print(f"[aupro-v7] FAIL {task.name} exit={code} log={task.log_file}", flush=True)
                raise SystemExit(code)
            print(f"[aupro-v7] DONE {task.name}", flush=True)
        running = next_running
        if pending or running:
            time.sleep(5)

    summarize(tasks, Path(args.output_root))


def main() -> None:
    run_tasks(parse_args())


if __name__ == "__main__":
    main()
