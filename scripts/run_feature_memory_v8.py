#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_downloaded_official_experiments import Experiment, downloaded_experiments, quoted_override, shell_join


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


@dataclass(frozen=True)
class Trial:
    name: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class Task:
    experiment: Experiment
    trial: Trial
    result_file: Path
    result_dir: Path
    log_file: Path

    @property
    def name(self) -> str:
        return f"{self.experiment.dataset}/{self.experiment.category}/{self.trial.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--output-root", default="outputs/feature_memory_v8")
    parser.add_argument("--log-root", default="outputs/logs/feature_memory_v8")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--skip-running", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def trials() -> list[Trial]:
    return [
        Trial(
            "fm448_z_replace_tk002",
            {
                "data.image_size": 448,
                "train.batch_size": 1,
                "train.epochs": 1,
                "graph.use_mask_topology": "false",
                "graph.beta_m": 0.0,
                "energy.alpha": 0.0,
                "energy.beta": 1.0,
                "energy.gamma": 0.0,
                "energy.eta": 0.0,
                "energy.topk_ratio": 0.002,
                "energy.upsample_mode": "bilinear",
                "data.mask_resize_mode": "nearest",
                "eval.apply_foreground_mask": "false",
                "eval.score_map_normalization": "none",
                "eval.score_map_smoothing_kernel": 0,
                "eval.feature_memory.enabled": "true",
                "eval.feature_memory.feature": "z",
                "eval.feature_memory.fusion": "replace",
                "eval.feature_memory.weight": 1.0,
                "eval.feature_memory.k": 1,
                "eval.feature_memory.max_patches": 40000,
                "eval.feature_memory.chunk_size": 1024,
                "eval.feature_memory.normalize": "true",
                "eval.feature_memory.normalize_maps": "true",
            },
        )
    ]


def completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("completed"))


def running_command_text() -> str:
    try:
        result = subprocess.run(["ps", "-eo", "cmd"], check=False, capture_output=True, text=True)
    except Exception:
        return ""
    return result.stdout


def is_running(task: Task, command_text: str | None = None) -> bool:
    if command_text is None:
        command_text = running_command_text()
    return f"eval.result_dir={task.result_dir}" in command_text


def command(args: argparse.Namespace, task: Task) -> list[str]:
    overrides = [
        "device=" + args.device,
        "data.few_shot=full",
        "data.robustness=none",
        "data.test_split_fraction=0.5",
        "data.test_split_role=tune",
        "data.root=" + str(task.experiment.root),
        quoted_override("data.category", task.experiment.category),
        "data.part_mask_root=" + str(task.experiment.part_mask_root),
        "eval.result_dir=" + str(task.result_dir),
    ]
    overrides.extend(f"{key}={value}" for key, value in task.trial.overrides.items())
    return [sys.executable, "scripts/evaluate_feature_memory.py", "--config", args.config, "--set", *overrides]


def build_tasks(args: argparse.Namespace) -> list[Task]:
    selected = set(args.datasets or [])
    experiments = downloaded_experiments()
    if selected:
        experiments = [experiment for experiment in experiments if experiment.dataset in selected]
    tasks: list[Task] = []
    for experiment in experiments:
        for trial in trials():
            result_dir = Path(args.output_root) / "results" / experiment.dataset / trial.name / experiment.category
            tasks.append(
                Task(
                    experiment=experiment,
                    trial=trial,
                    result_file=result_dir / f"{experiment.category}_train_metrics.json",
                    result_dir=result_dir,
                    log_file=Path(args.log_root) / experiment.dataset / trial.name / f"{experiment.category}.log",
                )
            )
    return tasks


def launch(args: argparse.Namespace, task: Task) -> tuple[subprocess.Popen, object]:
    task.log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = task.log_file.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = command(args, task)
    print(f"[feature-memory-v8] START {task.name} log={task.log_file}", flush=True)
    print("+ " + shell_join(cmd), file=handle, flush=True)
    return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env), handle


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
    summary_path = output_root / "all_results.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "category", "trial", *METRIC_KEYS, "path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[feature-memory-v8] wrote {summary_path}", flush=True)


def main() -> None:
    args = parse_args()
    tasks = build_tasks(args)
    command_text = running_command_text() if args.skip_running else ""
    pending = [
        task
        for task in tasks
        if not completed(task.result_file)
        and not (args.skip_running and is_running(task, command_text))
    ]
    print(
        f"[feature-memory-v8] total={len(tasks)} skipped={len(tasks) - len(pending)} "
        f"pending={len(pending)} max_parallel={args.max_parallel}",
        flush=True,
    )
    if args.dry_run:
        for task in pending:
            print("+ " + shell_join(command(args, task)), flush=True)
        return
    running: list[tuple[Task, subprocess.Popen, object]] = []
    while pending or running:
        while pending and len(running) < max(1, int(args.max_parallel)):
            task = pending.pop(0)
            if completed(task.result_file) or (args.skip_running and is_running(task)):
                print(f"[feature-memory-v8] SKIP {task.name}", flush=True)
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
                print(f"[feature-memory-v8] FAIL {task.name} exit={code} log={task.log_file}", flush=True)
                raise SystemExit(code)
            print(f"[feature-memory-v8] DONE {task.name}", flush=True)
        running = next_running
        if pending or running:
            time.sleep(5)
    summarize(tasks, Path(args.output_root))


if __name__ == "__main__":
    main()
