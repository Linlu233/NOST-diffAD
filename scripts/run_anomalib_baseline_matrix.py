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

from run_downloaded_official_experiments import Experiment, downloaded_experiments, shell_join


BASELINES = [
    "patchcore",
    "padim",
    "fastflow",
    "reverse_distillation",
    "stfpm",
    "cfa",
    "cflow",
    "dfkde",
    "dinomaly",
    "winclip",
]


@dataclass(frozen=True)
class Task:
    method: str
    experiment: Experiment
    result_file: Path
    log_file: Path
    work_dir: Path

    @property
    def name(self) -> str:
        return f"{self.method}/{self.experiment.dataset}/{self.experiment.category}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/baselines_anomalib")
    parser.add_argument("--log-root", default="outputs/logs/baselines_anomalib")
    parser.add_argument("--methods", nargs="*", default=BASELINES)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--representative-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=int, default=30)
    parser.add_argument("--skip-finished", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def result_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("completed"))


def representative(experiments: list[Experiment]) -> list[Experiment]:
    by_dataset: dict[str, list[Experiment]] = {}
    for experiment in experiments:
        by_dataset.setdefault(experiment.dataset, []).append(experiment)
    return [sorted(items, key=lambda item: item.category)[0] for _, items in sorted(by_dataset.items())]


def build_tasks(args: argparse.Namespace) -> list[Task]:
    experiments = downloaded_experiments()
    if args.datasets:
        wanted = set(args.datasets)
        experiments = [item for item in experiments if item.dataset in wanted]
    experiments = sorted(experiments, key=lambda item: (item.dataset, item.category))
    if args.representative_only:
        experiments = representative(experiments)
    tasks: list[Task] = []
    for method in args.methods:
        for experiment in experiments:
            result_dir = Path(args.output_root) / "results" / method / experiment.dataset / experiment.category
            result_file = result_dir / f"{experiment.category}_train_metrics.json"
            log_file = Path(args.log_root) / method / experiment.dataset / f"{experiment.category}.log"
            work_dir = Path(args.output_root) / "work" / method / experiment.dataset / experiment.category
            tasks.append(Task(method, experiment, result_file, log_file, work_dir))
    return tasks


def command(args: argparse.Namespace, task: Task) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_anomalib_baseline_task.py",
        "--method",
        task.method,
        "--dataset",
        task.experiment.dataset,
        "--category",
        task.experiment.category,
        "--root",
        str(task.experiment.root),
        "--result-dir",
        str(task.result_file.parent),
        "--work-dir",
        str(task.work_dir),
        "--device",
        args.device,
        "--image-size",
        str(args.image_size),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.max_epochs is not None:
        cmd += ["--max-epochs", str(args.max_epochs)]
    return cmd


def write_manifest(args: argparse.Namespace, tasks: list[Task]) -> None:
    selected_methods = [method.lower() for method in args.methods]
    is_full_matrix = (
        set(selected_methods) == set(BASELINES)
        and args.datasets is None
        and not args.representative_only
    )
    if is_full_matrix:
        path = Path(args.output_root) / "baseline_manifest.csv"
    else:
        method_slug = "_".join(selected_methods)
        if len(method_slug) > 96:
            method_slug = f"{len(selected_methods)}methods"
        path = Path(args.output_root) / f"baseline_manifest_{method_slug}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "dataset", "category", "root", "result_file", "log_file", "work_dir"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "method": task.method,
                    "dataset": task.experiment.dataset,
                    "category": task.experiment.category,
                    "root": str(task.experiment.root),
                    "result_file": str(task.result_file),
                    "log_file": str(task.log_file),
                    "work_dir": str(task.work_dir),
                }
            )


def run_tasks(args: argparse.Namespace, tasks: list[Task]) -> None:
    write_manifest(args, tasks)
    pending = [task for task in tasks if not args.skip_finished or not result_completed(task.result_file)]
    print(f"[baseline-matrix] total={len(tasks)} pending={len(pending)} max_parallel={args.max_parallel}", flush=True)
    if args.dry_run:
        for task in pending:
            print("+ " + shell_join(command(args, task)), flush=True)
        return
    queue: list[tuple[Task, int]] = [(task, 1) for task in pending]
    running: list[tuple[Task, int, subprocess.Popen, object]] = []
    failures: list[tuple[Task, int]] = []
    while queue or running:
        while queue and len(running) < max(1, int(args.max_parallel)):
            task, attempt = queue.pop(0)
            task.result_file.parent.mkdir(parents=True, exist_ok=True)
            task.log_file.parent.mkdir(parents=True, exist_ok=True)
            task.work_dir.mkdir(parents=True, exist_ok=True)
            handle = task.log_file.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            env.setdefault("HF_ENDPOINT", args.hf_endpoint)
            cmd = command(args, task)
            print(f"[baseline-matrix] START {task.name} attempt={attempt} log={task.log_file}", flush=True)
            print(f"# attempt={attempt}", file=handle, flush=True)
            print("+ " + shell_join(cmd), file=handle, flush=True)
            running.append((task, attempt, subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env), handle))
        next_running: list[tuple[Task, int, subprocess.Popen, object]] = []
        for task, attempt, process, handle in running:
            code = process.poll()
            if code is None:
                next_running.append((task, attempt, process, handle))
                continue
            try:
                handle.close()
            except Exception:
                pass
            if code == 0 and result_completed(task.result_file):
                print(f"[baseline-matrix] DONE {task.name}", flush=True)
            else:
                print(f"[baseline-matrix] FAIL {task.name} attempt={attempt} code={code}", flush=True)
                if attempt < args.max_retries:
                    time.sleep(max(0, args.retry_backoff_seconds))
                    queue.append((task, attempt + 1))
                else:
                    failures.append((task, attempt))
        running = next_running
        if running or queue:
            time.sleep(5)
    fail_file = Path(args.output_root) / "failed_tasks.txt"
    if failures:
        fail_file.parent.mkdir(parents=True, exist_ok=True)
        with fail_file.open("w", encoding="utf-8") as handle:
            for task, attempt in failures:
                handle.write(f"{task.name}\tattempts={attempt}\tlog={task.log_file}\n")
        if not args.continue_on_fail:
            raise SystemExit(1)
    elif fail_file.exists():
        fail_file.unlink()


def main() -> None:
    args = parse_args()
    tasks = build_tasks(args)
    run_tasks(args, tasks)


if __name__ == "__main__":
    main()
