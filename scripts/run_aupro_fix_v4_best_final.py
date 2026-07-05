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

from run_aupro_fix_trials import Trial, completed, is_running, running_command_text
from run_aupro_fix_v4_trials import shell_join, v4_trials
from run_downloaded_official_experiments import Experiment, downloaded_experiments, quoted_override


@dataclass(frozen=True)
class Task:
    experiment: Experiment
    trial: Trial
    result_file: Path
    result_dir: Path
    checkpoint_dir: Path
    log_file: Path

    @property
    def name(self) -> str:
        return f"{self.experiment.dataset}/{self.experiment.category}/{self.trial.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/aupro_fix_v4/best_tune_summary.csv")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="*", default=["mvtec_ad_2", "mvtec_loco"])
    parser.add_argument("--output-tag", default="aupro_fix_v4_best_final")
    parser.add_argument("--log-root", default="outputs/logs/aupro_fix_v4_best_final")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--min-epochs", type=int, default=45)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--skip-running", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_best_trials(path: Path) -> dict[tuple[str, str], Trial]:
    by_name = {trial.name: trial for trial in v4_trials()}
    best: dict[tuple[str, str], Trial] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = str(row["dataset"])
            category = str(row["category"])
            trial_name = str(row.get("best_trial") or row.get("trial") or "")
            if trial_name not in by_name:
                raise KeyError(f"Unknown v4 trial {trial_name!r} in {path}")
            best[(dataset, category)] = by_name[trial_name]
    return best


def build_tasks(args: argparse.Namespace) -> list[Task]:
    selected = set(args.datasets or [])
    best_trials = load_best_trials(Path(args.summary))
    experiments = [
        experiment
        for experiment in downloaded_experiments()
        if experiment.dataset in selected and (experiment.dataset, experiment.category) in best_trials
    ]
    if not experiments:
        raise RuntimeError(f"No experiments matched datasets={sorted(selected)} and summary={args.summary}")

    tasks: list[Task] = []
    for experiment in experiments:
        trial = best_trials[(experiment.dataset, experiment.category)]
        result_dir = Path("outputs") / f"results_{args.output_tag}" / experiment.dataset / experiment.category
        checkpoint_dir = Path("outputs") / f"checkpoints_{args.output_tag}" / experiment.dataset / experiment.category
        tasks.append(
            Task(
                experiment=experiment,
                trial=trial,
                result_file=result_dir / f"{experiment.category}_train_metrics.json",
                result_dir=result_dir,
                checkpoint_dir=checkpoint_dir,
                log_file=Path(args.log_root) / experiment.dataset / f"{experiment.category}.log",
            )
        )
    return tasks


def train_command(args: argparse.Namespace, task: Task) -> list[str]:
    overrides = [
        "device=" + args.device,
        "data.few_shot=full",
        "data.robustness=none",
        "data.test_split_fraction=0.5",
        "data.test_split_role=final",
        "train.resume=true",
        "train.epochs=" + str(args.epochs),
        "train.early_stopping.enabled=true",
        "train.early_stopping.patience=" + str(args.patience),
        "train.early_stopping.min_epochs=" + str(args.min_epochs),
        "graph.use_mask_topology=true",
        "graph.beta_m=1.0",
        "data.root=" + str(task.experiment.root),
        quoted_override("data.category", task.experiment.category),
        "data.part_mask_root=" + str(task.experiment.part_mask_root),
        "train.save_dir=" + str(task.checkpoint_dir),
        "eval.result_dir=" + str(task.result_dir),
    ]
    overrides.extend(f"{key}={value}" for key, value in task.trial.overrides.items())
    return [sys.executable, "scripts/train.py", "--config", args.config, "--set", *overrides]


def launch(args: argparse.Namespace, task: Task) -> tuple[subprocess.Popen, object]:
    task.log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = task.log_file.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command = train_command(args, task)
    print(f"[aupro-v4-final] START {task.name} log={task.log_file}", flush=True)
    print("+ " + shell_join(command), file=handle, flush=True)
    return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env), handle


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
        f"[aupro-v4-final] total={len(tasks)} skipped={len(tasks) - len(pending)} "
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
            task = pending.pop(0)
            if completed(task.result_file) or (args.skip_running and is_running(task)):
                print(f"[aupro-v4-final] SKIP {task.name}", flush=True)
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
                    print(f"[aupro-v4-final] RETRY {task.name} exit={code}", flush=True)
                    pending.insert(0, task)
                    continue
                print(f"[aupro-v4-final] FAIL {task.name} exit={code} log={task.log_file}", flush=True)
                raise SystemExit(code)
            print(f"[aupro-v4-final] DONE {task.name}", flush=True)
        running = next_running
        if pending or running:
            time.sleep(5)


def main() -> None:
    run_tasks(parse_args())


if __name__ == "__main__":
    main()
