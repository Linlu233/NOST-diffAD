#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from nostdiffad.config import load_yaml
from run_downloaded_official_experiments import Experiment, downloaded_experiments, shell_join
from tune_downloaded_datasets import (
    Trial,
    dataset_groups,
    ensure_masks_command,
    representative_experiments,
    summarize,
    train_command,
    trials,
)


@dataclass(frozen=True)
class TuningTask:
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
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--output-root", default="outputs/hparam_tuning")
    parser.add_argument("--log-root", default="outputs/logs/hparam_tuning_trials")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--tune-test-fraction", type=float, default=0.5)
    parser.add_argument("--selection-metric", default="selection_score")
    parser.add_argument("--trial-names", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--skip-running", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-gpu-memory-used-mb", type=int, default=None)
    parser.add_argument("--max-gpu-processes", type=int, default=None)
    parser.add_argument("--gpu-budget-poll-seconds", type=float, default=30.0)
    parser.add_argument("--launch-delay-seconds", type=float, default=0.0)
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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
        result = subprocess.run(
            ["ps", "-eo", "cmd"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout


def gpu_budget_state() -> tuple[int | None, int | None]:
    try:
        memory = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None, None
    used_values = []
    for line in memory.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            used_values.append(int(line))
        except ValueError:
            continue
    app_pids = {line.strip() for line in apps.stdout.splitlines() if line.strip()}
    return (max(used_values) if used_values else None), len(app_pids)


def gpu_budget_available(args: argparse.Namespace) -> bool:
    used_mb, process_count = gpu_budget_state()
    if args.max_gpu_memory_used_mb is not None and used_mb is not None:
        if used_mb >= int(args.max_gpu_memory_used_mb):
            return False
    if args.max_gpu_processes is not None and process_count is not None:
        if process_count >= int(args.max_gpu_processes):
            return False
    return True


def wait_for_gpu_budget(args: argparse.Namespace) -> None:
    if args.max_gpu_memory_used_mb is None and args.max_gpu_processes is None:
        return
    while not gpu_budget_available(args):
        used_mb, process_count = gpu_budget_state()
        print(
            "[parallel-tune] WAIT gpu_budget "
            f"used_mb={used_mb} max_used_mb={args.max_gpu_memory_used_mb} "
            f"gpu_processes={process_count} max_gpu_processes={args.max_gpu_processes}",
            flush=True,
        )
        time.sleep(max(1.0, float(args.gpu_budget_poll_seconds)))


def is_running(task: TuningTask, command_text: str | None = None) -> bool:
    if command_text is None:
        command_text = running_command_text()
    result_marker = "eval.result_dir=" + str(task.result_dir)
    checkpoint_marker = "train.save_dir=" + str(task.checkpoint_dir)
    return result_marker in command_text or checkpoint_marker in command_text


def build_tasks(args: argparse.Namespace, experiments: list[Experiment], trial_list: list[Trial]) -> list[TuningTask]:
    output_root = Path(args.output_root)
    log_root = Path(args.log_root)
    tasks: list[TuningTask] = []
    for experiment in representative_experiments(dataset_groups(experiments)):
        for trial in trial_list:
            result_dir = output_root / "results" / experiment.dataset / trial.name / experiment.category
            checkpoint_dir = output_root / "checkpoints" / experiment.dataset / trial.name / experiment.category
            result_file = result_dir / f"{experiment.category}_train_metrics.json"
            log_file = log_root / experiment.dataset / trial.name / f"{experiment.category}.log"
            tasks.append(TuningTask(experiment, trial, result_file, result_dir, checkpoint_dir, log_file))
    return tasks


def filter_experiments(args: argparse.Namespace, experiments: list[Experiment]) -> list[Experiment]:
    if not args.datasets:
        return experiments
    selected = set(args.datasets)
    return [experiment for experiment in experiments if experiment.dataset in selected]


def filter_trials(args: argparse.Namespace, trial_list: list[Trial]) -> list[Trial]:
    if not args.trial_names:
        return trial_list
    selected = set(args.trial_names)
    filtered = [trial for trial in trial_list if trial.name in selected]
    missing = sorted(selected - {trial.name for trial in filtered})
    if missing:
        raise ValueError(f"Unknown trial names: {', '.join(missing)}")
    return filtered


def run_mask_generation(args: argparse.Namespace, experiments: list[Experiment]) -> None:
    for experiment in representative_experiments(dataset_groups(experiments)):
        command = ensure_masks_command(args, experiment)
        if command is None:
            print(f"[parallel-tune] masks ready {experiment.dataset}/{experiment.category}", flush=True)
            continue
        print(f"[parallel-tune] generating masks {experiment.dataset}/{experiment.category}", flush=True)
        print("+ " + shell_join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


def launch(task: TuningTask, args: argparse.Namespace) -> tuple[subprocess.Popen, object]:
    task.log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = task.log_file.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    command = train_command(args, task.experiment, task.trial, task.result_dir, task.checkpoint_dir)
    print(f"[parallel-tune] START {task.name} log={task.log_file}", flush=True)
    print("+ " + shell_join(command), file=handle, flush=True)
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return process, handle


def terminate_running(running: list[tuple[TuningTask, subprocess.Popen, object]]) -> None:
    for task, process, handle in running:
        if process.poll() is None:
            print(f"[parallel-tune] TERM {task.name}", flush=True)
            process.terminate()
        handle.close()


def run_tasks(args: argparse.Namespace, tasks: list[TuningTask]) -> None:
    command_text = running_command_text() if args.skip_running else ""
    pending = [
        task
        for task in tasks
        if not completed(task.result_file)
        and not (args.skip_running and is_running(task, command_text))
    ]
    skipped = len(tasks) - len(pending)
    print(
        f"[parallel-tune] total={len(tasks)} skipped_completed={skipped} pending={len(pending)} "
        f"max_parallel={args.max_parallel}",
        flush=True,
    )
    if args.dry_run:
        for task in pending:
            command = train_command(args, task.experiment, task.trial, task.result_dir, task.checkpoint_dir)
            print("+ " + shell_join(command), flush=True)
        return

    running: list[tuple[TuningTask, subprocess.Popen, object]] = []
    attempts: dict[str, int] = {}
    try:
        while pending or running:
            while pending and len(running) < max(1, int(args.max_parallel)):
                task = pending.pop(0)
                if completed(task.result_file):
                    print(f"[parallel-tune] SKIP completed {task.name}", flush=True)
                    continue
                if args.skip_running and is_running(task):
                    print(f"[parallel-tune] SKIP running {task.name}", flush=True)
                    continue
                wait_for_gpu_budget(args)
                process, handle = launch(task, args)
                running.append((task, process, handle))
                if float(args.launch_delay_seconds) > 0:
                    time.sleep(float(args.launch_delay_seconds))

            next_running: list[tuple[TuningTask, subprocess.Popen, object]] = []
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
                        print(
                            f"[parallel-tune] RETRY {task.name} exit={code} "
                            f"attempt={attempt + 1}/{args.retries} log={task.log_file}",
                            flush=True,
                        )
                        pending.insert(0, task)
                        continue
                    print(f"[parallel-tune] FAIL {task.name} exit={code} log={task.log_file}", flush=True)
                    terminate_running(next_running)
                    raise SystemExit(code)
                print(f"[parallel-tune] DONE {task.name}", flush=True)
            running = next_running
            if pending or running:
                time.sleep(5)
    except KeyboardInterrupt:
        terminate_running(running)
        raise


def main() -> None:
    args = parse_args()
    base_config = load_yaml(args.config)
    experiments = filter_experiments(args, downloaded_experiments())
    if not experiments:
        raise RuntimeError("No downloaded supported datasets found under datasets/.")
    trial_list = filter_trials(args, trials())
    run_mask_generation(args, experiments)
    tasks = build_tasks(args, experiments, trial_list)
    run_tasks(args, tasks)
    if not args.dry_run and not args.no_summary:
        summarize(args, experiments, trial_list, base_config)


if __name__ == "__main__":
    main()
