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
from typing import Any

from run_downloaded_official_experiments import Experiment, downloaded_experiments, quoted_override, shell_join
from run_aupro_fix_trials import completed, is_running, running_command_text


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
    checkpoint_dir: Path
    log_file: Path

    @property
    def name(self) -> str:
        return f"{self.experiment.dataset}/{self.experiment.category}/{self.trial.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default="btad")
    parser.add_argument("--categories", nargs="*", default=["01", "02"])
    parser.add_argument("--output-root", default="outputs/aupro_backbone")
    parser.add_argument("--log-root", default="outputs/logs/aupro_backbone_trials")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--skip-running", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def trials() -> list[Trial]:
    vitb = {
        "model.feature_extractor": "dinov2_torchhub",
        "model.torchhub_model": "dinov2_vitb14",
        "model.feature_dim": 768,
        "model.train_feature_extractor": "false",
    }
    vitb_reg = {
        "model.feature_extractor": "dinov2_torchhub",
        "model.torchhub_model": "dinov2_vitb14_reg",
        "model.feature_dim": 768,
        "model.train_feature_extractor": "false",
    }
    vitl = {
        "model.feature_extractor": "dinov2_torchhub",
        "model.torchhub_model": "dinov2_vitl14",
        "model.feature_dim": 1024,
        "model.train_feature_extractor": "false",
    }
    proto = {"energy.alpha": 0.0, "energy.beta": 1.0, "energy.gamma": 0.0, "energy.eta": 0.0}
    return [
        Trial(
            "vitb14_224_proto_bilinear_aupro",
            {
                **vitb,
                **proto,
                "data.image_size": 224,
                "train.batch_size": 2,
                "energy.upsample_mode": "bilinear",
                "train.early_stopping.monitor": "au_pro",
            },
        ),
        Trial(
            "vitb14_336_proto_bilinear_aupro",
            {
                **vitb,
                **proto,
                "data.image_size": 336,
                "train.batch_size": 1,
                "energy.upsample_mode": "bilinear",
                "train.early_stopping.monitor": "au_pro",
            },
        ),
        Trial(
            "vitb14_336_proto_bicubic_aupro",
            {
                **vitb,
                **proto,
                "data.image_size": 336,
                "train.batch_size": 1,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
            },
        ),
        Trial(
            "vitb14_336_score01_proto_bilinear_aupro",
            {
                **vitb,
                "energy.alpha": 0.1,
                "energy.beta": 1.0,
                "energy.gamma": 0.0,
                "energy.eta": 0.0,
                "data.image_size": 336,
                "train.batch_size": 1,
                "energy.upsample_mode": "bilinear",
                "train.early_stopping.monitor": "au_pro",
            },
        ),
        Trial(
            "vitb14_reg_224_proto_bilinear_aupro",
            {
                **vitb_reg,
                **proto,
                "data.image_size": 224,
                "train.batch_size": 2,
                "energy.upsample_mode": "bilinear",
                "train.early_stopping.monitor": "au_pro",
            },
        ),
        Trial(
            "vitb14_reg_336_proto_bilinear_aupro",
            {
                **vitb_reg,
                **proto,
                "data.image_size": 336,
                "train.batch_size": 1,
                "energy.upsample_mode": "bilinear",
                "train.early_stopping.monitor": "au_pro",
            },
        ),
        Trial(
            "vitb14_reg_336_proto_bicubic_aupro",
            {
                **vitb_reg,
                **proto,
                "data.image_size": 336,
                "train.batch_size": 1,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
            },
        ),
        Trial(
            "vitl14_224_proto_bilinear_aupro",
            {
                **vitl,
                **proto,
                "data.image_size": 224,
                "train.batch_size": 1,
                "energy.upsample_mode": "bilinear",
                "train.early_stopping.monitor": "au_pro",
            },
        ),
        Trial(
            "vitl14_336_proto_bilinear_aupro",
            {
                **vitl,
                **proto,
                "data.image_size": 336,
                "train.batch_size": 1,
                "energy.upsample_mode": "bilinear",
                "train.early_stopping.monitor": "au_pro",
            },
        ),
        Trial(
            "vitl14_336_proto_bicubic_aupro",
            {
                **vitl,
                **proto,
                "data.image_size": 336,
                "train.batch_size": 1,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
            },
        ),
    ]


def experiments_for(args: argparse.Namespace) -> list[Experiment]:
    selected = set(args.categories or [])
    experiments = [experiment for experiment in downloaded_experiments() if experiment.dataset == args.dataset]
    if selected:
        experiments = [experiment for experiment in experiments if experiment.category in selected]
    if not experiments:
        raise RuntimeError(f"No experiments found for dataset={args.dataset!r} categories={args.categories!r}.")
    return experiments


def build_tasks(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    for experiment in experiments_for(args):
        for trial in trials():
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
    return tasks


def train_command(args: argparse.Namespace, task: Task) -> list[str]:
    overrides = [
        "device=" + args.device,
        "data.few_shot=full",
        "data.robustness=none",
        "data.test_split_fraction=0.5",
        "data.test_split_role=tune",
        "train.resume=false",
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


def launch(task: Task, args: argparse.Namespace) -> tuple[subprocess.Popen, object]:
    task.log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = task.log_file.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    command = train_command(args, task)
    print(f"[aupro-backbone] START {task.name} log={task.log_file}", flush=True)
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
        f"[aupro-backbone] total={len(tasks)} skipped={len(tasks) - len(pending)} "
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
                print(f"[aupro-backbone] SKIP {task.name}", flush=True)
                continue
            process, handle = launch(task, args)
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
                    print(f"[aupro-backbone] RETRY {task.name} exit={code}", flush=True)
                    pending.insert(0, task)
                    continue
                print(f"[aupro-backbone] FAIL {task.name} exit={code} log={task.log_file}", flush=True)
                raise SystemExit(code)
            print(f"[aupro-backbone] DONE {task.name}", flush=True)
        running = next_running
        if pending or running:
            time.sleep(5)


def main() -> None:
    args = parse_args()
    run_tasks(args)


if __name__ == "__main__":
    main()
