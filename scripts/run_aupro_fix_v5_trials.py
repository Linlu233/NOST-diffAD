#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_aupro_fix_trials import Task, Trial, completed, is_running, running_command_text, shell_join, train_command
from run_downloaded_official_experiments import downloaded_experiments


DEFAULT_TARGETS = {
    ("mvtec_ad_2", "can"),
    ("mvtec_ad_2", "wallplugs"),
    ("mvtec_loco", "pushpins"),
    ("mvtec_loco", "screw_bag"),
}


def v5_trials() -> list[Trial]:
    proto = {"energy.alpha": 0.0, "energy.beta": 1.0, "energy.gamma": 0.0, "energy.eta": 0.0}
    return [
        Trial(
            "proto672_bicubic_aupro_mask",
            {
                **proto,
                "data.image_size": 672,
                "train.batch_size": 1,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
                "graph.use_mask_topology": "true",
                "graph.beta_m": 1.0,
            },
        ),
        Trial(
            "proto672_bicubic_pro_mask",
            {
                **proto,
                "data.image_size": 672,
                "train.batch_size": 1,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "selection_score_pro",
                "graph.use_mask_topology": "true",
                "graph.beta_m": 1.0,
            },
        ),
        Trial(
            "proto672_bicubic_aupro_nomask_topk01",
            {
                **proto,
                "data.image_size": 672,
                "train.batch_size": 1,
                "energy.topk_ratio": 0.01,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
                "graph.use_mask_topology": "false",
                "graph.beta_m": 0.0,
            },
        ),
        Trial(
            "proto672_bicubic_aupro_nomask_topk20",
            {
                **proto,
                "data.image_size": 672,
                "train.batch_size": 1,
                "energy.topk_ratio": 0.2,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
                "graph.use_mask_topology": "false",
                "graph.beta_m": 0.0,
            },
        ),
        Trial(
            "proto560_bicubic_aupro_mask_topk20",
            {
                **proto,
                "data.image_size": 560,
                "train.batch_size": 1,
                "energy.topk_ratio": 0.2,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
                "graph.use_mask_topology": "true",
                "graph.beta_m": 1.0,
            },
        ),
        Trial(
            "proto560_bilinear_pro_mask",
            {
                **proto,
                "data.image_size": 560,
                "train.batch_size": 1,
                "energy.upsample_mode": "bilinear",
                "train.early_stopping.monitor": "selection_score_pro",
                "graph.use_mask_topology": "true",
                "graph.beta_m": 1.0,
            },
        ),
        Trial(
            "score10_proto560_bicubic_aupro_nomask",
            {
                "energy.alpha": 0.1,
                "energy.beta": 1.0,
                "energy.gamma": 0.0,
                "energy.eta": 0.0,
                "data.image_size": 560,
                "train.batch_size": 1,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
                "graph.use_mask_topology": "false",
                "graph.beta_m": 0.0,
            },
        ),
        Trial(
            "score25_proto560_bicubic_aupro_nomask",
            {
                "energy.alpha": 0.25,
                "energy.beta": 1.0,
                "energy.gamma": 0.0,
                "energy.eta": 0.0,
                "data.image_size": 560,
                "train.batch_size": 1,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
                "graph.use_mask_topology": "false",
                "graph.beta_m": 0.0,
            },
        ),
        Trial(
            "score05_proto672_bicubic_aupro_nomask",
            {
                "energy.alpha": 0.05,
                "energy.beta": 1.0,
                "energy.gamma": 0.0,
                "energy.eta": 0.0,
                "data.image_size": 672,
                "train.batch_size": 1,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
                "graph.use_mask_topology": "false",
                "graph.beta_m": 0.0,
            },
        ),
        Trial(
            "score10_proto672_bicubic_aupro_nomask",
            {
                "energy.alpha": 0.1,
                "energy.beta": 1.0,
                "energy.gamma": 0.0,
                "energy.eta": 0.0,
                "data.image_size": 672,
                "train.batch_size": 1,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
                "graph.use_mask_topology": "false",
                "graph.beta_m": 0.0,
            },
        ),
        Trial(
            "proto448_bicubic_aupro_nomask_topk005",
            {
                **proto,
                "data.image_size": 448,
                "train.batch_size": 1,
                "energy.topk_ratio": 0.005,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
                "graph.use_mask_topology": "false",
                "graph.beta_m": 0.0,
            },
        ),
        Trial(
            "proto448_bicubic_pro_nomask_topk20",
            {
                **proto,
                "data.image_size": 448,
                "train.batch_size": 1,
                "energy.topk_ratio": 0.2,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "selection_score_pro",
                "graph.use_mask_topology": "false",
                "graph.beta_m": 0.0,
            },
        ),
        Trial(
            "score05_topo001_proto560_bicubic_aupro_mask",
            {
                "energy.alpha": 0.05,
                "energy.beta": 1.0,
                "energy.gamma": 0.001,
                "energy.eta": 0.0,
                "data.image_size": 560,
                "train.batch_size": 1,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
                "graph.use_mask_topology": "true",
                "graph.beta_m": 1.0,
            },
        ),
        Trial(
            "proto560_wave001_bicubic_aupro_mask",
            {
                "energy.alpha": 0.0,
                "energy.beta": 1.0,
                "energy.gamma": 0.0,
                "energy.eta": 0.001,
                "data.image_size": 560,
                "train.batch_size": 1,
                "energy.upsample_mode": "bicubic",
                "train.early_stopping.monitor": "au_pro",
                "graph.use_mask_topology": "true",
                "graph.beta_m": 1.0,
            },
        ),
    ]


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
    parser.add_argument("--output-root", default="outputs/aupro_fix_v5")
    parser.add_argument("--log-root", default="outputs/logs/aupro_fix_v5_trials")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--max-parallel", type=int, default=2)
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
        for trial in v5_trials():
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
        raise RuntimeError(f"No v5 tasks matched targets={sorted(targets)}")
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
    print(f"[aupro-v5] START {task.name} log={task.log_file}", flush=True)
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
        f"[aupro-v5] total={len(tasks)} skipped={len(tasks) - len(pending)} "
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
                print(f"[aupro-v5] SKIP {task.name}", flush=True)
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
                    print(f"[aupro-v5] RETRY {task.name} exit={code}", flush=True)
                    pending.insert(0, task)
                    continue
                print(f"[aupro-v5] FAIL {task.name} exit={code} log={task.log_file}", flush=True)
                raise SystemExit(code)
            print(f"[aupro-v5] DONE {task.name}", flush=True)
        running = next_running
        if pending or running:
            time.sleep(5)


def main() -> None:
    run_tasks(parse_args())


if __name__ == "__main__":
    main()
