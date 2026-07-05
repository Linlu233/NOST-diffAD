#!/usr/bin/env python
from __future__ import annotations

import argparse
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


def proto_overrides(
    *,
    size: int,
    mode: str,
    monitor: str,
    topk: float = 0.005,
    mask: bool = False,
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
    }


def score_overrides(
    *,
    alpha: float,
    size: int,
    mode: str,
    monitor: str,
    topk: float = 0.005,
    mask: bool = False,
    gamma: float = 0.0,
    eta: float = 0.0,
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
    }


def small_defect_trials() -> list[Trial]:
    return [
        Trial("proto784_nearest_pixelap_nomask_tk005", proto_overrides(size=784, mode="nearest", monitor="pixel_ap")),
        Trial("proto784_nearest_aupro_nomask_tk005", proto_overrides(size=784, mode="nearest", monitor="au_pro")),
        Trial("proto784_area_pixelap_nomask_tk005", proto_overrides(size=784, mode="area", monitor="pixel_ap")),
        Trial("proto784_bicubic_pixelap_nomask_tk005", proto_overrides(size=784, mode="bicubic", monitor="pixel_ap")),
        Trial("proto672_nearest_pixelap_nomask_tk002", proto_overrides(size=672, mode="nearest", monitor="pixel_ap", topk=0.002)),
        Trial("proto560_nearest_pixelap_mask_tk005", proto_overrides(size=560, mode="nearest", monitor="pixel_ap", mask=True)),
        Trial("score01_proto672_nearest_pixelap_nomask", score_overrides(alpha=0.01, size=672, mode="nearest", monitor="pixel_ap")),
        Trial("score02_proto784_nearest_pixelap_nomask", score_overrides(alpha=0.02, size=784, mode="nearest", monitor="pixel_ap")),
        Trial("score05_proto784_nearest_pixelap_nomask", score_overrides(alpha=0.05, size=784, mode="nearest", monitor="pixel_ap")),
        Trial(
            "score02_topo0001_proto784_nearest_pixelap_mask",
            score_overrides(alpha=0.02, gamma=0.0001, size=784, mode="nearest", monitor="pixel_ap", mask=True),
        ),
        Trial(
            "score02_topo0005_proto672_nearest_aupro_mask",
            score_overrides(alpha=0.02, gamma=0.0005, size=672, mode="nearest", monitor="au_pro", mask=True),
        ),
        Trial(
            "proto784_wave0005_nearest_pixelap_mask",
            score_overrides(alpha=0.0, eta=0.0005, size=784, mode="nearest", monitor="pixel_ap", mask=True),
        ),
    ]


def loco_trials() -> list[Trial]:
    return [
        Trial("proto784_nearest_aupro_mask_tk005", proto_overrides(size=784, mode="nearest", monitor="au_pro", mask=True)),
        Trial("proto784_area_pro_mask_tk005", proto_overrides(size=784, mode="area", monitor="selection_score_pro", mask=True)),
        Trial("proto672_nearest_pixelap_mask_tk005", proto_overrides(size=672, mode="nearest", monitor="pixel_ap", mask=True)),
        Trial("proto672_nearest_aupro_nomask_tk002", proto_overrides(size=672, mode="nearest", monitor="au_pro", topk=0.002)),
        Trial("score01_proto672_nearest_aupro_nomask", score_overrides(alpha=0.01, size=672, mode="nearest", monitor="au_pro")),
        Trial("score02_proto672_nearest_aupro_nomask", score_overrides(alpha=0.02, size=672, mode="nearest", monitor="au_pro")),
        Trial(
            "score02_topo0001_proto672_nearest_aupro_mask",
            score_overrides(alpha=0.02, gamma=0.0001, size=672, mode="nearest", monitor="au_pro", mask=True),
        ),
        Trial(
            "proto784_wave0005_nearest_aupro_mask",
            score_overrides(alpha=0.0, eta=0.0005, size=784, mode="nearest", monitor="au_pro", mask=True),
        ),
    ]


def v5b_trials(dataset: str, category: str) -> list[Trial]:
    if dataset == "mvtec_ad_2" and category in {"can", "wallplugs"}:
        return small_defect_trials()
    return loco_trials()


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
    parser.add_argument("--output-root", default="outputs/aupro_fix_v5b")
    parser.add_argument("--log-root", default="outputs/logs/aupro_fix_v5b_trials")
    parser.add_argument("--epochs", type=int, default=190)
    parser.add_argument("--patience", type=int, default=28)
    parser.add_argument("--min-epochs", type=int, default=38)
    parser.add_argument("--max-parallel", type=int, default=1)
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
        for trial in v5b_trials(experiment.dataset, experiment.category):
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
        raise RuntimeError(f"No v5b tasks matched targets={sorted(targets)}")
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
    print(f"[aupro-v5b] START {task.name} log={task.log_file}", flush=True)
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
        f"[aupro-v5b] total={len(tasks)} skipped={len(tasks) - len(pending)} "
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
                print(f"[aupro-v5b] SKIP {task.name}", flush=True)
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
                    print(f"[aupro-v5b] RETRY {task.name} exit={code}", flush=True)
                    pending.insert(0, task)
                    continue
                print(f"[aupro-v5b] FAIL {task.name} exit={code} log={task.log_file}", flush=True)
                raise SystemExit(code)
            print(f"[aupro-v5b] DONE {task.name}", flush=True)
        running = next_running
        if pending or running:
            time.sleep(5)


def main() -> None:
    run_tasks(parse_args())


if __name__ == "__main__":
    main()
