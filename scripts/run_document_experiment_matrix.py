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


DOCUMENT_DATASETS = {
    "mvtec_ad",
    "mvtec_ad_2",
    "visa",
    "btad",
    "mvtec_loco",
    "mvtec_3d_rgb",
    "mpdd",
    "kolektorsdd2",
    "realiad_1024",
    "dagm",
}

ABLATIONS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_score_diffusion": {"diffusion_disabled": "true", "energy.alpha": 0.0},
    "no_patch_graph": {
        "graph.use_patch_graph": "false",
        "graph.beta_s": 0.0,
        "graph.beta_p": 0.0,
        "graph.beta_m": 0.0,
        "energy.gamma": 0.0,
    },
    "no_sam2_mask_topology": {"graph.use_mask_topology": "false", "graph.beta_m": 0.0},
    "no_wavelet_residual": {"energy.eta": 0.0, "model.wave_dim": 0},
    "no_nmf": {"loss.lambda_nmf": 0.0},
    "no_conformal_threshold": {"conformal_disabled": "true"},
    "dinov2_features": {"model.feature_extractor": "dinov2_torchhub", "model.torchhub_model": "dinov2_vits14"},
    "clip_features": {
        "model.feature_extractor": "clip",
        "model.hf_model": "openai/clip-vit-base-patch16",
        "model.patch_size": 16,
        "model.feature_dim": 768,
    },
    "sam2_features": {"model.feature_extractor": "sam2"},
}

FEW_SHOTS = ["1", "2", "4", "8", "16", "full"]
ROBUSTNESS = ["none", "brightness", "gaussian_noise", "jpeg_compression"]


@dataclass(frozen=True)
class MatrixTask:
    group: str
    variant: str
    experiment: Experiment
    result_dir: Path
    result_file: Path
    log_file: Path
    overrides: dict[str, Any]

    @property
    def name(self) -> str:
        return f"{self.group}/{self.variant}/{self.experiment.dataset}/{self.experiment.category}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="outputs/document_matrix")
    parser.add_argument("--log-root", default="outputs/logs/document_matrix")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--test-split-fraction", type=float, default=0.5)
    parser.add_argument("--test-split-role", choices=["final", "tune", "all"], default="final")
    parser.add_argument("--representative-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-main", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-ablations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-few-shot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-robustness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-sam2", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-clip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-finished", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-fail", action=argparse.BooleanOptionalAction, default=True)
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


def representative_experiments(experiments: list[Experiment]) -> list[Experiment]:
    by_dataset: dict[str, list[Experiment]] = {}
    for experiment in experiments:
        by_dataset.setdefault(experiment.dataset, []).append(experiment)
    reps: list[Experiment] = []
    for dataset in sorted(by_dataset):
        reps.append(sorted(by_dataset[dataset], key=lambda item: item.category)[0])
    return reps


def supported_experiments(args: argparse.Namespace) -> list[Experiment]:
    experiments = [item for item in downloaded_experiments() if item.dataset in DOCUMENT_DATASETS]
    requested = set(args.datasets or [])
    if requested:
        experiments = [item for item in experiments if item.dataset in requested]
    experiments = sorted(experiments, key=lambda item: (item.dataset, item.category))
    if args.representative_only:
        experiments = representative_experiments(experiments)
    return experiments


def base_overrides(args: argparse.Namespace, experiment: Experiment, result_dir: Path) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "device": args.device,
        "data.few_shot": "full",
        "data.robustness": "none",
        "data.test_split_fraction": args.test_split_fraction,
        "data.test_split_role": args.test_split_role,
        "train.resume": "true",
        "data.root": str(experiment.root),
        "data.category": quoted_override("data.category", experiment.category).split("=", 1)[1],
        "data.part_mask_root": str(experiment.part_mask_root),
        "train.save_dir": str(result_dir.parent / "_checkpoints" / result_dir.name),
        "eval.result_dir": str(result_dir),
    }
    if args.epochs is not None:
        overrides["train.epochs"] = int(args.epochs)
    return overrides


def make_task(
    args: argparse.Namespace,
    experiment: Experiment,
    group: str,
    variant: str,
    extra: dict[str, Any],
) -> MatrixTask:
    result_dir = Path(args.output_root) / "results" / group / variant / experiment.dataset / experiment.category
    result_file = result_dir / f"{experiment.category}_train_metrics.json"
    log_file = Path(args.log_root) / group / variant / experiment.dataset / f"{experiment.category}.log"
    overrides = base_overrides(args, experiment, result_dir)
    overrides.update(extra)
    return MatrixTask(group, variant, experiment, result_dir, result_file, log_file, overrides)


def build_tasks(args: argparse.Namespace) -> list[MatrixTask]:
    all_experiments = supported_experiments(args)
    reps = representative_experiments(all_experiments)
    tasks: list[MatrixTask] = []

    if args.include_main:
        for experiment in all_experiments:
            tasks.append(make_task(args, experiment, "main", "full", {}))

    if args.include_ablations:
        for name, overrides in ABLATIONS.items():
            if name == "sam2_features" and not args.include_sam2:
                continue
            if name == "clip_features" and not args.include_clip:
                continue
            for experiment in reps:
                tasks.append(make_task(args, experiment, "ablation", name, overrides))

    if args.include_few_shot:
        for few_shot in FEW_SHOTS:
            for experiment in reps:
                tasks.append(make_task(args, experiment, "few_shot", f"shot_{few_shot}", {"data.few_shot": few_shot}))

    if args.include_robustness:
        for robustness in ROBUSTNESS:
            for experiment in reps:
                tasks.append(make_task(args, experiment, "robustness", robustness, {"data.robustness": robustness}))

    return tasks


def command(args: argparse.Namespace, task: MatrixTask) -> list[str]:
    overrides = []
    for key, value in task.overrides.items():
        if key == "data.category":
            overrides.append(f"{key}={value}")
        else:
            overrides.append(f"{key}={value}")
    return [sys.executable, "scripts/train.py", "--config", args.config, "--set", *overrides]


def write_manifest(tasks: list[MatrixTask], args: argparse.Namespace) -> None:
    path = Path(args.output_root) / "document_experiment_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["group", "variant", "dataset", "category", "result_file", "log_file", "overrides"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "group": task.group,
                    "variant": task.variant,
                    "dataset": task.experiment.dataset,
                    "category": task.experiment.category,
                    "result_file": str(task.result_file),
                    "log_file": str(task.log_file),
                    "overrides": json.dumps(task.overrides, ensure_ascii=False, sort_keys=True),
                }
            )


def run_tasks(args: argparse.Namespace, tasks: list[MatrixTask]) -> None:
    if args.skip_finished:
        pending = [task for task in tasks if not result_completed(task.result_file)]
    else:
        pending = list(tasks)
    print(f"[document-matrix] total={len(tasks)} pending={len(pending)} max_parallel={args.max_parallel}", flush=True)
    write_manifest(tasks, args)
    if args.dry_run:
        for task in pending:
            print("+ " + shell_join(command(args, task)), flush=True)
        return

    queue: list[tuple[MatrixTask, int]] = [(task, 1) for task in pending]
    running: list[tuple[MatrixTask, int, subprocess.Popen, Any]] = []
    failures: list[tuple[MatrixTask, int]] = []
    while queue or running:
        while queue and len(running) < max(1, int(args.max_parallel)):
            task, attempt = queue.pop(0)
            task.log_file.parent.mkdir(parents=True, exist_ok=True)
            task.result_dir.mkdir(parents=True, exist_ok=True)
            handle = task.log_file.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            cmd = command(args, task)
            print(f"[document-matrix] START {task.name} attempt={attempt} log={task.log_file}", flush=True)
            print(f"# attempt={attempt}", file=handle, flush=True)
            print("+ " + shell_join(cmd), file=handle, flush=True)
            running.append((task, attempt, subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env), handle))

        next_running: list[tuple[MatrixTask, int, subprocess.Popen, Any]] = []
        for task, attempt, process, handle in running:
            code = process.poll()
            if code is None:
                next_running.append((task, attempt, process, handle))
                continue
            handle.close()
            if code == 0 and result_completed(task.result_file):
                print(f"[document-matrix] DONE {task.name}", flush=True)
                continue
            print(f"[document-matrix] FAIL {task.name} exit={code} attempt={attempt}/{args.max_retries + 1}", flush=True)
            if attempt <= int(args.max_retries):
                time.sleep(max(0, int(args.retry_backoff_seconds)))
                queue.insert(0, (task, attempt + 1))
            else:
                failures.append((task, code if code is not None else 1))
                if not args.continue_on_fail:
                    raise SystemExit(code if code else 1)
        running = next_running
        if queue or running:
            time.sleep(5)

    if failures:
        path = Path(args.output_root) / "failed_tasks.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(f"{task.name}\texit={code}\tlog={task.log_file}" for task, code in failures) + "\n",
            encoding="utf-8",
        )
        print(f"[document-matrix] failed={len(failures)} wrote {path}", flush=True)
        if not args.continue_on_fail:
            raise SystemExit(1)


def main() -> None:
    args = parse_args()
    tasks = build_tasks(args)
    if not tasks:
        raise RuntimeError("No document-matrix tasks were generated. Check downloaded datasets.")
    run_tasks(args, tasks)


if __name__ == "__main__":
    main()
