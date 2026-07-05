#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from run_downloaded_official_experiments import Experiment, downloaded_experiments, shell_join


@dataclass(frozen=True)
class ScanTask:
    experiment: Experiment
    category_file: Path
    log_file: Path

    @property
    def name(self) -> str:
        return f"{self.experiment.dataset}/{self.experiment.category}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", default="outputs/hparam_tuning/best_configs")
    parser.add_argument("--checkpoint-root", default="outputs/checkpoints_official_tuned")
    parser.add_argument("--output-root", default="outputs/diagnostics/energy_weight_scan_tune")
    parser.add_argument("--log-root", default="outputs/logs/energy_weight_scan_parallel")
    parser.add_argument("--write-config-root", default="outputs/hparam_tuning_fixed/best_energy_configs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-role", default="tune", choices=["tune", "final", "all"])
    parser.add_argument("--test-split-fraction", type=float, default=0.5)
    parser.add_argument("--selection-metric", default="selection_score")
    parser.add_argument("--selected-summary", default=None)
    parser.add_argument("--au-pro-steps", type=int, default=20)
    parser.add_argument("--candidate-set", choices=["compact", "full"], default="compact")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def valid_category_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, list) and bool(payload)


def build_tasks(args: argparse.Namespace) -> list[ScanTask]:
    output_root = Path(args.output_root)
    log_root = Path(args.log_root)
    tasks: list[ScanTask] = []
    for experiment in downloaded_experiments():
        tasks.append(
            ScanTask(
                experiment=experiment,
                category_file=output_root / "categories" / experiment.dataset / f"{experiment.category}.json",
                log_file=log_root / experiment.dataset / f"{experiment.category}.log",
            )
        )
    return tasks


def command(args: argparse.Namespace, task: ScanTask) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/scan_energy_weights.py",
        "--config-root",
        str(args.config_root),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--output-root",
        str(args.output_root),
        "--device",
        str(args.device),
        "--split-role",
        str(args.split_role),
        "--test-split-fraction",
        str(args.test_split_fraction),
        "--selection-metric",
        str(args.selection_metric),
        "--au-pro-steps",
        str(args.au_pro_steps),
        "--candidate-set",
        str(args.candidate_set),
    ]
    if args.selected_summary:
        cmd.extend(["--selected-summary", str(args.selected_summary)])
    cmd.extend(
        [
            "--dataset",
            task.experiment.dataset,
            "--category",
            task.experiment.category,
            "--no-summary",
        ]
    )
    return cmd


def launch(args: argparse.Namespace, task: ScanTask) -> tuple[subprocess.Popen, object]:
    task.log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = task.log_file.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    cmd = command(args, task)
    print(f"[parallel-energy-scan] START {task.name} log={task.log_file}", flush=True)
    print("+ " + shell_join(cmd), file=handle, flush=True)
    return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env), handle


def run_tasks(args: argparse.Namespace, tasks: list[ScanTask]) -> None:
    pending = [task for task in tasks if not valid_category_file(task.category_file)]
    print(
        f"[parallel-energy-scan] total={len(tasks)} skipped_completed={len(tasks) - len(pending)} "
        f"pending={len(pending)} max_parallel={args.max_parallel}",
        flush=True,
    )
    if args.dry_run:
        for task in pending:
            print("+ " + shell_join(command(args, task)), flush=True)
        return

    running: list[tuple[ScanTask, subprocess.Popen, object]] = []
    attempts: dict[str, int] = {}
    while pending or running:
        while pending and len(running) < max(1, int(args.max_parallel)):
            task = pending.pop(0)
            process, handle = launch(args, task)
            running.append((task, process, handle))

        next_running: list[tuple[ScanTask, subprocess.Popen, object]] = []
        for task, process, handle in running:
            code = process.poll()
            if code is None:
                next_running.append((task, process, handle))
                continue
            handle.close()
            if code == 0 and valid_category_file(task.category_file):
                print(f"[parallel-energy-scan] DONE {task.name}", flush=True)
                continue
            attempt = attempts.get(task.name, 0)
            if attempt < max(0, int(args.retries)):
                attempts[task.name] = attempt + 1
                print(f"[parallel-energy-scan] RETRY {task.name} exit={code}", flush=True)
                pending.insert(0, task)
                continue
            raise RuntimeError(f"Energy scan failed for {task.name}; see {task.log_file}")
        running = next_running
        if pending or running:
            time.sleep(5)


def metric_value(row: dict[str, Any], metric: str) -> float:
    value = float((row.get("metrics") or {}).get(metric, float("nan")))
    return value if math.isfinite(value) else float("nan")


def write_best_config(base_path: Path, weights: dict[str, Any], output_path: Path) -> None:
    config = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    config["energy"]["alpha"] = float(weights["alpha"])
    config["energy"]["beta"] = float(weights["beta"])
    config["energy"]["gamma"] = float(weights["gamma"])
    config["energy"]["eta"] = float(weights["eta"])
    config["energy"]["topk_ratio"] = float(weights["topk_ratio"])
    config["train"]["resume"] = False
    config["train"].setdefault("early_stopping", {})
    config["train"]["early_stopping"]["monitor"] = "selection_score"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def summarize(args: argparse.Namespace, tasks: list[ScanTask]) -> None:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        if valid_category_file(task.category_file):
            rows.extend(json.loads(task.category_file.read_text(encoding="utf-8")))
    if not rows:
        raise RuntimeError("No category scan results found.")

    summary: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        names = sorted({row["candidate"] for row in dataset_rows})
        scores = []
        for name in names:
            candidate_rows = [row for row in dataset_rows if row["candidate"] == name]
            mean_metrics = {}
            for metric in [
                "selection_score",
                "selection_score_pro",
                "image_auroc",
                "image_ap",
                "pixel_auroc",
                "pixel_ap",
                "au_pro",
                "f1_max",
            ]:
                vals = [metric_value(row, metric) for row in candidate_rows]
                vals = [value for value in vals if math.isfinite(value)]
                mean_metrics[metric] = float(sum(vals) / len(vals)) if vals else float("nan")
            scores.append(
                {
                    "candidate": name,
                    "completed_categories": len(candidate_rows),
                    "mean_metrics": mean_metrics,
                    "weights": candidate_rows[0]["weights"],
                }
            )
        valid = [item for item in scores if math.isfinite(float(item["mean_metrics"].get(args.selection_metric, float("nan"))))]
        best = max(valid, key=lambda item: float(item["mean_metrics"][args.selection_metric])) if valid else None
        if best and args.write_config_root:
            config_path = Path(args.write_config_root) / f"{dataset}.yaml"
            write_best_config(Path(args.config_root) / f"{dataset}.yaml", best["weights"], config_path)
            best["config"] = str(config_path)
        summary[dataset] = {
            "split_role": args.split_role,
            "selection_metric": args.selection_metric,
            "scores": scores,
            "best": best,
        }

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    tasks = build_tasks(args)
    run_tasks(args, tasks)
    if not args.dry_run:
        summarize(args, tasks)


if __name__ == "__main__":
    main()
