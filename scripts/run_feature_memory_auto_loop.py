#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_downloaded_official_experiments import downloaded_experiments, shell_join


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trial-sets", nargs="*", default=["v12", "v13"])
    parser.add_argument("--base-tune-roots", nargs="*", default=["outputs/feature_memory_v8", "outputs/feature_memory_v9"])
    parser.add_argument("--output-prefix", default="outputs/feature_memory_")
    parser.add_argument("--log-prefix", default="outputs/logs/feature_memory_")
    parser.add_argument("--gate-prefix", default="outputs/feature_memory_gate_")
    parser.add_argument("--official-output-root", default="outputs/feature_memory_official")
    parser.add_argument("--official-log-root", default="outputs/logs/feature_memory_official")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--official-max-parallel", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=int, default=45)
    parser.add_argument("--min-passing-metrics", type=int, default=3)
    parser.add_argument("--test-split-fraction", type=float, default=0.5)
    parser.add_argument("--sleep-between-rounds", type=int, default=60)
    parser.add_argument("--log", default="outputs/logs/feature_memory_auto_loop.log")
    return parser.parse_args()


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{stamp()}] + {shell_join(command)}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        return process.wait()


def tune_roots(args: argparse.Namespace, completed_trial_sets: list[str]) -> list[str]:
    roots = list(args.base_tune_roots)
    roots.extend(f"{args.output_prefix}{trial_set}" for trial_set in completed_trial_sets)
    return roots


def gate_passed(report_dir: Path) -> bool:
    gate_file = report_dir / "strong_baseline_gate.csv"
    if not gate_file.exists():
        return False
    with gate_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return bool(rows) and all(row.get("status") == "PASS" for row in rows)


def log_message(args: argparse.Namespace, message: str) -> None:
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    path = Path(args.log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_sweep(args: argparse.Namespace, trial_set: str) -> int:
    return run_command(
        [
            sys.executable,
            "scripts/run_feature_memory_sweep.py",
            "--config",
            args.config,
            "--device",
            args.device,
            "--trial-set",
            trial_set,
            "--output-root",
            f"{args.output_prefix}{trial_set}",
            "--log-root",
            f"{args.log_prefix}{trial_set}",
            "--test-split-role",
            "tune",
            "--test-split-fraction",
            str(args.test_split_fraction),
            "--max-parallel",
            str(args.max_parallel),
            "--max-retries",
            str(args.max_retries),
            "--retry-backoff-seconds",
            str(args.retry_backoff_seconds),
            "--continue-on-fail",
        ],
        Path(args.log),
    )


def run_gate(args: argparse.Namespace, trial_set: str, roots: list[str]) -> tuple[int, Path]:
    report_dir = Path(f"{args.gate_prefix}{trial_set}")
    command = [
        sys.executable,
        "scripts/run_feature_memory_gate_and_official.py",
        "--config",
        args.config,
        "--device",
        args.device,
        "--tune-roots",
        *roots,
        "--official-output-root",
        args.official_output_root,
        "--official-log-root",
        args.official_log_root,
        "--report-dir",
        str(report_dir),
        "--min-passing-metrics",
        str(args.min_passing_metrics),
        "--expected-count",
        str(len(downloaded_experiments())),
        "--max-parallel",
        str(args.official_max_parallel),
        "--official-max-retries",
        str(args.max_retries),
        "--official-retry-backoff-seconds",
        str(args.retry_backoff_seconds),
        "--start-official",
        "--strict-all-datasets",
    ]
    return run_command(command, Path(args.log)), report_dir


def main() -> None:
    args = parse_args()
    completed_trial_sets: list[str] = []
    for trial_set in args.trial_sets:
        log_message(args, f"START sweep trial_set={trial_set} max_parallel={args.max_parallel}")
        sweep_code = run_sweep(args, trial_set)
        if sweep_code != 0:
            log_message(args, f"SWEEP nonzero trial_set={trial_set} exit={sweep_code}; continuing to gate with completed rows")
        completed_trial_sets.append(trial_set)
        roots = tune_roots(args, completed_trial_sets)
        log_message(args, f"START gate trial_set={trial_set} roots={','.join(roots)}")
        gate_code, report_dir = run_gate(args, trial_set, roots)
        if gate_code == 0 and gate_passed(report_dir):
            log_message(args, f"GATE PASS report={report_dir / 'GATE_REPORT.md'}; official final started/completed by gate script")
            return
        log_message(args, f"GATE not passed trial_set={trial_set} report={report_dir / 'GATE_REPORT.md'}")
        time.sleep(max(0, int(args.sleep_between_rounds)))
    log_message(args, "All configured trial sets finished without gate pass. Add a new trial set/strategy before official final.")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
