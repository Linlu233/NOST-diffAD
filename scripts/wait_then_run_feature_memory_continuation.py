#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_downloaded_official_experiments import shell_join


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-match", default="run_feature_memory_auto_loop.py --trial-sets v12 v13")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--official-output-root", default="outputs/feature_memory_official")
    parser.add_argument("--gate-report-dir", default="outputs/feature_memory_gate_v13")
    parser.add_argument("--log", default="outputs/logs/feature_memory_continuation.log")
    parser.add_argument("--next-trial-sets", nargs="*", default=["v14", "v15"])
    parser.add_argument(
        "--base-tune-roots",
        nargs="*",
        default=["outputs/feature_memory_v8", "outputs/feature_memory_v9", "outputs/feature_memory_v12", "outputs/feature_memory_v13"],
    )
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--official-max-parallel", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=int, default=60)
    return parser.parse_args()


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(path: Path, message: str) -> None:
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def matching_processes(pattern: str) -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid,cmd"], check=False, capture_output=True, text=True)
    current = str(os.getpid())
    out: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if "wait_then_run_feature_memory_continuation.py" in stripped:
            continue
        if pattern in stripped and not stripped.startswith(current + " "):
            out.append(stripped)
    return out


def count_results(root: str) -> int:
    result = subprocess.run(
        ["bash", "-lc", f"find {root}/results -name '*_train_metrics.json' 2>/dev/null | wc -l"],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def gate_passed(report_dir: str) -> bool:
    path = Path(report_dir) / "strong_baseline_gate.csv"
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return bool(rows) and all(row.get("status") == "PASS" for row in rows)


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    while True:
        matches = matching_processes(args.wait_match)
        if not matches:
            break
        log(log_path, f"waiting for active loop count={len(matches)} pattern={args.wait_match!r}")
        time.sleep(max(30, int(args.poll_seconds)))

    if count_results(args.official_output_root) > 0 or gate_passed(args.gate_report_dir):
        log(log_path, "official/gate already active or passed; continuation will not start")
        return

    command = [
        sys.executable,
        "scripts/run_feature_memory_auto_loop.py",
        "--trial-sets",
        *args.next_trial_sets,
        "--base-tune-roots",
        *args.base_tune_roots,
        "--max-parallel",
        str(args.max_parallel),
        "--official-max-parallel",
        str(args.official_max_parallel),
        "--max-retries",
        str(args.max_retries),
        "--retry-backoff-seconds",
        str(args.retry_backoff_seconds),
        "--sleep-between-rounds",
        "60",
        "--log",
        "outputs/logs/feature_memory_auto_loop_v14_v15.log",
    ]
    log(log_path, "+ " + shell_join(command))
    with log_path.open("a", encoding="utf-8") as handle:
        raise SystemExit(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT).wait())


if __name__ == "__main__":
    main()
