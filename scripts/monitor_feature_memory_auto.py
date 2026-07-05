#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def count(pattern: str) -> int:
    result = subprocess.run(["bash", "-lc", f"find {pattern} 2>/dev/null | wc -l"], capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def shell(command: str) -> str:
    result = subprocess.run(["bash", "-lc", command], capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


def feature_memory_counts() -> str:
    return shell(
        "for d in outputs/feature_memory_v*; do "
        "[ -d \"$d/results\" ] || continue; "
        "printf '%s=%s ' \"$(basename \"$d\")\" \"$(find \"$d/results\" -name '*_train_metrics.json' 2>/dev/null | wc -l)\"; "
        "done"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--log", default="outputs/logs/feature_memory_auto_monitor.log")
    args = parser.parse_args()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fm_counts = feature_memory_counts()
        official = count("outputs/feature_memory_official/results -name '*_train_metrics.json'")
        screens = shell("screen -ls | sed -n '1,12p'")
        gpu = shell("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")
        errors = shell(
            "grep -R \"CUDA out\\|Traceback\\|FAIL\\|RuntimeError\\|No space\\|PytorchStreamWriter\\|exit=-9\" "
            "-n outputs/logs/feature_memory_auto_loop.log outputs/logs/feature_memory_v12 "
            "outputs/logs/feature_memory_v13 outputs/logs/feature_memory_v14 outputs/logs/feature_memory_v15 "
            "outputs/logs/feature_memory_official 2>/dev/null | tail -n 30"
        )
        failed_tasks = shell(
            "for f in outputs/feature_memory_v12/failed_tasks.txt outputs/feature_memory_v13/failed_tasks.txt "
            "outputs/feature_memory_v14/failed_tasks.txt outputs/feature_memory_v15/failed_tasks.txt; do "
            "[ -f \"$f\" ] && echo \"# $f\" && tail -n 10 \"$f\"; done 2>/dev/null | tail -n 30"
        )
        message = (
            f"[{stamp}] {fm_counts or 'feature_memory=0'} official={official}\n"
            f"gpu={gpu}\n"
            f"screens:\n{screens}\n"
            f"recent_errors:\n{errors or 'none'}\n"
            f"failed_tasks:\n{failed_tasks or 'none'}\n"
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        print(message, flush=True)
        time.sleep(max(60, int(args.interval_seconds)))


if __name__ == "__main__":
    main()
