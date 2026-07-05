#!/usr/bin/env python3
"""Write periodic experiment progress snapshots to a log file."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def process_alive(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def count_active_children(ppid: int) -> int:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,ppid,cmd"], text=True, errors="ignore")
    except Exception:
        return 0
    count = 0
    for line in out.splitlines():
        if "scripts/train.py" not in line or " eval.result_dir=" not in line:
            continue
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[1] == str(ppid):
            count += 1
    return count


def count_metrics(root: Path) -> int:
    return len(list(root.glob("**/*_train_metrics.json"))) if root.exists() else 0


def btad_best_summary(root: Path) -> str:
    by_class: dict[str, tuple[float, str, int | None, float, float, float]] = {}
    for path in root.glob("**/*_train_metrics.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        parts = path.relative_to(root).parts
        if len(parts) < 4 or parts[0] != "btad":
            continue
        variant, cls = parts[1], parts[2]
        ev = data.get("best_eval") or {}
        score = ev.get("selection_score")
        if score is None:
            continue
        row = (
            float(score),
            variant,
            data.get("best_epoch"),
            float(ev.get("image_auroc", 0.0)),
            float(ev.get("pixel_auroc", 0.0)),
            float(ev.get("au_pro", 0.0)),
        )
        if cls not in by_class or row[0] > by_class[cls][0]:
            by_class[cls] = row
    if not by_class:
        return "btad_best unavailable"
    lines = ["btad_best"]
    for cls in sorted(by_class):
        score, variant, epoch, image, pixel, pro = by_class[cls]
        lines.append(
            f"{cls} {variant} epoch={epoch} selection={score:.4f} "
            f"I={image:.4f} P={pixel:.4f} PRO={pro:.4f}"
        )
    n = len(by_class)
    lines.append(
        "btad_mean "
        f"I={sum(v[3] for v in by_class.values()) / n:.4f} "
        f"P={sum(v[4] for v in by_class.values()) / n:.4f} "
        f"PRO={sum(v[5] for v in by_class.values()) / n:.4f}"
    )
    return "\n".join(lines)


def aupro_summary(result_root: Path, log_root: Path) -> str:
    rows: list[tuple[str, int | None, float | None, float | None, float | None, float | None]] = []
    for path in result_root.glob("**/*_train_metrics.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        ev = data.get("best_eval") or {}
        rows.append(
            (
                str(path.relative_to(result_root)),
                data.get("best_epoch"),
                ev.get("selection_score_pro"),
                ev.get("image_auroc"),
                ev.get("pixel_auroc"),
                ev.get("au_pro"),
            )
        )

    lines: list[str] = []
    if rows:
        lines.append("aupro_top_by_pro")
        for rel, epoch, score, image, pixel, pro in sorted(
            rows, key=lambda item: item[5] if item[5] is not None else -1.0, reverse=True
        )[:5]:
            lines.append(
                f"{rel} epoch={epoch} selection_pro={float(score or -1):.4f} "
                f"I={float(image or -1):.4f} P={float(pixel or -1):.4f} PRO={float(pro or -1):.4f}"
            )

    for path in sorted(log_root.glob("*/*.log")):
        result = result_root / path.parent.name / path.stem / f"{path.stem}_train_metrics.json"
        if result.exists():
            continue
        best = None
        last = None
        count = 0
        for raw in path.read_text(errors="ignore").splitlines():
            line = raw.strip()
            if not line.startswith("{'epoch':"):
                continue
            try:
                data = ast.literal_eval(line)
            except Exception:
                continue
            ev = data.get("eval") or {}
            count += 1
            score = ev.get("selection_score_pro") or ev.get("selection_score")
            row = (data.get("epoch"), score, ev.get("image_auroc"), ev.get("pixel_auroc"), ev.get("au_pro"))
            last = row
            if score is not None and (best is None or score > best[1]):
                best = row
        if count:
            lines.append(f"running_aupro {path} epochs={count} best={best} last={last}")
    return "\n".join(lines) if lines else "aupro_summary unavailable"


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, errors="ignore").strip()
    except Exception as exc:
        return f"command_failed {' '.join(cmd)}: {exc}"


def tail(path: Path, lines: int) -> str:
    if not path.exists():
        return f"missing {path}"
    text = path.read_text(errors="ignore").splitlines()
    return "\n".join(text[-lines:])


def write_snapshot(args: argparse.Namespace) -> None:
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    main_root = Path(args.main_result_root)
    aupro_root = Path(args.aupro_result_root)
    aupro_log_root = Path(args.aupro_trial_log_root)

    chunks = [
        f"==== {datetime.now(timezone.utc).astimezone().isoformat()} ====",
        f"main_completed {count_metrics(main_root)} of {args.main_total}",
        f"aupro_completed {count_metrics(aupro_root)} of {args.aupro_total}",
        f"active_main {count_active_children(args.main_pid)}",
        f"active_aupro {count_active_children(args.aupro_pid)}",
        btad_best_summary(main_root),
        aupro_summary(aupro_root, aupro_log_root),
        command_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        ),
        "main_scheduler_tail\n" + tail(Path(args.main_log), 8),
        "aupro_scheduler_tail\n" + tail(Path(args.aupro_log), 8),
        "",
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(chunks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="outputs/logs/progress_3h_monitor.log")
    parser.add_argument("--interval-seconds", type=int, default=10800)
    parser.add_argument("--main-pid", type=int, default=587877)
    parser.add_argument("--aupro-pid", type=int, default=663181)
    parser.add_argument("--main-total", type=int, default=225)
    parser.add_argument("--aupro-total", type=int, default=15)
    parser.add_argument("--main-result-root", default="outputs/hparam_tuning_fixed/results")
    parser.add_argument("--aupro-result-root", default="outputs/aupro_fix/results/btad")
    parser.add_argument("--aupro-trial-log-root", default="outputs/logs/aupro_fix_trials/btad")
    parser.add_argument("--main-log", default="outputs/logs/hparam_tuning_fixed_parallel_x12.log")
    parser.add_argument("--aupro-log", default="outputs/logs/aupro_fix_parallel.log")
    parser.add_argument("--main-summary-command", default="/root/miniconda3/envs/nostdiffad/bin/python scripts/tune_downloaded_datasets.py --output-root outputs/hparam_tuning_fixed")
    args = parser.parse_args()

    main_summary_done = False
    aupro_done_logged = False
    while True:
        write_snapshot(args)
        if not main_summary_done and not process_alive(args.main_pid):
            with Path(args.log).open("a", encoding="utf-8") as handle:
                handle.write(f"==== main scheduler finished {datetime.now(timezone.utc).astimezone().isoformat()} ====\n")
            subprocess.run(args.main_summary_command, shell=True, stdout=open(args.log, "a"), stderr=subprocess.STDOUT)
            main_summary_done = True
        if not aupro_done_logged and not process_alive(args.aupro_pid):
            with Path(args.log).open("a", encoding="utf-8") as handle:
                handle.write(f"==== aupro scheduler finished {datetime.now(timezone.utc).astimezone().isoformat()} ====\n")
            aupro_done_logged = True
        if main_summary_done and aupro_done_logged:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
