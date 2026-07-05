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
from run_feature_memory_sweep import METRIC_KEYS, Trial, all_trials, metric_value
from run_feature_memory_v8 import trials as legacy_v8_trials


CORE_METRICS = ["image_auroc", "pixel_auroc", "au_pro", "pixel_ap"]


@dataclass(frozen=True)
class Baseline:
    source: str
    metrics: dict[str, float]
    paper_comparable: bool = True


STRONG_BASELINES: dict[str, Baseline] = {
    "mvtec_ad": Baseline(
        "Dinomaly CVPR 2025/arXiv tables",
        {"image_auroc": 0.996, "pixel_auroc": 0.984, "au_pro": 0.948, "pixel_ap": 0.693},
    ),
    "visa": Baseline(
        "Dinomaly CVPR 2025/arXiv tables",
        {"image_auroc": 0.987, "pixel_auroc": 0.987, "au_pro": 0.945, "pixel_ap": 0.532},
    ),
    "realiad_1024": Baseline(
        "Dinomaly CVPR 2025/arXiv tables",
        {"image_auroc": 0.893, "pixel_auroc": 0.988, "au_pro": 0.939, "pixel_ap": 0.428},
    ),
    "mpdd": Baseline(
        "Dinomaly CVPR 2025/arXiv appendix tables",
        {"image_auroc": 0.972, "pixel_auroc": 0.991, "au_pro": 0.966, "pixel_ap": 0.595},
    ),
    "btad": Baseline(
        "Dinomaly CVPR 2025/arXiv appendix tables",
        {"image_auroc": 0.954, "pixel_auroc": 0.978, "au_pro": 0.765, "pixel_ap": 0.701},
    ),
    "mvtec_loco": Baseline(
        "Recent LOCO papers report image AUROC and sPRO; pixel metrics here are internal guardrails",
        {"image_auroc": 0.926, "pixel_auroc": 0.900, "au_pro": 0.697, "pixel_ap": 0.300},
        paper_comparable=False,
    ),
    "mvtec_3d_rgb": Baseline(
        "RGB-only internal guardrail for MVTec 3D-AD converted split",
        {"image_auroc": 0.900, "pixel_auroc": 0.970, "au_pro": 0.920, "pixel_ap": 0.400},
        paper_comparable=False,
    ),
    "mvtec_ad_2": Baseline(
        "MVTec AD 2 local converted split guardrail; official server metrics are not directly comparable",
        {"image_auroc": 0.800, "pixel_auroc": 0.850, "au_pro": 0.600, "pixel_ap": 0.200},
        paper_comparable=False,
    ),
    "kolektorsdd2": Baseline(
        "KolektorSDD2 internal guardrail; no configured 2025-2026 paper-comparable table",
        {"image_auroc": 0.950, "pixel_auroc": 0.970, "au_pro": 0.940, "pixel_ap": 0.300},
        paper_comparable=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tune-roots", nargs="*", default=["outputs/feature_memory_v8", "outputs/feature_memory_v9"])
    parser.add_argument("--official-output-root", default="outputs/feature_memory_official")
    parser.add_argument("--official-log-root", default="outputs/logs/feature_memory_official")
    parser.add_argument("--report-dir", default="outputs/feature_memory_gate")
    parser.add_argument("--min-passing-metrics", type=int, default=3)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--wait-for-tune", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--official-max-retries", type=int, default=2)
    parser.add_argument("--official-retry-backoff-seconds", type=int, default=30)
    parser.add_argument("--start-official", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-all-datasets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--paper-only-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_metric_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not payload.get("completed"):
        return None
    metrics = payload.get("best_eval") or payload.get("latest_eval") or {}
    return metrics if isinstance(metrics, dict) else None


def rows_from_json_root(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results = root / "results"
    if not results.exists():
        return rows
    for path in sorted(results.glob("*/*/*/*_train_metrics.json")):
        try:
            dataset, trial, category, _ = path.relative_to(results).parts
        except ValueError:
            continue
        metrics = read_metric_payload(path)
        if metrics is None:
            continue
        row: dict[str, Any] = {"dataset": dataset, "category": category, "trial": trial, "path": str(path)}
        for key in METRIC_KEYS:
            row[key] = metric_value(metrics, key)
        rows.append(row)
    return rows


def rows_from_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out = dict(row)
            for key in METRIC_KEYS:
                out[key] = parse_float(out.get(key))
            rows.append(out)
    return rows


def load_rows(roots: list[str]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for root_text in roots:
        root = Path(root_text)
        source_rows = [*rows_from_csv(root / "all_results.csv"), *rows_from_json_root(root)]
        for row in source_rows:
            key = (
                str(row.get("dataset", "")),
                str(row.get("category", "")),
                str(row.get("trial", "")),
                str(row.get("path", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def baseline_score(row: dict[str, Any], baseline: Baseline | None) -> tuple[int, float, float]:
    if baseline is None:
        return (0, parse_float(row.get("selection_score_pro")), parse_float(row.get("selection_score")))
    passed = 0
    ratio_sum = 0.0
    ratios = 0
    for key, threshold in baseline.metrics.items():
        value = parse_float(row.get(key))
        if value == value:
            passed += int(value >= threshold)
            ratio_sum += value / max(threshold, 1e-12)
            ratios += 1
    return (passed, ratio_sum / max(ratios, 1), parse_float(row.get("selection_score_pro")))


def dataset_selection_score(rows: list[dict[str, Any]], baseline: Baseline) -> tuple[int, float, float]:
    ratios = []
    pro_scores = []
    means = dataset_means(rows)
    mean_row = means[0] if means else {}
    passing = 0
    for key, threshold in baseline.metrics.items():
        value = parse_float(mean_row.get(key))
        if value == value:
            passing += int(value >= threshold)
            ratios.append(value / max(threshold, 1e-12))
    for row in rows:
        value = parse_float(row.get("selection_score_pro"))
        if value == value:
            pro_scores.append(value)
    return (passing, mean(ratios), mean(pro_scores))


def select_dataset_rows(dataset_rows: list[dict[str, Any]], baseline: Baseline | None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in dataset_rows:
        category = str(row.get("category", ""))
        if category:
            grouped.setdefault(category, []).append(row)
    if not grouped:
        return []

    greedy = []
    for category in sorted(grouped):
        rows_for_category = grouped[category]
        greedy.append(max(rows_for_category, key=lambda row: baseline_score(row, baseline)))
    if baseline is None:
        return greedy

    best_rows = greedy
    best_score = dataset_selection_score(best_rows, baseline)
    weight_values = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    for image_weight in weight_values:
        for pixel_weight in weight_values:
            for pro_weight in weight_values:
                for ap_weight in weight_values:
                    weights = {
                        "image_auroc": image_weight,
                        "pixel_auroc": pixel_weight,
                        "au_pro": pro_weight,
                        "pixel_ap": ap_weight,
                    }
                    if not any(weights.values()):
                        continue
                    selected = []
                    for category in sorted(grouped):
                        rows_for_category = grouped[category]

                        def weighted_score(row: dict[str, Any]) -> tuple[float, float]:
                            score = 0.0
                            for key, weight in weights.items():
                                value = parse_float(row.get(key))
                                if value == value:
                                    score += weight * value / max(baseline.metrics[key], 1e-12)
                            return (score, parse_float(row.get("selection_score_pro")))

                        selected.append(max(rows_for_category, key=weighted_score))
                    score = dataset_selection_score(selected, baseline)
                    if score > best_score:
                        best_rows = selected
                        best_score = score
    return best_rows


def best_category_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        dataset = str(row.get("dataset", ""))
        category = str(row.get("category", ""))
        if dataset and category:
            grouped.setdefault(dataset, []).append(row)
    best: list[dict[str, Any]] = []
    for dataset in sorted(grouped):
        best.extend(select_dataset_rows(grouped[dataset], STRONG_BASELINES.get(dataset)))
    return best


def mean(values: list[float]) -> float:
    valid = [value for value in values if value == value]
    return sum(valid) / len(valid) if valid else float("nan")


def dataset_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)
    out: list[dict[str, Any]] = []
    for dataset, dataset_rows in sorted(grouped.items()):
        row: dict[str, Any] = {"dataset": dataset, "categories": len(dataset_rows)}
        for key in METRIC_KEYS:
            row[key] = mean([parse_float(item.get(key)) for item in dataset_rows])
        out.append(row)
    return out


def gate_rows(means: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in means:
        dataset = str(row["dataset"])
        baseline = STRONG_BASELINES.get(dataset)
        if baseline is None:
            out.append(
                {
                    "dataset": dataset,
                    "status": "NO_BASELINE",
                    "passing_metrics": 0,
                    "required_metrics": args.min_passing_metrics,
                    "source": "",
                    "paper_comparable": False,
                }
            )
            continue
        if args.paper_only_gate and not baseline.paper_comparable:
            status = "SKIP_NON_PAPER"
            passing = 0
        else:
            passing = sum(
                int(parse_float(row.get(key)) >= threshold)
                for key, threshold in baseline.metrics.items()
            )
            status = "PASS" if passing >= args.min_passing_metrics else "FAIL"
        gate_row: dict[str, Any] = {
            "dataset": dataset,
            "status": status,
            "passing_metrics": passing,
            "required_metrics": args.min_passing_metrics,
            "source": baseline.source,
            "paper_comparable": baseline.paper_comparable,
        }
        for key in CORE_METRICS:
            gate_row[f"{key}_value"] = parse_float(row.get(key))
            gate_row[f"{key}_baseline"] = baseline.metrics.get(key, float("nan"))
            gate_row[f"{key}_pass"] = parse_float(row.get(key)) >= baseline.metrics.get(key, float("inf"))
        out.append(gate_row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(report_dir: Path, best_rows: list[dict[str, Any]], means: list[dict[str, Any]], gates: list[dict[str, Any]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "best_category_rows.csv", best_rows, ["dataset", "category", "trial", *METRIC_KEYS, "path"])
    write_csv(report_dir / "dataset_means.csv", means, ["dataset", "categories", *METRIC_KEYS])
    gate_fields = [
        "dataset",
        "status",
        "passing_metrics",
        "required_metrics",
        "paper_comparable",
        "source",
        *[f"{key}_{suffix}" for key in CORE_METRICS for suffix in ("value", "baseline", "pass")],
    ]
    write_csv(report_dir / "strong_baseline_gate.csv", gates, gate_fields)

    lines = ["# Feature Memory Strong-Baseline Gate", ""]
    lines.append("| dataset | status | pass/required | image AUROC | pixel AUROC | AU-PRO | pixel AP | source |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in gates:
        lines.append(
            "| {dataset} | {status} | {passing_metrics}/{required_metrics} | "
            "{image_auroc_value:.4f}/{image_auroc_baseline:.4f} | "
            "{pixel_auroc_value:.4f}/{pixel_auroc_baseline:.4f} | "
            "{au_pro_value:.4f}/{au_pro_baseline:.4f} | "
            "{pixel_ap_value:.4f}/{pixel_ap_baseline:.4f} | {source} |".format(**row)
        )
    lines.append("")
    lines.append("Only paper-comparable rows should be described as exceeding a published strong baseline.")
    (report_dir / "GATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def all_required_pass(gates: list[dict[str, Any]], args: argparse.Namespace) -> bool:
    blocking = []
    for row in gates:
        if args.paper_only_gate and not bool(row.get("paper_comparable")):
            continue
        if str(row["status"]) != "PASS":
            blocking.append(str(row["dataset"]))
    if args.strict_all_datasets:
        return not blocking
    return any(str(row["status"]) == "PASS" for row in gates)


def trial_lookup() -> dict[str, Trial]:
    lookup = all_trials()
    for trial in legacy_v8_trials():
        lookup[trial.name] = Trial(trial.name, trial.overrides)
    return lookup


def experiment_lookup() -> dict[tuple[str, str], Experiment]:
    return {(experiment.dataset, experiment.category): experiment for experiment in downloaded_experiments()}


def official_command(args: argparse.Namespace, experiment: Experiment, trial: Trial, result_dir: Path) -> list[str]:
    overrides = [
        "device=" + args.device,
        "data.few_shot=full",
        "data.robustness=none",
        "data.test_split_fraction=0.5",
        "data.test_split_role=final",
        "data.root=" + str(experiment.root),
        quoted_override("data.category", experiment.category),
        "data.part_mask_root=" + str(experiment.part_mask_root),
        "eval.result_dir=" + str(result_dir),
    ]
    overrides.extend(f"{key}={value}" for key, value in trial.overrides.items())
    return [sys.executable, "scripts/evaluate_feature_memory.py", "--config", args.config, "--set", *overrides]


def result_completed(path: Path) -> bool:
    return read_metric_payload(path) is not None


def run_official(args: argparse.Namespace, best_rows: list[dict[str, Any]]) -> None:
    experiments = experiment_lookup()
    trials = trial_lookup()
    tasks: list[tuple[str, Path, Path, list[str], int]] = []
    for row in best_rows:
        dataset = str(row["dataset"])
        category = str(row["category"])
        trial_name = str(row["trial"])
        experiment = experiments.get((dataset, category))
        trial = trials.get(trial_name)
        if experiment is None or trial is None:
            print(f"[feature-memory-official] SKIP unknown {dataset}/{category}/{trial_name}", flush=True)
            continue
        result_dir = Path(args.official_output_root) / "results" / dataset / trial_name / category
        result_file = result_dir / f"{category}_train_metrics.json"
        if result_completed(result_file):
            print(f"[feature-memory-official] SKIP finished {dataset}/{category}/{trial_name}", flush=True)
            continue
        log_file = Path(args.official_log_root) / dataset / trial_name / f"{category}.log"
        tasks.append((f"{dataset}/{category}/{trial_name}", result_file, log_file, official_command(args, experiment, trial, result_dir), 1))

    print(f"[feature-memory-official] pending={len(tasks)} max_parallel={args.max_parallel}", flush=True)
    if args.dry_run:
        for _, _, _, command, _ in tasks:
            print("+ " + shell_join(command), flush=True)
        return

    running: list[tuple[str, Path, Path, list[str], subprocess.Popen, object, int]] = []
    while tasks or running:
        while tasks and len(running) < max(1, int(args.max_parallel)):
            name, result_file, log_file, command, attempt = tasks.pop(0)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handle = log_file.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            print(f"[feature-memory-official] START {name} attempt={attempt} log={log_file}", flush=True)
            print(f"# attempt={attempt}", file=handle, flush=True)
            print("+ " + shell_join(command), file=handle, flush=True)
            running.append((name, result_file, log_file, command, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env), handle, attempt))
        next_running: list[tuple[str, Path, Path, list[str], subprocess.Popen, object, int]] = []
        for name, result_file, log_file, command, process, handle, attempt in running:
            code = process.poll()
            if code is None:
                next_running.append((name, result_file, log_file, command, process, handle, attempt))
                continue
            handle.close()
            if code != 0 or not result_completed(result_file):
                print(f"[feature-memory-official] FAIL {name} exit={code} attempt={attempt}/{args.official_max_retries + 1}", flush=True)
                if attempt <= int(args.official_max_retries):
                    time.sleep(max(0, int(args.official_retry_backoff_seconds)))
                    tasks.insert(0, (name, result_file, log_file, command, attempt + 1))
                    continue
                raise SystemExit(code if code != 0 else 1)
            print(f"[feature-memory-official] DONE {name}", flush=True)
        running = next_running
        if tasks or running:
            time.sleep(5)


def expected_completed_count(args: argparse.Namespace) -> int:
    if args.expected_count is not None:
        return int(args.expected_count)
    return len(downloaded_experiments())


def wait_for_tune(args: argparse.Namespace) -> list[dict[str, Any]]:
    expected = expected_completed_count(args)
    while True:
        rows = load_rows(args.tune_roots)
        print(f"[feature-memory-gate] tune completed rows={len(rows)} expected>={expected}", flush=True)
        if len(rows) >= expected:
            return rows
        if not args.wait_for_tune:
            return rows
        time.sleep(max(5, int(args.poll_seconds)))


def main() -> None:
    args = parse_args()
    rows = wait_for_tune(args)
    best_rows = best_category_rows(rows)
    means = dataset_means(best_rows)
    gates = gate_rows(means, args)
    write_report(Path(args.report_dir), best_rows, means, gates)
    passed = all_required_pass(gates, args)
    print(f"[feature-memory-gate] wrote {Path(args.report_dir) / 'GATE_REPORT.md'}", flush=True)
    print(f"[feature-memory-gate] gate_pass={passed}", flush=True)
    if not passed:
        failed = ", ".join(str(row["dataset"]) for row in gates if row["status"] == "FAIL")
        print(f"[feature-memory-gate] official not started; failed datasets: {failed}", flush=True)
        return
    if args.start_official:
        run_official(args, best_rows)


if __name__ == "__main__":
    main()
