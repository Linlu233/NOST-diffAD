#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src:scripts
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PY="${PY:-/root/miniconda3/envs/nostdiffad/bin/python}"
TARGET_NONVISA="${TARGET_NONVISA:-288}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STAGE1_ROOT="${STAGE1_ROOT:-outputs/hparam_all_datasets_stage1}"
VISA_ROOT="${VISA_ROOT:-outputs/hparam_visa_stage1}"
TRIALS=(
  proto_only
  proto_only_topk001
  proto_only_topk010
  score_only
  score_proto_only
  score_proto_topo_tiny
  topo_tiny
  graph_sigma25
  lr3e-4
)

count_nonvisa() {
  find "$STAGE1_ROOT/results" -path '*/visa/*' -prune -o -name '*_train_metrics.json' -print 2>/dev/null | wc -l
}

gpu_process_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {n++} END {print n+0}'
}

run_repair_if_idle() {
  local current
  current="$(count_nonvisa)"
  if [ "$current" -ge "$TARGET_NONVISA" ]; then
    return
  fi
  if [ "$(gpu_process_count)" -ne 0 ]; then
    return
  fi
  echo "[visa-watch] non-ViSA stage1 incomplete (${current}/${TARGET_NONVISA}) and GPU idle; running conservative repair"
  "$PY" -u scripts/run_parallel_hparam_tuning.py \
    --output-root "$STAGE1_ROOT" \
    --log-root outputs/logs/hparam_all_datasets_stage1_trials \
    --epochs 80 --patience 15 --min-epochs 25 \
    --selection-metric selection_score_pro \
    --max-parallel 6 \
    --max-gpu-memory-used-mb 18000 \
    --max-gpu-processes 8 \
    --gpu-budget-poll-seconds 30 \
    --launch-delay-seconds 20 \
    --trial-names "${TRIALS[@]}" \
    --datasets realiad_1024 \
    --skip-running \
    --retries 2 \
    --no-summary
}

echo "[visa-watch] preparing ViSA MVTec-style view"
"$PY" scripts/prepare_dataset_views.py --datasets-root datasets --skip-realiad

while [ "$(count_nonvisa)" -lt "$TARGET_NONVISA" ]; do
  echo "[visa-watch] waiting for non-ViSA stage1: $(count_nonvisa)/${TARGET_NONVISA}; gpu_processes=$(gpu_process_count)"
  run_repair_if_idle
  sleep "$POLL_SECONDS"
done

echo "[visa-watch] non-ViSA stage1 complete; starting ViSA tuning"
"$PY" -u scripts/run_parallel_hparam_tuning.py \
  --output-root "$VISA_ROOT" \
  --log-root outputs/logs/hparam_visa_stage1_trials \
  --epochs 80 --patience 15 --min-epochs 25 \
  --selection-metric selection_score_pro \
  --max-parallel 8 \
  --max-gpu-memory-used-mb 20500 \
  --max-gpu-processes 12 \
  --gpu-budget-poll-seconds 30 \
  --launch-delay-seconds 20 \
  --trial-names "${TRIALS[@]}" \
  --datasets visa \
  --skip-running \
  --retries 2

echo "[visa-watch] ViSA tuning complete; starting ViSA official final-split run"
"$PY" scripts/run_downloaded_official_experiments.py \
  --datasets visa \
  --best-config-root "$VISA_ROOT/best_configs" \
  --output-tag visa_official_tuned \
  --write-bash outputs/logs/official_visa_queue.sh \
  --skip-finished

bash outputs/logs/official_visa_queue.sh
