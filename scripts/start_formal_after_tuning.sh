#!/usr/bin/env bash
set -euo pipefail

tuning_pid="${1:?usage: $0 TUNING_PID [best_config_root] [formal_queue] [formal_log] [formal_pid_file]}"
best_config_root="${2:-outputs/hparam_tuning/best_configs}"
formal_queue="${3:-outputs/logs/official_experiments_queue.sh}"
formal_log="${4:-outputs/logs/official_experiments.log}"
formal_pid_file="${5:-outputs/logs/official_experiments.pid}"
PY="${PY:-/root/miniconda3/envs/nostdiffad/bin/python}"

echo "[watcher] waiting for tuning pid ${tuning_pid}"
while kill -0 "${tuning_pid}" 2>/dev/null; do
  sleep 60
done

echo "[watcher] tuning pid ${tuning_pid} exited; checking tuned configs"
mapfile -t datasets < <("${PY}" - <<'PY'
from collections import Counter
import sys
sys.path[:0] = ["src", "scripts"]
from run_downloaded_official_experiments import downloaded_experiments
for name in sorted(Counter(experiment.dataset for experiment in downloaded_experiments())):
    print(name)
PY
)
for dataset in "${datasets[@]}"; do
  config="${best_config_root}/${dataset}.yaml"
  if [[ ! -s "${config}" ]]; then
    echo "[watcher] missing tuned config: ${config}" >&2
    exit 1
  fi
done

"${PY}" scripts/run_downloaded_official_experiments.py \
  --best-config-root "${best_config_root}" \
  --write-bash "${formal_queue}" \
  --skip-finished

echo "[watcher] starting formal queue: ${formal_queue}"
nohup setsid bash "${formal_queue}" > "${formal_log}" 2>&1 < /dev/null &
echo "$!" > "${formal_pid_file}"
echo "[watcher] formal pid $(cat "${formal_pid_file}")"
