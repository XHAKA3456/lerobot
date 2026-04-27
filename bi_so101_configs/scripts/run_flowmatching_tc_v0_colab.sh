#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/content/.cache/huggingface}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-xhaka3456/aic_ur5e_sim_v3_tc_v0}"
export DATASET_DIR="${DATASET_DIR:-/content/data/aic_ur5e_sim_v3_tc_v0}"

if [[ "${MODE}" == "smoke" ]]; then
  CONFIG_PATH="${REPO_ROOT}/bi_so101_configs/scripts/train_flowmatching_tc_v0_colab_smoke.yaml"
elif [[ "${MODE}" == "v1-weighting" ]]; then
  CONFIG_PATH="${REPO_ROOT}/bi_so101_configs/scripts/train_flowmatching_tc_v1_weighting_colab.yaml"
elif [[ "${MODE}" == "full" ]]; then
  CONFIG_PATH="${REPO_ROOT}/bi_so101_configs/scripts/train_flowmatching_tc_v0_colab.yaml"
else
  echo "Usage: bash bi_so101_configs/scripts/run_flowmatching_tc_v0_colab.sh [smoke|full|v1-weighting]"
  exit 1
fi

python -m pip install -U pip
python -m pip install -U "huggingface_hub[cli]"
python -m pip install -e ".[transformers-dep]"

mkdir -p "$(dirname "${DATASET_DIR}")"
if [[ ! -f "${DATASET_DIR}/meta/info.json" ]]; then
  hf download "${DATASET_REPO_ID}" \
    --repo-type dataset \
    --local-dir "${DATASET_DIR}"
fi

cd "${REPO_ROOT}"
python -m lerobot.scripts.lerobot_train --config_path "${CONFIG_PATH}"
