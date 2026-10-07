#!/usr/bin/env bash
set -euo pipefail

# RoboJudge full-parameter SFT launcher.
# Required env vars:
#   MODEL_NAME_OR_PATH: base Qwen3.5-9B model path or HF id
#   OUTPUT_DIR: destination checkpoint directory
# Optional env vars:
#   DATA_ROOT: downloaded Hugging Face dataset root; defaults to ${ROOT}/data_root
#   SOURCE_DATA_DIR: annotation directory; defaults to ${ROOT}/data
#   LLAMAFACTORY_CLI: llamafactory-cli executable
#   NPROC_PER_NODE: number of GPUs for LLaMAFactory torchrun; defaults to 1
#   PREPARED_DATA_DIR: generated LLaMAFactory dataset directory
#   CONFIG_OUT: rendered config path

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ROOT
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data_root}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-${ROOT}/data}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${ROOT}/code/sft/train.yaml}"
CONFIG_OUT="${CONFIG_OUT:-${OUTPUT_DIR:-${ROOT}/outputs/sft}.train_config.yaml}"

if [[ -z "${MODEL_NAME_OR_PATH:-}" || -z "${OUTPUT_DIR:-}" ]]; then
  echo "ERROR: set MODEL_NAME_OR_PATH and OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-${OUTPUT_DIR}.prepared_data}"
python3 "${ROOT}/code/sft/prepare_data.py" \
  --source-data-dir "$SOURCE_DATA_DIR" \
  --data-root "$DATA_ROOT" \
  --output-dir "$PREPARED_DATA_DIR"
export PREPARED_DATA_DIR

python3 - "$CONFIG_TEMPLATE" "$CONFIG_OUT" <<'PY'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding='utf-8').read()
text = text.replace('${MODEL_NAME_OR_PATH}', os.environ['MODEL_NAME_OR_PATH'])
text = text.replace('${OUTPUT_DIR}', os.environ['OUTPUT_DIR'])
text = text.replace('${DATASET_DIR}', os.environ['PREPARED_DATA_DIR'])
text = text.replace('${DEEPSPEED_CONFIG}', os.path.join(os.environ['ROOT'], 'code/sft/ds_z2_config.json'))
open(dst, 'w', encoding='utf-8').write(text)
PY

exec "$LLAMAFACTORY_CLI" train "$CONFIG_OUT"
