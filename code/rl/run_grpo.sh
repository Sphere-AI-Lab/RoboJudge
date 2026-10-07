#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NUM_GPUS="${NUM_GPUS:-4}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-50}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data_root}"
DATA_PA="${DATA_PA:-${ROOT}/data/train/physical_adherence.json}"
DATA_IA="${DATA_IA:-${ROOT}/data/train/instruction_alignment.json}"

if [[ -z "${MODEL_CKPT:-}" || -z "${OUTPUT_DIR:-}" ]]; then
  echo "ERROR: set MODEL_CKPT and OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="${ROOT}/code/rl:${PYTHONPATH:-}"
exec accelerate launch \
  --num_processes "$NUM_GPUS" \
  --mixed_precision bf16 \
  --use_fsdp \
  --fsdp_version 1 \
  --fsdp_sharding_strategy FULL_SHARD \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer \
  --fsdp_backward_prefetch BACKWARD_PRE \
  --fsdp_state_dict_type FULL_STATE_DICT \
  --fsdp_use_orig_params true \
  --fsdp_cpu_ram_efficient_loading true \
  --fsdp_sync_module_states true \
  --fsdp_activation_checkpointing true \
  "${ROOT}/code/rl/train.py" \
  --data "$DATA_IA" \
  --data "$DATA_PA" \
  --data-root "$DATA_ROOT" \
  --ckpt "$MODEL_CKPT" \
  --out "$OUTPUT_DIR" \
  --shuffle \
  --max-steps "$MAX_STEPS" \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --steps-per-generation 1 \
  --num-generations 8 \
  --max-completion-length "$MAX_COMPLETION_LENGTH" \
  --learning-rate 1e-6 \
  --optim adafactor \
  --save-steps "$SAVE_STEPS" \
  --save-only-model \
  --logging-steps 1 \
  --attn-implementation sdpa
