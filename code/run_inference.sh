#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
GOLD="${GOLD:-${ROOT}/data/benchmark/robojudge_bench.jsonl}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data_root}"
OUT="${OUT:-${ROOT}/outputs/predictions.jsonl}"
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
LIMIT="${LIMIT:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
if [[ -z "${CKPT:-}" ]]; then echo "ERROR: set CKPT" >&2; exit 2; fi
mkdir -p "$(dirname "$OUT")"
exec "$PYTHON" "${ROOT}/code/inference.py" --ckpt "$CKPT" --gold "$GOLD" --data-root "$DATA_ROOT" --out "$OUT" --shard "$SHARD" --num-shards "$NUM_SHARDS" --limit "$LIMIT" --max-new-tokens "$MAX_NEW_TOKENS" --attn-implementation "$ATTN_IMPLEMENTATION"
