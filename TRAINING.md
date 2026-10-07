# RoboJudge training

This release includes the two training stages used for RoboJudge-9B:

1. full-parameter supervised fine-tuning (SFT) with LLaMA-Factory;
2. GRPO refinement with the released ordinal PA/IA reward.

Both stages use the same video contract as inference: 4 fps, at most 32
frames, and 65,536 maximum pixels per frame. The vision tower and multimodal
projector are frozen during SFT; the language model is updated.

## Setup

Use Python 3.11 and a CUDA-compatible PyTorch build, then install:

```bash
python -m pip install -r requirements-train.txt
```

For SFT, install a current LLaMA-Factory checkout with Qwen3.5 multimodal
support:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
python -m pip install -e ./LLaMA-Factory
```

Download the video assets while preserving the Hugging Face repository
layout:

```bash
huggingface-cli download \
  HuggingFriends/RoboJudge-Train \
  --repo-type dataset \
  --local-dir data_root
```

The annotation files are already included under `data/train/`. The SFT
launcher prepares temporary LLaMA-Factory-compatible copies with explicit
video markers and absolute media paths; source annotations are not modified.

## Stage 1: SFT

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3.5-9B \
DATA_ROOT="$PWD/data_root" \
OUTPUT_DIR="$PWD/outputs/robojudge-sft" \
NPROC_PER_NODE=8 \
bash code/sft/run_sft.sh
```

The released configuration uses one epoch, learning rate `1e-5`, per-device
batch size 4, gradient accumulation 4, BF16, and DeepSpeed ZeRO-2. Override a
setting by copying `code/sft/train.yaml` and setting `CONFIG_TEMPLATE` to the
new file.

## Stage 2: GRPO

Start from the SFT checkpoint:

```bash
MODEL_CKPT="$PWD/outputs/robojudge-sft" \
DATA_ROOT="$PWD/data_root" \
OUTPUT_DIR="$PWD/outputs/robojudge-grpo" \
NUM_GPUS=8 \
bash code/rl/run_grpo.sh
```

The GRPO reward validates the exact JSON output schema and scores the ordinal
1–5 prediction: exact agreement receives `1.0`, a one-level error receives
`0.5`, and larger errors receive `0.0`; valid JSON formatting adds `0.1`.
The default launcher uses 8 generations per prompt, learning rate `1e-6`,
BF16 FSDP, and 200 optimization steps.

Before launching a full job, validate data loading and the reward locally:

```bash
python code/sft/prepare_data.py \
  --source-data-dir data \
  --data-root data_root \
  --output-dir /tmp/robojudge-sft-data \
  --limit 2

python code/rl/test_reward.py

python code/rl/train.py \
  --data data/train/physical_adherence.json \
  --data data/train/instruction_alignment.json \
  --data-root data_root \
  --ckpt Qwen/Qwen3.5-9B \
  --out /tmp/robojudge-grpo-smoke \
  --limit 2 \
  --dry-run
```

The final dry run loads the processor, validates multimodal collation, and
checks the reward contract; it therefore requires the base model and sample
videos to be available.
