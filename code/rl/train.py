#!/usr/bin/env python3
"""GRPO training for RoboJudge-9B with video-aligned PA/IA rewards."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

# Keep heavy ML imports inside main so startup prints data/preflight progress promptly.

SFT_IMAGE_MAX_PIXELS = 262144
SFT_IMAGE_MIN_PIXELS = 1024
SFT_VIDEO_FPS = 4.0
SFT_VIDEO_MAXLEN = 32
SFT_VIDEO_MAX_PIXELS = 65536
SFT_VIDEO_MIN_PIXELS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="append", required=True, help="SFT json file. Repeat for PA+IA.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root", default="data_root")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-media", action="store_true", help="Stat media files while loading rows; slow on full NAS data.")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--steps-per-generation", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-only-model", action="store_true", help="Save model weights only; skip optimizer/scheduler/RNG checkpoint state.")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--per-token-logps-batch-size", type=int, default=1, help="Chunk size for policy logprob/entropy forward; 1 minimizes VRAM without changing GRPO batch semantics.")
    return parser.parse_args()


def resolve_media(path: str, data_root: str) -> str:
    if os.path.isabs(path):
        return path
    parts = Path(path).parts
    if parts and parts[0] == "data_root":
        parts = parts[1:]
    return str(Path(data_root).joinpath(*parts))


def split_marker_prefix(text: str) -> tuple[list[str], str]:
    markers = re.findall(r"^(?:<(?:image|video)>)+", text)
    if not markers:
        return [], text
    prefix = markers[0]
    return re.findall(r"<(image|video)>", prefix), text[len(prefix) :].lstrip()


def build_qwen_vl_messages(messages: list[dict[str, Any]], image_paths: list[str], video_paths: list[str]) -> list[dict[str, Any]]:
    prompt_messages = deepcopy(messages)
    image_idx = 0
    video_idx = 0
    for message in prompt_messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        marker_order, text = split_marker_prefix(content)
        if not marker_order and (image_paths or video_paths):
            marker_order = ["image"] * len(image_paths) + ["video"] * len(video_paths)
        parts: list[dict[str, Any]] = []
        for marker in marker_order:
            if marker == "image":
                if image_idx >= len(image_paths):
                    raise ValueError("image marker count exceeds images list")
                parts.append(
                    {
                        "type": "image",
                        "image": image_paths[image_idx],
                        "max_pixels": SFT_IMAGE_MAX_PIXELS,
                        "min_pixels": SFT_IMAGE_MIN_PIXELS,
                    }
                )
                image_idx += 1
            elif marker == "video":
                if video_idx >= len(video_paths):
                    raise ValueError("video marker count exceeds videos list")
                parts.append(
                    {
                        "type": "video",
                        "video": video_paths[video_idx],
                        "fps": SFT_VIDEO_FPS,
                        "max_frames": SFT_VIDEO_MAXLEN,
                        "max_pixels": SFT_VIDEO_MAX_PIXELS,
                        "min_pixels": SFT_VIDEO_MIN_PIXELS,
                    }
                )
                video_idx += 1
        parts.append({"type": "text", "text": text})
        message["content"] = parts
        break
    if image_idx != len(image_paths) or video_idx != len(video_paths):
        raise ValueError(f"unused media: images {image_idx}/{len(image_paths)}, videos {video_idx}/{len(video_paths)}")
    return prompt_messages


def assistant_score(messages: list[dict[str, Any]]) -> tuple[str, str]:
    assistant = messages[-1]
    if assistant.get("role") != "assistant":
        raise ValueError("last message is not assistant")
    payload = json.loads(assistant["content"])
    if "instruction_alignment" in payload:
        return "instruction_alignment", str(payload["instruction_alignment"])
    if "physical_adherence" in payload:
        return "physical_adherence", str(payload["physical_adherence"])
    raise ValueError("assistant payload has no known score key")


def load_rows(paths: list[str], limit: int, data_root: str, shuffle: bool, seed: int, check_media: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing = 0
    for path in paths:
        raw_rows = json.load(open(path, encoding="utf-8"))
        for raw in raw_rows:
            images = [resolve_media(p, data_root) for p in raw.get("images") or []]
            videos = [resolve_media(p, data_root) for p in raw.get("videos") or []]
            if check_media and any(not os.path.exists(p) for p in images + videos):
                missing += 1
                continue
            metric, truth = assistant_score(raw["messages"])
            # RL prompt must not include the labeled assistant answer; media/template parameters still match SFT.
            prompt = build_qwen_vl_messages(raw["messages"][:-1], images, videos)
            rows.append(
                {
                    "prompt": json.dumps(prompt, ensure_ascii=False),
                    "ground_truth": truth,
                    "extra_info": {"metric": metric},
                    "row_id": raw.get("item_id"),
                }
            )
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError("no usable SFT-aligned rows found")
    print("loaded_rows", len(rows), "missing_media_rows", missing, flush=True)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["extra_info"]["metric"]] = counts.get(row["extra_info"]["metric"], 0) + 1
    print("metric_counts", counts, flush=True)
    return rows


def embodied_reward(completions: list[Any], ground_truth: list[str], extra_info: list[dict[str, Any]], **_: Any) -> list[float]:
    from embodied_judge_reward import compute_score

    scores: list[float] = []
    for completion, truth, info in zip(completions, ground_truth, extra_info, strict=True):
        if isinstance(completion, list) and completion and isinstance(completion[0], dict):
            text = completion[0].get("content", "")
        else:
            text = str(completion)
        result = compute_score(text, truth, extra_info=info)
        scores.append(float(result.get("score", 0.0)) + float(result.get("format", 0.0)))
    return scores


def decode_prompt(prompt: Any) -> Any:
    return json.loads(prompt) if isinstance(prompt, str) else prompt


def collate_video_kwargs(processor: Any, prompts: list[list[dict[str, Any]]]) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    rendered = processor.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        prompts,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_metadata = None
    if video_inputs and isinstance(video_inputs[0], tuple):
        video_metadata = [metadata for _, metadata in video_inputs]
        video_inputs = [video for video, _ in video_inputs]
    processor_kwargs: dict[str, Any] = dict(
        text=rendered,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    if video_metadata is not None:
        processor_kwargs["video_metadata"] = video_metadata
    return processor(**processor_kwargs)


def count_media(prompt: list[dict[str, Any]], media_type: str) -> int:
    return sum(
        1
        for message in prompt
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        if part.get("type") == media_type
    )


def repeat_mm_rows(value: Any, grid_thw: Any, repeat_factor: int) -> Any:
    import torch

    if repeat_factor == 1:
        return value
    sizes = grid_thw.prod(dim=-1).tolist()
    chunks = list(value.split(sizes, dim=0))
    return torch.cat([chunk for chunk in chunks for _ in range(repeat_factor)], dim=0)


def main() -> None:
    args = parse_args()
    print("script_start", flush=True)

    rows = load_rows(args.data, args.limit, args.data_root, args.shuffle, args.seed, args.check_media)
    print("importing_datasets", flush=True)
    from datasets import Dataset

    dataset = Dataset.from_list(rows)

    print("loading_processor", args.ckpt, flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import torch

    lock_path = "/tmp/robojudge_qwen_processor_load.lock"
    try:
        from filelock import FileLock
    except Exception:
        FileLock = None
    if FileLock is not None:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        with FileLock(lock_path, timeout=600):
            print("processor_load_enter", {"rank": rank, "lock": lock_path}, flush=True)
            processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
            print("processor_load_exit", {"rank": rank, "processor": type(processor).__name__}, flush=True)
            time.sleep(0.5)
    else:
        print("processor_load_no_filelock", flush=True)
        processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    print("processor_loaded", type(processor).__name__, flush=True)
    print("sft_params", {
        "video_fps": SFT_VIDEO_FPS,
        "video_maxlen": SFT_VIDEO_MAXLEN,
        "video_max_pixels": SFT_VIDEO_MAX_PIXELS,
        "video_min_pixels": SFT_VIDEO_MIN_PIXELS,
        "image_max_pixels": SFT_IMAGE_MAX_PIXELS,
        "image_min_pixels": SFT_IMAGE_MIN_PIXELS,
    }, flush=True)
    probe_inputs = collate_video_kwargs(processor, [decode_prompt(rows[0]["prompt"])])
    print(
        "multimodal_probe",
        {key: tuple(value.shape) for key, value in probe_inputs.items() if hasattr(value, "shape")},
        flush=True,
    )
    rewards = embodied_reward(
        ['{"reasoning":"ok","%s":%s}' % (rows[0]["extra_info"]["metric"], rows[0]["ground_truth"])],
        [rows[0]["ground_truth"]],
        [rows[0]["extra_info"]],
    )
    print("reward_probe", rewards, flush=True)
    if args.dry_run:
        print("dry_run_ok", flush=True)
        return

    from trl import GRPOConfig, GRPOTrainer

    class SFTAlignedVideoGRPOTrainer(GRPOTrainer):
        def _tokenize_prompts(self, prompts: list):
            prompts = [decode_prompt(prompt) for prompt in prompts]
            tokenized = collate_video_kwargs(self.processing_class, prompts)
            prompt_ids = tokenized["input_ids"]
            multimodal_fields = {k: v for k, v in tokenized.items() if k not in ("input_ids", "attention_mask")}
            return prompt_ids, None, multimodal_fields

        def _generate_and_score_completions(self, inputs):
            inputs = [{**example, "prompt": decode_prompt(example["prompt"])} for example in inputs]
            prompts = [example["prompt"] for example in inputs]
            tokenized = collate_video_kwargs(self.processing_class, prompts)
            output = super()._generate_and_score_completions(inputs)
            repeat_factor = len(output["prompt_ids"]) // len(prompts)
            for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "mm_token_type_ids"]:
                if key not in tokenized:
                    continue
                value = tokenized[key]
                if key in ("image_grid_thw", "video_grid_thw"):
                    value = value.repeat_interleave(repeat_factor, dim=0)
                elif key == "pixel_values" and "image_grid_thw" in tokenized:
                    value = repeat_mm_rows(value, tokenized["image_grid_thw"], repeat_factor)
                elif key == "pixel_values_videos" and "video_grid_thw" in tokenized:
                    value = repeat_mm_rows(value, tokenized["video_grid_thw"], repeat_factor)
                elif key == "mm_token_type_ids":
                    prompt_ids = output["prompt_ids"]
                    completion_ids = output["completion_ids"]
                    value = value.repeat_interleave(repeat_factor, dim=0)
                    if value.size(1) < prompt_ids.size(1):
                        pad = value.new_zeros((value.size(0), prompt_ids.size(1) - value.size(1)))
                        value = torch.cat([pad, value], dim=1)
                    value = torch.cat([value, value.new_zeros(completion_ids.shape)], dim=1)
                output[key] = value.to(self.accelerator.device) if isinstance(value, torch.Tensor) else value
            output["num_images"] = [count_media(prompt, "image") for prompt in prompts for _ in range(repeat_factor)]
            output["num_videos"] = [count_media(prompt, "video") for prompt in prompts for _ in range(repeat_factor)]
            if self.accelerator.is_main_process and self.state.global_step < 3:
                print(
                    "full_debug_subclass_output",
                    {
                        key: tuple(output[key].shape) if hasattr(output.get(key), "shape") else output.get(key)
                        for key in ["prompt_ids", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "num_images", "num_videos"]
                        if key in output
                    },
                    flush=True,
                )
            return output

    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map=None,
    )
    model.config.use_cache = False

    train_args = GRPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        steps_per_generation=args.steps_per_generation,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        optim=args.optim,
        use_vllm=False,
        beta=0.0,
        bf16=True,
        gradient_checkpointing=False,
        report_to="none",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_only_model=args.save_only_model,
        save_strategy="steps",
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        log_completions=False,
    )
    trainer = SFTAlignedVideoGRPOTrainer(
        model=model,
        reward_funcs=embodied_reward,
        args=train_args,
        train_dataset=dataset,
        processing_class=processor,
    )
    trainer.per_token_logps_batch_size = args.per_token_logps_batch_size if args.per_token_logps_batch_size > 0 else None
    print("per_token_logps_batch_size", trainer.per_token_logps_batch_size, flush=True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.out)
    print("train_ok", flush=True)


if __name__ == "__main__":
    main()
