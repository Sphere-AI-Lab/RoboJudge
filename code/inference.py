#!/usr/bin/env python3
"""RoboJudge-9B Hugging Face inference for PA and IA.

Videos are passed as structured Qwen video inputs, decoded by
qwen_vl_utils.process_vision_info, and uniformly sampled at the released
RoboJudge settings. IA follows the same video-plus-instruction contract used by
RoboJudge-Bench.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from typing import Any

HF_PREFIXES = (
    "https://huggingface.co/datasets/HuggingFriends/RoboJudge-Bench/resolve/main/",
    "https://huggingface.co/datasets/HuggingFriends/RoboJudge-Train/resolve/main/",
    "https://huggingface.co/datasets/HuggingFriends/mllm-as-embodied-world-judge/resolve/main/",
)

IMAGE_MAX_PIXELS = 262144
IMAGE_MIN_PIXELS = 1024
VIDEO_FPS = 4.0
VIDEO_MAXLEN = 32
VIDEO_MAX_PIXELS = 65536
VIDEO_MIN_PIXELS = 256

PA_SYSTEM_BODY = "You are a strict, calibrated evaluator of the PHYSICAL REALISM of AI-generated embodied / robot-manipulation videos (a robot arm/gripper or a human hand acting on objects). You are shown uniformly-sampled frames of one generated video in temporal order. Judge the physics of the video itself. Be conservative: reserve 5 for clearly flawless physics and 1 for clearly broken physics."

PA_PROMPT_BODY = """Task: Judge the PHYSICAL REALISM of this AI-generated robot / embodied-manipulation
video, from the video alone (ignore any task instruction).

Criteria (your reasoning must address each; you may also note other issues):
1. Agent integrity - the arm/gripper/hand stays structurally complete and consistent
   (no melting, fused/extra fingers, warping).
2. Scene & object consistency - background and objects stay temporally stable
   (no flicker, teleport, morphing, appear/disappear).
3. Interaction realism - contacts obey physics (grasps close and bear weight, no
   interpenetration, motion respects gravity/inertia).

Score (integer 1-5): 1 = gross violations throughout; 2 = major violations;
3 = noticeable local inconsistencies; 4 = minor issues only; 5 = no visible violation.

Reason first, then score. Output JSON only:
{"reasoning": "<assess agent integrity, scene & object consistency, and interaction realism, each with concrete visual evidence>", "physical_adherence": <1-5>}
"""

IA_SYSTEM_BODY = "You are a strict, calibrated evaluator of whether an AI-generated embodied-manipulation video correctly performs a given task instruction. You are shown the instruction and uniformly-sampled frames of one generated video in temporal order; the FIRST frame is the initial scene the video was conditioned on. Judge task execution, not raw visual quality. Be conservative: reserve 5 for full, correct task completion and 1 for unrelated videos."

IA_PROMPT_BODY = """Task: Judge whether this AI-generated video performs the instructed manipulation
task. The first frame is the initial scene the video was conditioned on.

Instruction: "{instruction}"

Criteria (your reasoning must address each; you may also note other issues):
1. Agent match - the task is done by the SAME manipulator shown in the first frame
   (not a different/new agent).
2. Object correctness - the manipulated object is the instruction's target object.
3. Goal completion - the instructed goal is actually achieved by the end
   (not merely approached).

Score (integer 1-5): 1 = unrelated or task not performed; 2 = major misalignment;
3 = partial completion; 4 = minor shortfalls only; 5 = full, correct completion.

Reason first, then score. Output JSON only:
{{"reasoning": "<assess agent match, object correctness, and goal completion, each with concrete evidence>", "instruction_alignment": <1-5>}}
"""


def _local(url: str | None, root: str) -> str:
    url = (url or "").strip()
    for prefix in HF_PREFIXES:
        if url.startswith(prefix):
            return os.path.join(root, url[len(prefix):])
    return url


def _instruction_of(row: dict[str, Any], root: str) -> str | None:
    for key in ("instruction", "instruction_text", "prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value and isinstance(value[0], str) and value[0].strip():
            return value[0].strip()
    init = _local(row.get("init_frame_url"), root)
    cands: list[str] = []
    if row.get("instruction_url"):
        cands.append(_local(row.get("instruction_url"), root))
    cands.extend([
        os.path.join(os.path.dirname(init), "instruction.txt"),
        os.path.join(os.path.dirname(init), "prompt.txt"),
    ])
    ds, task, ep = row.get("dataset"), row.get("task"), row.get("episode")
    if ds and task and ep:
        prompt_dir = os.path.join(root, "data", str(ds), "gt_data", str(task), str(ep), "prompt")
        cands.extend([os.path.join(prompt_dir, "instruction.txt"), os.path.join(prompt_dir, "prompt.txt")])
    for path in cands:
        if os.path.exists(path):
            text = open(path, encoding="utf-8").read().strip()
            if text:
                return text
    return None


def split_marker_prefix(text: str) -> tuple[list[str], str]:
    markers = re.findall(r"^(?:<(?:image|video)>)+", text)
    if not markers:
        return [], text
    prefix = markers[0]
    return re.findall(r"<(image|video)>", prefix), text[len(prefix):].lstrip()


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
                parts.append({
                    "type": "image",
                    "image": image_paths[image_idx],
                    "max_pixels": IMAGE_MAX_PIXELS,
                    "min_pixels": IMAGE_MIN_PIXELS,
                })
                image_idx += 1
            elif marker == "video":
                if video_idx >= len(video_paths):
                    raise ValueError("video marker count exceeds videos list")
                parts.append({
                    "type": "video",
                    "video": video_paths[video_idx],
                    "fps": VIDEO_FPS,
                    "max_frames": VIDEO_MAXLEN,
                    "max_pixels": VIDEO_MAX_PIXELS,
                    "min_pixels": VIDEO_MIN_PIXELS,
                })
                video_idx += 1
        parts.append({"type": "text", "text": text})
        message["content"] = parts
        break
    if image_idx != len(image_paths) or video_idx != len(video_paths):
        raise ValueError(f"unused media: images {image_idx}/{len(image_paths)}, videos {video_idx}/{len(video_paths)}")
    return prompt_messages


def apply_chat_template(processor: Any, prompts: list[list[dict[str, Any]]], *, enable_thinking: bool) -> tuple[Any, bool]:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return processor.apply_chat_template(prompts, enable_thinking=enable_thinking, **kwargs), enable_thinking
    except TypeError:
        return processor.apply_chat_template(prompts, **kwargs), True


def collate_video_kwargs(processor: Any, prompts: list[list[dict[str, Any]]]) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    rendered, _ = apply_chat_template(processor, prompts, enable_thinking=False)
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
    kwargs: dict[str, Any] = dict(
        text=rendered,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    if video_metadata is not None:
        kwargs["video_metadata"] = video_metadata
    return processor(**kwargs)


def make_prompt(axis: str, row: dict[str, Any], video_path: str, instruction: str | None) -> list[dict[str, Any]]:
    if axis == "PA":
        system = PA_SYSTEM_BODY
        user = PA_PROMPT_BODY
    elif axis == "IA":
        if instruction is None:
            raise ValueError("IA requires instruction")
        system = IA_SYSTEM_BODY
        user = IA_PROMPT_BODY.format(instruction=instruction)
    else:
        raise ValueError(axis)
    return build_qwen_vl_messages(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        image_paths=[],
        video_paths=[video_path],
    )


def merge(args: argparse.Namespace) -> None:
    rows = []
    for path in args.merge:
        rows.extend(json.loads(line) for line in open(path, encoding="utf-8") if line.strip())
    gold_order = {json.loads(line)["item_id"]: i for i, line in enumerate(open(args.gold, encoding="utf-8"))}
    pa = sorted([r for r in rows if r["axis"] == "PA"], key=lambda r: gold_order[r["item_id"]])
    ia = sorted([r for r in rows if r["axis"] == "IA"], key=lambda r: gold_order[r["item_id"]])
    n = len(gold_order)
    if len(pa) != n or len(ia) != n:
        print(f"WARNING: expected {n} per axis, got PA={len(pa)} IA={len(ia)}", file=sys.stderr)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in pa + ia:
            fh.write(json.dumps({k: row[k] for k in ("item_id", "prompt", "predict")}, ensure_ascii=False) + "\n")
    print(f"merged {len(pa)+len(ia)} rows -> {args.out} (PA {len(pa)}, IA {len(ia)})")
    if len(pa) != n or len(ia) != n:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--enable-thinking", action="store_true", help="kept for CLI compatibility; default runner disables Qwen thinking in chat template")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--merge", nargs="*")
    args = parser.parse_args()
    if args.merge:
        return merge(args)
    if not args.ckpt:
        sys.exit("--ckpt required unless --merge is used")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = [json.loads(line) for line in open(args.gold, encoding="utf-8") if line.strip()]
    mine = [row for i, row in enumerate(rows) if i % args.num_shards == args.shard]
    if args.limit > 0:
        mine = mine[:args.limit]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} rows", flush=True)
    print("robojudge_standard", {
        "video_fps": VIDEO_FPS,
        "video_maxlen": VIDEO_MAXLEN,
        "video_max_pixels": VIDEO_MAX_PIXELS,
        "video_min_pixels": VIDEO_MIN_PIXELS,
        "ia_init_image": False,
        "uses_qwen_vl_utils_process_vision_info": True,
    }, flush=True)

    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model_kwargs: dict[str, Any] = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda:0")
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(args.ckpt, **model_kwargs)
    model.eval()

    def run(prompt: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
        inputs = collate_video_kwargs(processor, [prompt])
        rendered, _ = apply_chat_template(
            processor,
            [prompt],
            enable_thinking=False,
        )
        rendered = rendered[0]
        meta = {}
        for key in ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts"):
            value = inputs.get(key)
            if hasattr(value, "shape"):
                meta[key] = tuple(value.shape)
            elif value is not None:
                meta[key] = str(value)
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return rendered, processor.decode(gen, skip_special_tokens=True), meta

    written = skipped = no_instr = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, row in enumerate(mine):
            video_path = _local(row.get("video_url"), args.data_root)
            if not os.path.exists(video_path):
                skipped += 1
                print(f"missing video {row.get('item_id')}: {video_path}", file=sys.stderr)
                continue
            instr = _instruction_of(row, args.data_root)
            pa_prompt = make_prompt("PA", row, video_path, None)
            pa_text, pa_pred, pa_meta = run(pa_prompt)
            fh.write(json.dumps({"item_id": row["item_id"], "axis": "PA", "prompt": pa_text, "predict": pa_pred, "meta": pa_meta}, ensure_ascii=False) + "\n")
            if instr is None:
                no_instr += 1
            else:
                ia_prompt = make_prompt("IA", row, video_path, instr)
                ia_text, ia_pred, ia_meta = run(ia_prompt)
                fh.write(json.dumps({"item_id": row["item_id"], "axis": "IA", "prompt": ia_text, "predict": ia_pred, "meta": ia_meta}, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(mine)} items", flush=True)
    print(f"done written_items={written} skipped={skipped} no_instr={no_instr} out={args.out}")
    if written == 0 or no_instr:
        sys.exit(1)


if __name__ == "__main__":
    main()
