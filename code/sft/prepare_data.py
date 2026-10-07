#!/usr/bin/env python3
"""Prepare RoboJudge-Train annotations for upstream LLaMA-Factory.

The released annotations intentionally keep text prompts free of literal media
markers. LLaMA-Factory's standard Qwen3.5 template expects one ``<video>``
marker for each video, so this script adds the marker and resolves media paths
without modifying the source annotations.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


DATASETS = ("physical_adherence", "instruction_alignment")


def resolve_video(path: str, data_root: Path) -> str:
    source = Path(path)
    if source.is_absolute():
        return str(source)
    parts = source.parts
    if parts and parts[0] == "data_root":
        parts = parts[1:]
    return str(data_root.joinpath(*parts).resolve())


def prepare_row(row: dict, data_root: Path) -> dict:
    output = deepcopy(row)
    videos = [resolve_video(path, data_root) for path in output.get("videos", [])]
    if len(videos) != 1:
        raise ValueError(f"expected one video for {row.get('item_id')}, got {len(videos)}")
    output["videos"] = videos
    user = next((message for message in output["messages"] if message.get("role") == "user"), None)
    if user is None or not isinstance(user.get("content"), str):
        raise ValueError(f"missing user prompt for {row.get('item_id')}")
    if not user["content"].lstrip().startswith("<video>"):
        user["content"] = "<video>" + user["content"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="per-dataset debug limit; 0 keeps all rows")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = {}
    for name in DATASETS:
        source = args.source_data_dir / "train" / f"{name}.json"
        rows = json.loads(source.read_text(encoding="utf-8"))
        if args.limit > 0:
            rows = rows[: args.limit]
        prepared = [prepare_row(row, args.data_root) for row in rows]
        target = args.output_dir / f"{name}.json"
        target.write_text(json.dumps(prepared, ensure_ascii=False), encoding="utf-8")
        dataset_info[name] = {
            "file_name": target.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "videos": "videos"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
        print(f"prepared {len(prepared)} rows -> {target}")
    (args.output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
