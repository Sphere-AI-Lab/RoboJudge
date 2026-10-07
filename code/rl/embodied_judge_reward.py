"""Reward for the embodied-world-judge task.

The policy is asked to output raw JSON only (no <think>/<answer> wrapper),
matching the original SFT annotation format:

    {"reasoning": "<...>", "physical_adherence": <1-5>}
    {"reasoning": "<...>", "instruction_alignment": <1-5>}

Reward design:
  - format reward: 0.1 if the response is exactly a JSON object with a string
    "reasoning" field plus a single integer (1-5) metric field
    ("physical_adherence" or "instruction_alignment"); otherwise 0.0.
  - accuracy reward: only evaluated when the format is well-formed.
      * predicted score == ground truth        -> 1.0
      * |predicted score - ground truth| == 1   -> 0.5
      * otherwise                                -> 0.0
  - total score = format reward + accuracy reward
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "format_reward",
    "acc_reward",
    "compute_score",
    "compute_score_from_data_source",
]

_METRIC_KEYS = ("physical_adherence", "instruction_alignment")
_FORMAT_SCORE = 0.1


def _parse_prediction(predict_str: str, expected_metric: str | None = None) -> tuple[str, int] | None:
    """Return (metric_key, score) if `predict_str` matches the expected JSON format, else None."""
    if not isinstance(predict_str, str):
        return None

    try:
        obj = json.loads(predict_str.strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    if not isinstance(obj, dict):
        return None
    if not isinstance(obj.get("reasoning"), str):
        return None

    present_metric_keys = [key for key in _METRIC_KEYS if key in obj]
    if len(present_metric_keys) != 1:
        return None
    metric_key = present_metric_keys[0]

    # The response must contain exactly {"reasoning", <metric_key>}, nothing else.
    if set(obj.keys()) != {"reasoning", metric_key}:
        return None

    if expected_metric and metric_key != expected_metric:
        return None

    value = obj.get(metric_key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = int(value)
    if not (1 <= score <= 5):
        return None

    return metric_key, score


def _extract_expected_metric(extra_info: Any) -> str | None:
    if isinstance(extra_info, dict):
        metric = extra_info.get("metric")
        if metric in _METRIC_KEYS:
            return metric
    return None


def format_reward(predict_str: str, extra_info: Any = None) -> float:
    expected_metric = _extract_expected_metric(extra_info)
    parsed = _parse_prediction(predict_str, expected_metric=expected_metric)
    return _FORMAT_SCORE if parsed is not None else 0.0


def acc_reward(predict_str: str, ground_truth: str, extra_info: Any = None, **_: Any) -> float:
    expected_metric = _extract_expected_metric(extra_info)
    parsed = _parse_prediction(predict_str, expected_metric=expected_metric)
    if parsed is None:
        return 0.0
    _, predicted_score = parsed

    try:
        truth_score = int(float(str(ground_truth).strip()))
    except (TypeError, ValueError):
        return 0.0

    diff = abs(predicted_score - truth_score)
    if diff == 0:
        return 1.0
    if diff == 1:
        # Restored 2026-09-06. Exact-match-only reward has a degenerate optimum on an ordinal
        # 1-5 scale: always predict the mode. PA's gold mode is 3 at 30.4%, and GRPO@100 reached
        # 0.368 exact-match while its extreme-label use collapsed to 0.5% (gold 24.8%) -- it was
        # optimising exactly what we paid for. Partial credit for being one level off reduces the
        # advantage of that degenerate policy.
        return 0.5
    return 0.0


def compute_score(
    predict_str: str,
    ground_truth: str,
    extra_info: Any = None,
    **_: Any,
) -> dict[str, float]:
    predict_str = str(predict_str)
    ground_truth = str(ground_truth)
    formatting = format_reward(predict_str, extra_info=extra_info)
    accuracy = acc_reward(predict_str, ground_truth, extra_info=extra_info)
    return {
        "score":  accuracy,
        "accuracy": accuracy,
        "format": formatting,
    }


def compute_score_from_data_source(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    **_: dict,
) -> dict[str, float]:
    """Adapter for custom_reward_function configs that expect the default signature."""
    return compute_score(solution_str, ground_truth, extra_info=extra_info)
