#!/usr/bin/env python3
"""Minimal dependency-free checks for the released RoboJudge reward."""

from embodied_judge_reward import compute_score


def main() -> None:
    exact = compute_score(
        '{"reasoning":"The grasp remains stable.","physical_adherence":4}',
        "4",
        extra_info={"metric": "physical_adherence"},
    )
    adjacent = compute_score(
        '{"reasoning":"The task is mostly complete.","instruction_alignment":3}',
        "4",
        extra_info={"metric": "instruction_alignment"},
    )
    invalid = compute_score("score: 4", "4", extra_info={"metric": "physical_adherence"})
    assert exact == {"score": 1.0, "accuracy": 1.0, "format": 0.1}
    assert adjacent == {"score": 0.5, "accuracy": 0.5, "format": 0.1}
    assert invalid == {"score": 0.0, "accuracy": 0.0, "format": 0.0}
    print("reward checks passed")


if __name__ == "__main__":
    main()
