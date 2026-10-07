#!/usr/bin/env python3
"""Paired, corpus-stratified source-clip bootstrap for paper results."""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def axis_of(row):
    axis = row.get("axis")
    if axis in ("PA", "IA"):
        return axis
    prompt = str(row.get("prompt", ""))
    prediction = str(row.get("predict", ""))
    if "physical_adherence" in prediction or "PHYSICAL REALISM" in prompt:
        return "PA"
    if "instruction_alignment" in prediction or "manipulation task" in prompt:
        return "IA"
    return None


def parse_score(value, axis):
    key = "physical_adherence" if axis == "PA" else "instruction_alignment"
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    try:
        return float(json.loads(text)[key])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        match = re.search(rf'"{key}"\s*:\s*([1-5](?:\.\d+)?)', text)
        return float(match.group(1)) if match else None


def load_predictions(path, gold_ids):
    predictions = {"PA": {}, "IA": {}}
    for row in read_jsonl(path):
        item = str(row.get("item_id") or row.get("id", "")).removesuffix(".mp4")
        if item not in gold_ids and f"data__{item}" in gold_ids:
            item = f"data__{item}"
        if item not in gold_ids:
            continue
        explicit = False
        for axis, key in (("PA", "physical_adherence"), ("IA", "instruction_alignment")):
            if row.get(key) is not None:
                predictions[axis][item] = float(row[key])
                explicit = True
        if explicit:
            continue
        axis = axis_of(row)
        parsed = parse_score(row.get("predict"), axis) if axis else None
        if parsed is not None:
            predictions[axis][item] = parsed
    return predictions


def pearson(xs, ys):
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return numerator / denominator


def correlation(model, gold, sample, axis):
    key = "physical_adherence" if axis == "PA" else "instruction_alignment"
    return pearson([model[axis][item] for item in sample], [gold[item][key] for item in sample])


def overall_correlation(model, gold, sample):
    predicted = [model["PA"][item] for item in sample] + [model["IA"][item] for item in sample]
    reference = [gold[item]["physical_adherence"] for item in sample] + [gold[item]["instruction_alignment"] for item in sample]
    return pearson(predicted, reference)


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/benchmark/robojudge_bench.jsonl")
    parser.add_argument("--robojudge", required=True, help="RoboJudge-9B prediction JSONL")
    parser.add_argument("--gemini", required=True, help="Gemini-3.7-Flash prediction JSONL")
    parser.add_argument("--gpt", required=True, help="GPT-5.5 prediction JSONL")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output")
    args = parser.parse_args()

    gold_rows = read_jsonl(args.gold)
    gold = {row["item_id"]: row for row in gold_rows}
    ids = list(gold)
    models = {
        "RoboJudge-9B": load_predictions(args.robojudge, set(ids)),
        "Gemini-3.7-Flash": load_predictions(args.gemini, set(ids)),
        "GPT-5.5": load_predictions(args.gpt, set(ids)),
    }
    for name, model in models.items():
        for axis in ("PA", "IA"):
            if set(model[axis]) != set(ids):
                raise ValueError(f"{name} {axis} does not cover all {len(ids)} gold items")

    clusters = defaultdict(list)
    for row in gold_rows:
        clusters[(row["dataset"], row["task"], row["episode"])].append(row["item_id"])
    strata = defaultdict(list)
    for cluster in clusters:
        strata[cluster[0]].append(cluster)

    comparisons = {
        "PA": ("Gemini-3.7-Flash", lambda model, sample: correlation(model, gold, sample, "PA")),
        "IA": ("GPT-5.5", lambda model, sample: correlation(model, gold, sample, "IA")),
        "Overall": ("Gemini-3.7-Flash", lambda model, sample: overall_correlation(model, gold, sample)),
    }
    point = {}
    for metric, (baseline, function) in comparisons.items():
        ours = function(models["RoboJudge-9B"], ids)
        other = function(models[baseline], ids)
        point[metric] = {"baseline": baseline, "robojudge_r": ours, "baseline_r": other, "delta_r": ours - other}

    rng = np.random.default_rng(args.seed)
    bootstrap = {metric: [] for metric in comparisons}
    for _ in range(args.replicates):
        sample = []
        for corpus in sorted(strata):
            corpus_clusters = strata[corpus]
            for index in rng.integers(0, len(corpus_clusters), size=len(corpus_clusters)):
                sample.extend(clusters[corpus_clusters[int(index)]])
        for metric, (baseline, function) in comparisons.items():
            bootstrap[metric].append(function(models["RoboJudge-9B"], sample) - function(models[baseline], sample))

    result = {
        "method": {
            "replicates": args.replicates,
            "seed": args.seed,
            "cluster": "dataset/task/episode",
            "clusters": len(clusters),
            "stratified_by": "dataset",
            "overall": "Pearson correlation after pooling PA and IA axis-level predictions",
        },
        "results": {},
    }
    for metric in comparisons:
        values = bootstrap[metric]
        result["results"][metric] = {**point[metric], "ci_95": [percentile(values, 0.025), percentile(values, 0.975)]}
    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
