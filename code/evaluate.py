#!/usr/bin/env python3
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

pred_path = Path(sys.argv[1])
gold_path = Path(sys.argv[2])

SCORE_PATTERNS = [
    re.compile(r'"physical_adherence"\s*:\s*([1-5])'),
    re.compile(r'"instruction_alignment"\s*:\s*([1-5])'),
    re.compile(r'(?:score|rating)\s*[:=]\s*([1-5])', re.I),
    re.compile(r'\b([1-5])\s*/\s*5\b'),
    re.compile(r'^\s*([1-5])\b'),
]


def get_score(text):
    s = str(text)
    for pattern in SCORE_PATTERNS:
        match = pattern.search(s)
        if match:
            return int(match.group(1))
    nums = re.findall(r'\b[1-5]\b', s)
    return int(nums[-1]) if nums else None


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float('nan')


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def item_id(record):
    for key in ('item_id', 'id', 'video_id', 'uid'):
        if record.get(key):
            return record[key]
    raise KeyError(f'no item id keys in {record.keys()}')


def resolve_item_id(record, gold):
    iid = str(item_id(record)).removesuffix('.mp4')
    if (iid, 'PA') in gold or (iid, 'IA') in gold:
        return iid
    prefixed = f'data__{iid}'
    return prefixed if (prefixed, 'PA') in gold or (prefixed, 'IA') in gold else iid


def load_gold(path):
    gold = {}
    for line in path.open(encoding='utf-8'):
        if not line.strip():
            continue
        record = json.loads(line)
        iid = item_id(record)
        axis = str(record.get('axis') or record.get('score_type') or '').upper()
        if axis in ('PA', 'PHYSICAL_ADHERENCE'):
            keys = ('physical_adherence', 'gold', 'score', 'label')
            out_axis = 'PA'
        elif axis in ('IA', 'INSTRUCTION_ALIGNMENT'):
            keys = ('instruction_alignment', 'gold', 'score', 'label')
            out_axis = 'IA'
        else:
            for out_axis, keys in [('PA', ('physical_adherence', 'pa', 'pa_score')), ('IA', ('instruction_alignment', 'ia', 'ia_score'))]:
                for key in keys:
                    if record.get(key) is not None:
                        gold[(iid, out_axis)] = int(round(float(record[key])))
                        break
            continue
        for key in keys:
            if record.get(key) is not None:
                gold[(iid, out_axis)] = int(round(float(record[key])))
                break
    return gold


gold = load_gold(gold_path)
rows = defaultdict(list)
nulls = defaultdict(int)
missing_gold = defaultdict(int)
for line in pred_path.open(encoding='utf-8'):
    if not line.strip():
        continue
    record = json.loads(line)
    iid = resolve_item_id(record, gold)
    direct_scores = {
        'PA': record.get('physical_adherence'),
        'IA': record.get('instruction_alignment'),
    }
    if any(value is not None for value in direct_scores.values()):
        for direct_axis, value in direct_scores.items():
            if value is None:
                continue
            try:
                pred = float(value)
            except (TypeError, ValueError):
                nulls[direct_axis] += 1
                continue
            g = gold.get((iid, direct_axis))
            if g is None:
                missing_gold[direct_axis] += 1
                continue
            rows[direct_axis].append((g, pred))
        continue
    axis = str(record.get('axis') or '').upper()
    if axis not in ('PA', 'IA'):
        prompt = str(record.get('prompt', ''))
        prediction = str(record.get('predict', ''))
        if 'physical_adherence' in prediction or 'PHYSICAL REALISM' in prompt:
            axis = 'PA'
        elif 'instruction_alignment' in prediction or 'manipulation task' in prompt:
            axis = 'IA'
        else:
            continue
    pred = get_score(record.get('predict'))
    if pred is None:
        nulls[axis] += 1
        continue
    g = gold.get((iid, axis))
    if g is None:
        missing_gold[axis] += 1
        continue
    rows[axis].append((g, pred))

for axis in ('PA', 'IA'):
    pairs = rows[axis]
    gs = [g for g, _ in pairs]
    ps = [p for _, p in pairs]
    n = len(pairs)
    if n == 0:
        print(f'{axis} n=0 null={nulls[axis]} missing_gold={missing_gold[axis]}')
        print('  pearson=nan')
        print('  spearman=nan')
        print('  exact=nan')
        print('  relaxed=nan')
        print('  mse=nan')
        continue
    rounded = [min(5, max(1, math.floor(p + 0.5))) for p in ps]
    exact = sum(g == p for g, p in zip(gs, rounded)) / n
    relaxed = sum(abs(g - p) <= 1 for g, p in zip(gs, rounded)) / n
    mse = sum((g - p) ** 2 for g, p in pairs) / n
    print(f'{axis} n={n} null={nulls[axis]} missing_gold={missing_gold[axis]}')
    print(f'  pearson={pearson(gs, ps):.6f}')
    print(f'  spearman={pearson(ranks(gs), ranks(ps)):.6f}')
    print(f'  exact={exact:.6f}')
    print(f'  relaxed={relaxed:.6f}')
    print(f'  mse={mse:.6f}')

pooled_pairs = rows['PA'] + rows['IA']
if not pooled_pairs:
    raise SystemExit('no scored predictions matched the gold file')
if rows['PA'] and rows['IA']:
    pooled_gold = [gold_score for gold_score, _ in pooled_pairs]
    pooled_pred = [pred_score for _, pred_score in pooled_pairs]
    overall_exact = sum(g == min(5, max(1, math.floor(p + 0.5))) for g, p in pooled_pairs) / len(pooled_pairs)
    overall_mse = sum((g - p) ** 2 for g, p in pooled_pairs) / len(pooled_pairs)
    print(f'Overall n={len(pooled_pairs)}')
    print(f'  pearson={pearson(pooled_gold, pooled_pred):.6f}')
    print(f'  exact={overall_exact:.6f}')
    print(f'  mse={overall_mse:.6f}')
else:
    print('Overall unavailable because one evaluation axis is missing')
