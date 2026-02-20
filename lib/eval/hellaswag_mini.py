"""Tiny HellaSwag-style multiple-choice eval.

For each item we have a context and N candidate continuations; the score is
the per-token average log-likelihood of each candidate. The model "answers"
by picking the highest-scoring option, and accuracy is the fraction of items
it picks the correct one.

We ship a small bilingual JSON dataset under ``lib/eval/data/hellaswag_mini.json``
so the eval works out of the box. Replace it with a bigger / official split
when you want a more rigorous benchmark.

Usage:
    python lib/eval/hellaswag_mini.py --checkpoint out/ckpt.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _loader import auto_device, load_checkpoint, resolve_codec, sequence_logprob


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / 'data' / 'hellaswag_mini.json'


def score_item(model, encode, item, device) -> int:
    context_ids = encode(item['context'])
    scores = []
    for choice in item['choices']:
        ids = context_ids + encode(choice)
        if not ids:
            scores.append(float('-inf')); continue
        lp = sequence_logprob(model, ids, device)
        # length-normalise so longer choices aren't penalised
        scores.append(lp / max(1, len(ids) - 1))
    return int(max(range(len(scores)), key=lambda i: scores[i]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default=str(DEFAULT_DATA))
    p.add_argument('--device', default=auto_device())
    args = p.parse_args()

    items = json.loads(Path(args.data).read_text())
    model, ck = load_checkpoint(args.checkpoint, args.device)
    encode, _ = resolve_codec(ck)

    correct = 0
    for i, item in enumerate(items):
        pred = score_item(model, encode, item, args.device)
        ok = pred == item['answer']
        correct += ok
        flag = '✓' if ok else '✗'
        print(f"{flag} [{i+1}/{len(items)}] {item['context'][:50]!r}  "
              f"pred={pred}  ans={item['answer']}")
    acc = correct / len(items) if items else 0.0
    print(f"\nHellaSwag-mini accuracy: {correct}/{len(items)}  ({acc:.1%})")
    return acc


if __name__ == '__main__':
    main()
