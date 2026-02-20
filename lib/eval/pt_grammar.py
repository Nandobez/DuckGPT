"""Brazilian Portuguese grammar multiple-choice (toy).

Each item has a sentence with a blank and 2-4 grammatical / lexical options.
We score each filled sentence by its per-token log-likelihood and pick the
best. Items focus on noun-adjective and subject-verb agreement.

Usage:
    python lib/eval/pt_grammar.py --checkpoint out/ckpt.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _loader import auto_device, load_checkpoint, resolve_codec, sequence_logprob

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / 'data' / 'pt_grammar.json'


def score(model, encode, sentence: str, device: str) -> float:
    ids = encode(sentence)
    if not ids:
        return float('-inf')
    return sequence_logprob(model, ids, device) / max(1, len(ids) - 1)


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
        scores = [score(model, encode, c, args.device) for c in item['choices']]
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        ok = pred == item['answer']
        correct += ok
        print(f"{'✓' if ok else '✗'} [{i+1}/{len(items)}] {item['category']:>18}: "
              f"pred={pred} ans={item['answer']}")
    acc = correct / len(items) if items else 0.0
    print(f"\nPT-Grammar accuracy: {correct}/{len(items)}  ({acc:.1%})")
    return acc


if __name__ == '__main__':
    main()
