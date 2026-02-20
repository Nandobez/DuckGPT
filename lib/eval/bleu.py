"""Tiny EN ↔ PT translation eval (BLEU-4).

The model is given a short instruction (``"Traduza para o inglês: …"`` or
``"Translate to Portuguese: …"``) and prompted to continue. We then compare
its generation against a reference using corpus BLEU-4 with the canonical
brevity-penalty formula (Papineni et al., 2002).

This is *not* a real MT benchmark — it's a sanity test that ranks two
checkpoints sensibly. Pair it with a serious eval (e.g. FLORES) if you need
something publishable.

Usage:
    python lib/eval/bleu.py --checkpoint out/ckpt.pt
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from _loader import auto_device, load_checkpoint, resolve_codec


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / 'data' / 'mini_mt.json'


# ---------------------------------------------------------------------------
# BLEU implementation
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(refs: List[str], hyps: List[str], n_max: int = 4) -> float:
    weights = [1.0 / n_max] * n_max
    precisions = [0.0] * n_max
    matches = [0] * n_max
    totals = [0] * n_max
    hyp_len = 0
    ref_len = 0
    for ref, hyp in zip(refs, hyps):
        r_tok = ref.split()
        h_tok = hyp.split()
        hyp_len += len(h_tok)
        ref_len += len(r_tok)
        for n in range(1, n_max + 1):
            h_ng = _ngrams(h_tok, n)
            r_ng = _ngrams(r_tok, n)
            for ng, c in h_ng.items():
                matches[n - 1] += min(c, r_ng.get(ng, 0))
            totals[n - 1] += max(0, len(h_tok) - n + 1)
    for i in range(n_max):
        precisions[i] = matches[i] / totals[i] if totals[i] > 0 else 0.0
    if min(precisions) == 0:
        return 0.0
    bp = math.exp(1 - ref_len / hyp_len) if hyp_len < ref_len else 1.0
    score = bp * math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))
    return score


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_continuation(model, encode, decode, prompt: str, max_new: int, device: str) -> str:
    ids = torch.tensor([encode(prompt) or [0]], dtype=torch.long, device=device)
    block = model.config.block_size
    for _ in range(max_new):
        idx = ids if ids.size(1) <= block else ids[:, -block:]
        logits, _ = model(idx)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    full = decode(ids[0].tolist())
    return full[len(prompt):].split("\n", 1)[0].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default=str(DEFAULT_DATA))
    p.add_argument('--max-new', type=int, default=40)
    p.add_argument('--device', default=auto_device())
    args = p.parse_args()

    items = json.loads(Path(args.data).read_text())
    model, ck = load_checkpoint(args.checkpoint, args.device)
    encode, decode = resolve_codec(ck)

    refs, hyps = [], []
    for item in items:
        hyp = greedy_continuation(model, encode, decode,
                                  item['prompt'], args.max_new, args.device)
        refs.append(item['reference'])
        hyps.append(hyp)
        print(f"  {item['direction']:>5}  {hyp!r}   ⇄   {item['reference']!r}")
    bleu = corpus_bleu(refs, hyps)
    print(f"\nBLEU-4 (case-sensitive whitespace tokenisation): {bleu:.4f}")
    return bleu


if __name__ == '__main__':
    main()
