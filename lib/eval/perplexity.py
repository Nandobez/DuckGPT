"""Perplexity on a held-out token stream.

PPL = exp(− 1/N · Σ log P(x_t | x_{<t})).

Usage:
    python lib/eval/perplexity.py --checkpoint out/ckpt.pt \\
        --bin data/wikipedia_ptbr/val.bin
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _loader import auto_device, load_checkpoint


def compute_perplexity(model, ids: np.ndarray, device: str,
                        block_size: int, batch_size: int = 8) -> float:
    """Stride a window of length ``block_size`` across the corpus, accumulate
    the log-likelihood of every token, and return exp(mean NLL).
    """
    total_nll = 0.0
    total_tokens = 0
    block = min(block_size, model.config.block_size)
    # naive non-overlapping chunks; for tighter estimates use a stride of 1
    n = (len(ids) // block) * block
    chunks = ids[:n].reshape(-1, block)
    with torch.no_grad():
        for start in range(0, len(chunks), batch_size):
            batch = torch.from_numpy(chunks[start:start + batch_size]).long().to(device)
            logits, _ = model(batch)
            target = batch[:, 1:]
            preds = logits[:, :-1].reshape(-1, logits.size(-1))
            nll = F.cross_entropy(preds, target.reshape(-1), reduction='sum')
            total_nll += float(nll.item())
            total_tokens += int(target.numel())
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--bin', required=True, help='.bin file (uint16 tokens)')
    p.add_argument('--block-size', type=int, default=0, help='0 = use model.config')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default=auto_device())
    args = p.parse_args()

    model, _ = load_checkpoint(args.checkpoint, args.device)
    ids = np.fromfile(args.bin, dtype=np.uint16)
    block = args.block_size or model.config.block_size
    ppl = compute_perplexity(model, ids, args.device, block, args.batch_size)
    print(f"{Path(args.bin).name}: {len(ids):,} tokens  PPL = {ppl:.2f}")
    return ppl


if __name__ == '__main__':
    main()
