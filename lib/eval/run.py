"""Run the whole DuckGPT eval suite and print a markdown table.

Usage:
    python lib/eval/run.py --checkpoint out/ckpt.pt \\
        [--val-bin data/wikipedia_ptbr/val.bin]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))

import numpy as np
import torch

from _loader import auto_device, load_checkpoint, resolve_codec, sequence_logprob
from hellaswag_mini import score_item as hs_score
from pt_grammar import score as grammar_score
from perplexity import compute_perplexity
from bleu import greedy_continuation, corpus_bleu
import json


def evaluate(args):
    model, ck = load_checkpoint(args.checkpoint, args.device)
    encode, decode = resolve_codec(ck)
    rows = []

    # ---- Perplexity ------------------------------------------------------
    if args.val_bin:
        bins = args.val_bin if isinstance(args.val_bin, list) else [args.val_bin]
        for bin_path in bins:
            ids = np.fromfile(bin_path, dtype=np.uint16)
            ppl = compute_perplexity(model, ids, args.device,
                                     model.config.block_size, batch_size=4)
            rows.append((f"PPL · {Path(bin_path).stem}", f"{ppl:.2f}",
                         f"{len(ids):,} tokens"))

    # ---- HellaSwag-mini --------------------------------------------------
    hs_data = json.loads((HERE / 'data' / 'hellaswag_mini.json').read_text())
    correct = sum(hs_score(model, encode, item, args.device) == item['answer']
                  for item in hs_data)
    rows.append(("HellaSwag-mini", f"{correct/len(hs_data):.1%}",
                 f"{correct}/{len(hs_data)}"))

    # ---- PT grammar ------------------------------------------------------
    gr_data = json.loads((HERE / 'data' / 'pt_grammar.json').read_text())
    correct = 0
    for item in gr_data:
        scores = [grammar_score(model, encode, c, args.device) for c in item['choices']]
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        correct += pred == item['answer']
    rows.append(("PT-Grammar", f"{correct/len(gr_data):.1%}",
                 f"{correct}/{len(gr_data)}"))

    # ---- BLEU ------------------------------------------------------------
    mt_data = json.loads((HERE / 'data' / 'mini_mt.json').read_text())
    refs, hyps = [], []
    for item in mt_data:
        hyp = greedy_continuation(model, encode, decode,
                                  item['prompt'], args.max_new, args.device)
        refs.append(item['reference']); hyps.append(hyp)
    bleu = corpus_bleu(refs, hyps)
    rows.append(("BLEU-4 (EN↔PT mini)", f"{bleu:.4f}", f"{len(mt_data)} pairs"))

    # ---- Print markdown --------------------------------------------------
    print("\n## DuckGPT eval — " + Path(args.checkpoint).name)
    print()
    print("| Metric | Value | Notes |")
    print("|---|---|---|")
    for name, val, note in rows:
        print(f"| {name} | {val} | {note} |")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--val-bin', nargs='+', default=None)
    p.add_argument('--max-new', type=int, default=40)
    p.add_argument('--device', default=auto_device())
    args = p.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
