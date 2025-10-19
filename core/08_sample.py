"""DuckGPT — core lesson 08: text generation.

Loads the checkpoint that ``07_train.py`` saved and runs an autoregressive
sampling loop:

1. Start from a prompt (encode to token ids).
2. Forward through the model to get logits for the *next* position.
3. Optionally scale by a ``temperature`` (T < 1 = sharper, T > 1 = more random).
4. Optionally restrict to the ``top_k`` most likely tokens.
5. Sample one token from the resulting distribution.
6. Append the new token to the running sequence, then go to 2.

That's the whole "text generation" trick. Modern engines add: top-p (nucleus),
repetition penalty, beam search, KV cache, etc. — but the autoregressive idea
is exactly this loop.

Run with:

.. code:: bash

    python core/08_sample.py --start "ROMEO:" --max-new-tokens 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from importlib import import_module
GPT, GPTConfig = (lambda m: (m.GPT, m.GPTConfig))(import_module("05_model"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(HERE / "out" / "ckpt.pt"))
    parser.add_argument("--start", default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"loading {args.ckpt}…")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    stoi, itos = ck["vocab"]
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join(itos[i] for i in l)

    ids = torch.tensor([encode(args.start) or [0]], dtype=torch.long)
    print(f"prompt: {args.start!r}")
    print("─" * 60)
    print(args.start, end="", flush=True)

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            # crop to block_size (we never look at more context than the model
            # was trained on)
            idx = ids if ids.size(1) <= cfg.block_size else ids[:, -cfg.block_size:]
            logits, _ = model(idx)
            logits = logits[:, -1, :] / args.temperature       # (1, vocab)
            if args.top_k:
                top_vals, _ = torch.topk(logits, args.top_k)
                logits[logits < top_vals[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)
            ids = torch.cat([ids, next_id], dim=1)
            ch = itos[next_id.item()]
            print(ch, end="", flush=True)
    print("\n" + "─" * 60)


if __name__ == "__main__":
    main()
