"""DuckGPT — core lesson 07: a complete training loop on Shakespeare-char.

This is the minimum honest implementation of "train a GPT on text":

1. Load the prepared ``data/shakespeare_char/{train,val}.bin`` arrays.
2. Sample random fixed-length windows on every step.
3. Forward + backward + Adam step (using ``core/06_optim.py``).
4. Periodically evaluate on the validation split.
5. Save the best checkpoint to ``core/out/ckpt.pt``.

It uses ``core/05_model.py`` for the model and ``core/06_optim.py`` for the
optimiser. No DDP, no AMP, no compile, no resume — start to finish in ~150
lines, runs on CPU in ~2-3 minutes for the smoke configuration.

When you want a real training run, see ``lib/train.py``.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(HERE))   # so we can import 05_model / 06_optim by name

from importlib import import_module
GPT, GPTConfig = (lambda m: (m.GPT, m.GPTConfig))(import_module("05_model"))
Adam, cosine_warmup_lr = (lambda m: (m.Adam, m.cosine_warmup_lr))(import_module("06_optim"))


# ----- 1. config -----------------------------------------------------------
DATA_DIR = ROOT / "data" / "shakespeare_char"
OUT_DIR = HERE / "out"
OUT_DIR.mkdir(exist_ok=True)

# hyperparameters (kept tiny so this runs on CPU in a couple of minutes)
batch_size = 32
block_size = 64
n_layer = 3
n_head = 4
n_embd = 96
max_iters = 300
eval_interval = 100
eval_iters = 20
lr_max = 3e-3
lr_min = 1e-4
warmup_iters = 30


# ----- 2. load data --------------------------------------------------------
with open(DATA_DIR / "meta.pkl", "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]
stoi, itos = meta["stoi"], meta["itos"]
print(f"vocab_size: {vocab_size}")

train_data = np.fromfile(DATA_DIR / "train.bin", dtype=np.uint16)
val_data   = np.fromfile(DATA_DIR / "val.bin",   dtype=np.uint16)
print(f"train tokens: {len(train_data):,}   val tokens: {len(val_data):,}")


def get_batch(split: str):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i: i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1: i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


# ----- 3. model + optimiser -----------------------------------------------
torch.manual_seed(0)
cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                n_layer=n_layer, n_head=n_head, n_embd=n_embd)
model = GPT(cfg)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

opt = Adam(model.parameters(), lr=lr_max, weight_decay=0.1)


# ----- 4. training loop ---------------------------------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


print("\nstarting training\n")
best_val = float("inf")
t0 = time.time()
for it in range(max_iters):
    opt.lr = cosine_warmup_lr(it, warmup_iters, max_iters, lr_max, lr_min)
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad(); loss.backward(); opt.step()

    if it % eval_interval == 0 or it == max_iters - 1:
        ls = estimate_loss()
        print(f"iter {it:>4}  lr {opt.lr:.4f}  "
              f"train {ls['train']:.3f}  val {ls['val']:.3f}  "
              f"elapsed {time.time() - t0:.1f}s")
        if ls["val"] < best_val:
            best_val = ls["val"]
            torch.save({"model": model.state_dict(),
                        "config": cfg,
                        "vocab": (stoi, itos)},
                       OUT_DIR / "ckpt.pt")
            print(f"  ↳ saved best checkpoint  (val={best_val:.3f})")

print(f"\ndone — best val loss {best_val:.3f}")
print(f"\nNext file: 08_sample.py — generate text from the saved checkpoint.")
