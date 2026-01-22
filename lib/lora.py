"""
DuckGPT — LoRA fine-tuning.

LoRA (Hu et al., 2021) freezes the pretrained weights ``W`` and learns a
low-rank update :math:`\\Delta W = B A` such that the effective transform
becomes

.. math::

    h = W x + \\frac{\\alpha}{r} \\, B (A x)

with :math:`A \\in \\mathbb{R}^{r \\times d}`, :math:`B \\in \\mathbb{R}^{d \\times r}`
and ``r ≪ d``.

What this file gives you
========================

- :class:`LoRALinear` — a drop-in wrapper around an ``nn.Linear`` that adds an
  ``A`` and ``B`` adapter. ``B`` is initialised to zero so the adapter starts
  as a no-op (the wrapped model behaves identically until the adapter learns
  something).
- :func:`inject_lora(model, target_modules, r, alpha)` — walks the GPT module
  tree and swaps every linear layer whose dotted name matches a target into a
  ``LoRALinear``. By default we patch ``c_attn`` (the fused QKV projection) and
  ``c_proj`` (output projection) — these are the layers original LoRA paper
  recommends.
- :func:`save_lora`, :func:`load_lora` — write / read only the adapter
  parameters as a small ``.pt`` file (typically a few MB instead of GB).
- :func:`merge_lora` — collapse the adapter back into the base weights so
  inference has zero overhead.

CLI
===

.. code:: bash

    # Fine-tune Shakespeare-trained model on PT-BR literature
    python lib/lora.py finetune \\
        --base-ckpt out-shakespeare/ckpt.pt \\
        --dataset data/literatura_ptbr \\
        --out out-lora-ptbr.pt --r 8 --alpha 16 \\
        --iters 1000 --lr 1e-3

    # Inspect what was trained
    python lib/lora.py info --adapter out-lora-ptbr.pt

    # Merge the adapter back into the base for zero-overhead inference
    python lib/lora.py merge \\
        --base-ckpt out-shakespeare/ckpt.pt \\
        --adapter out-lora-ptbr.pt --out out-merged.pt
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from model import GPT, GPTConfig
from optim import AdamW, cosine_warmup_lr


# ---------------------------------------------------------------------------
# LoRA module
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` and add a low-rank trainable update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        in_f = base.in_features
        out_f = base.out_features
        # A is small * input;  B is output * small;   ΔW = B @ A
        self.A = nn.Parameter(torch.empty(r, in_f))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        out = out + self.scale * F.linear(self.drop(F.linear(x, self.A)), self.B)
        return out

    @torch.no_grad()
    def merge_into_base(self) -> None:
        """Fold (B @ A) * scale into self.base.weight; LoRA params left at zero."""
        delta = self.scale * (self.B @ self.A)             # (out, in)
        self.base.weight.add_(delta)
        self.B.zero_(); self.A.zero_()                     # so re-running forward is a no-op


# ---------------------------------------------------------------------------
# Injection / save / load
# ---------------------------------------------------------------------------

DEFAULT_TARGETS = ("c_attn", "c_proj")


def _iter_named_children(root: nn.Module, prefix: str = ""):
    for name, child in root.named_children():
        full = f"{prefix}.{name}" if prefix else name
        yield root, name, full, child
        yield from _iter_named_children(child, full)


def inject_lora(model: nn.Module,
                target_modules: Iterable[str] = DEFAULT_TARGETS,
                r: int = 8,
                alpha: float = 16.0,
                dropout: float = 0.0) -> dict[str, LoRALinear]:
    """Replace matching ``nn.Linear`` layers with ``LoRALinear``.

    A module matches if its short name (last component of the dotted path) is
    in ``target_modules``. Returns a dict ``{full_name: LoRALinear}``.
    """
    targets = set(target_modules)
    inserted: dict[str, LoRALinear] = {}
    for parent, attr, full_name, child in _iter_named_children(model):
        if attr in targets and isinstance(child, nn.Linear):
            lora = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, lora)
            inserted[full_name] = lora
    # Freeze all non-LoRA params
    for n, p in model.named_parameters():
        if not (n.endswith(".A") or n.endswith(".B")):
            p.requires_grad = False
    return inserted


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only LoRA adapter tensors (A + B)."""
    return {n: p.detach().clone()
            for n, p in model.named_parameters()
            if n.endswith(".A") or n.endswith(".B")}


def save_lora(model: nn.Module, path: str, meta: dict | None = None) -> None:
    payload = {"adapter": lora_state_dict(model), "meta": meta or {}}
    torch.save(payload, path)


def load_lora(model: nn.Module, path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sd = payload["adapter"]
    missing = []
    for n, p in model.named_parameters():
        if n in sd:
            with torch.no_grad():
                p.copy_(sd[n])
        elif n.endswith(".A") or n.endswith(".B"):
            missing.append(n)
    if missing:
        print(f"WARN: {len(missing)} adapter params not in file (kept random)")
    return payload.get("meta", {})


def merge_lora(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge_into_base()


# ---------------------------------------------------------------------------
# Fine-tune loop
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _load_base(base_ckpt: str, device: str):
    obj = torch.load(base_ckpt, map_location=device, weights_only=False)
    cfg = obj.get('model_args') or obj.get('config')
    if isinstance(cfg, dict):
        cfg = GPTConfig(**{k: v for k, v in cfg.items()
                           if k in GPTConfig.__dataclass_fields__})
    model = GPT(cfg)
    state = obj['model']
    for k in list(state.keys()):
        if k.startswith('_orig_mod.'):
            state[k[len('_orig_mod.'):]] = state.pop(k)
    model.load_state_dict(state)
    return model.to(device), cfg


def _data_loader(dataset_dir: Path, block_size: int, batch_size: int, device: str):
    train = np.memmap(dataset_dir / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(dataset_dir / "val.bin", dtype=np.uint16, mode="r")
    def get_batch(split):
        data = train if split == "train" else val
        ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
        x = torch.from_numpy(np.stack([data[i: i + block_size] for i in ix])).long().to(device)
        y = torch.from_numpy(np.stack([data[i + 1: i + 1 + block_size] for i in ix])).long().to(device)
        return x, y
    return get_batch


def finetune(args):
    device = args.device
    print(f"loading base from {args.base_ckpt}…")
    model, cfg = _load_base(args.base_ckpt, device)
    print(f"params (base): {sum(p.numel() for p in model.parameters()):,}")

    inserted = inject_lora(model, target_modules=args.targets.split(","),
                           r=args.r, alpha=args.alpha, dropout=args.dropout)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA adapters: {len(inserted)} layers, {trainable:,} trainable params")

    get_batch = _data_loader(Path(args.dataset), cfg.block_size, args.batch_size, device)
    opt = AdamW((p for p in model.parameters() if p.requires_grad),
                lr=args.lr, weight_decay=args.weight_decay)
    t0 = time.time()
    for it in range(args.iters):
        opt.lr = cosine_warmup_lr(it, args.warmup, args.iters, args.lr, args.lr / 10)
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % args.eval_interval == 0 or it == args.iters - 1:
            with torch.no_grad():
                xv, yv = get_batch("val")
                _, vloss = model(xv, yv)
            print(f"iter {it:>5}  lr {opt.lr:.4e}  "
                  f"train {loss.item():.3f}  val {vloss.item():.3f}  "
                  f"elapsed {time.time() - t0:.0f}s")

    meta = {"base_ckpt": args.base_ckpt, "r": args.r, "alpha": args.alpha,
            "targets": args.targets.split(","), "dataset": args.dataset,
            "iters": args.iters}
    save_lora(model, args.out, meta=meta)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"\nsaved adapter to {args.out}  ({size_mb:.2f} MB)")


def cmd_info(args):
    payload = torch.load(args.adapter, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    sd = payload["adapter"]
    total = sum(t.numel() for t in sd.values())
    print(f"adapter file : {args.adapter}")
    print(f"adapter size : {Path(args.adapter).stat().st_size / 1e6:.2f} MB")
    print(f"trainable    : {total:,} params across {len(sd)} tensors")
    if meta:
        print("metadata     :")
        for k, v in meta.items():
            print(f"  {k}: {v}")


def cmd_merge(args):
    model, cfg = _load_base(args.base_ckpt, args.device)
    payload = torch.load(args.adapter, map_location=args.device, weights_only=False)
    meta = payload.get("meta", {})
    inject_lora(model, target_modules=meta.get("targets", DEFAULT_TARGETS),
                r=meta.get("r", 8), alpha=meta.get("alpha", 16.0))
    load_lora(model, args.adapter)
    merge_lora(model)
    # save as a normal GPT checkpoint
    torch.save({
        "model": {k: v for k, v in model.state_dict().items()
                  if not k.endswith(".A") and not k.endswith(".B")},
        "model_args": cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg),
        "config": {"merged_from": [args.base_ckpt, args.adapter]},
    }, args.out)
    print(f"merged checkpoint -> {args.out}")


def main():
    p = argparse.ArgumentParser(description="DuckGPT LoRA")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("finetune", help="train an adapter on top of a base checkpoint")
    pf.add_argument("--base-ckpt", required=True)
    pf.add_argument("--dataset", required=True, help="dir containing train.bin + val.bin")
    pf.add_argument("--out", required=True)
    pf.add_argument("--r", type=int, default=8)
    pf.add_argument("--alpha", type=float, default=16.0)
    pf.add_argument("--dropout", type=float, default=0.0)
    pf.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    pf.add_argument("--iters", type=int, default=500)
    pf.add_argument("--warmup", type=int, default=50)
    pf.add_argument("--lr", type=float, default=1e-3)
    pf.add_argument("--weight-decay", type=float, default=1e-2)
    pf.add_argument("--batch-size", type=int, default=16)
    pf.add_argument("--eval-interval", type=int, default=50)
    pf.add_argument("--device", default=_auto_device())
    pf.set_defaults(func=finetune)

    pi = sub.add_parser("info", help="inspect a saved adapter")
    pi.add_argument("--adapter", required=True)
    pi.set_defaults(func=cmd_info)

    pm = sub.add_parser("merge", help="fold adapter weights into the base model")
    pm.add_argument("--base-ckpt", required=True)
    pm.add_argument("--adapter", required=True)
    pm.add_argument("--out", required=True)
    pm.add_argument("--device", default=_auto_device())
    pm.set_defaults(func=cmd_merge)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
