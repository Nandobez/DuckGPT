"""
DuckGPT — knowledge distillation.

Trains a small *student* GPT to mimic the next-token distribution of a larger
*teacher* GPT. Loss is a convex combination of:

- KL divergence between the temperature-softened teacher and student logits
  (Hinton, Vinyals, Dean 2015):

  .. math::

      \\mathcal{L}_{\\text{KD}} = T^2 \\, \\mathrm{KL}\\!\\left(
          \\mathrm{softmax}(z_T / T) \\,\\|\\, \\mathrm{softmax}(z_S / T)
      \\right)

- the usual cross-entropy against the real next-token labels.

Final loss is :math:`\\alpha \\mathcal{L}_{\\text{KD}} + (1 - \\alpha) \\mathcal{L}_{\\text{CE}}`.

The result is a model 4-8× smaller that keeps most of the teacher's quality.

Usage:
    python lib/distill.py \\
        --teacher-ckpt out-teacher/ckpt.pt \\
        --dataset data/wikipedia_ptbr \\
        --student-config '{"n_layer":4,"n_head":4,"n_embd":256}' \\
        --out out-student/ckpt.pt \\
        --iters 5000 --alpha 0.8 --temperature 2.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from model import GPT, GPTConfig
from optim import AdamW, cosine_warmup_lr


def _auto_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _load_teacher(ckpt: str, device: str):
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = obj.get('model_args') or obj.get('config')
    if isinstance(cfg, dict):
        cfg = GPTConfig(**{k: v for k, v in cfg.items()
                           if k in GPTConfig.__dataclass_fields__})
    teacher = GPT(cfg)
    sd = obj['model']
    for k in list(sd.keys()):
        if k.startswith('_orig_mod.'):
            sd[k[len('_orig_mod.'):]] = sd.pop(k)
    teacher.load_state_dict(sd)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval().to(device)
    return teacher, cfg


def _build_student(teacher_cfg: GPTConfig, overrides: dict) -> GPT:
    fields = {**teacher_cfg.__dict__, **overrides}
    cfg = GPTConfig(**{k: v for k, v in fields.items()
                       if k in GPTConfig.__dataclass_fields__})
    return GPT(cfg), cfg


def distillation_loss(student_logits: torch.Tensor,
                      teacher_logits: torch.Tensor,
                      targets: torch.Tensor,
                      alpha: float,
                      temperature: float) -> tuple[torch.Tensor, dict]:
    T = temperature
    teacher_soft = F.log_softmax(teacher_logits / T, dim=-1).detach()
    student_soft = F.log_softmax(student_logits / T, dim=-1)
    # KL(P_teacher || P_student) — `kl_div` expects log-probs in input,
    # probs in target, so we use `log_target=True` to keep both in log-space.
    kd = F.kl_div(student_soft, teacher_soft, reduction='batchmean',
                  log_target=True) * (T * T)
    ce = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)),
                         targets.reshape(-1))
    return alpha * kd + (1 - alpha) * ce, {"kd": float(kd.item()),
                                            "ce": float(ce.item())}


def main():
    p = argparse.ArgumentParser(description="DuckGPT distillation")
    p.add_argument('--teacher-ckpt', required=True)
    p.add_argument('--dataset', required=True, help='dir with train.bin + val.bin')
    p.add_argument('--out', required=True)
    p.add_argument('--student-config', default='{"n_layer":4,"n_head":4,"n_embd":256}',
                   help='JSON dict of overrides on top of teacher config.')
    p.add_argument('--iters', type=int, default=5000)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--alpha', type=float, default=0.8, help='KD weight; 1.0 = pure KD')
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--eval-interval', type=int, default=200)
    p.add_argument('--device', default=_auto_device())
    args = p.parse_args()

    print(f"loading teacher from {args.teacher_ckpt}…")
    teacher, t_cfg = _load_teacher(args.teacher_ckpt, args.device)
    print(f"teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    student_overrides = json.loads(args.student_config)
    student, s_cfg = _build_student(t_cfg, student_overrides)
    student.to(args.device)
    print(f"student params: {sum(p.numel() for p in student.parameters()):,}  "
          f"(ratio: {sum(p.numel() for p in teacher.parameters()) / max(1, sum(p.numel() for p in student.parameters())):.1f}x)")

    block = student.config.block_size
    train = np.memmap(Path(args.dataset) / 'train.bin', dtype=np.uint16, mode='r')
    val   = np.memmap(Path(args.dataset) / 'val.bin',   dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train if split == 'train' else val
        ix = np.random.randint(0, len(data) - block - 1, size=args.batch_size)
        x = torch.from_numpy(np.stack([data[i:i+block] for i in ix])).long().to(args.device)
        y = torch.from_numpy(np.stack([data[i+1:i+1+block] for i in ix])).long().to(args.device)
        return x, y

    opt = AdamW(student.parameters(), lr=args.lr, weight_decay=1e-2)
    t0 = time.time()
    for it in range(args.iters):
        opt.lr = cosine_warmup_lr(it, args.warmup, args.iters, args.lr, args.lr / 10)
        x, y = get_batch('train')
        # teacher forward (no grad)
        with torch.no_grad():
            t_logits, _ = teacher(x)
        # student forward (with grad)
        s_logits, _ = student(x, y)   # `forward` returns (logits, ce_loss)
        loss, parts = distillation_loss(s_logits, t_logits, y,
                                        alpha=args.alpha,
                                        temperature=args.temperature)
        opt.zero_grad(); loss.backward(); opt.step()

        if it % args.eval_interval == 0 or it == args.iters - 1:
            with torch.no_grad():
                xv, yv = get_batch('val')
                t_v, _ = teacher(xv)
                s_v, vl = student(xv, yv)
                v_loss, _ = distillation_loss(s_v, t_v, yv,
                                              alpha=args.alpha,
                                              temperature=args.temperature)
            print(f"iter {it:>5}  lr {opt.lr:.4e}  "
                  f"train {loss.item():.3f} (kd {parts['kd']:.3f}, ce {parts['ce']:.3f})  "
                  f"val {v_loss.item():.3f}  elapsed {time.time() - t0:.0f}s")

    os.makedirs(Path(args.out).parent, exist_ok=True)
    torch.save({
        "model": student.state_dict(),
        "model_args": {f: getattr(s_cfg, f) for f in GPTConfig.__dataclass_fields__},
        "config": {"distilled_from": args.teacher_ckpt,
                   "alpha": args.alpha,
                   "temperature": args.temperature,
                   "dataset": args.dataset},
    }, args.out)
    print(f"\nsaved student checkpoint -> {args.out}")


if __name__ == '__main__':
    main()
