"""DuckGPT — core lesson 06: Adam + cosine warm-up LR, written from scratch.

Adam (Kingma & Ba, 2014) keeps running estimates of the first and second
gradient moments and uses them to take an *adaptive* step per parameter:

.. math::

    m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\\\
    v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\\\
    \\hat m_t &= m_t / (1 - \\beta_1^t)        \\quad \\text{(bias-correct)} \\\\
    \\hat v_t &= v_t / (1 - \\beta_2^t)        \\\\
    \\theta_t &= \\theta_{t-1}
        - \\eta \\hat m_t / (\\sqrt{\\hat v_t} + \\varepsilon)

The AdamW variant (Loshchilov & Hutter, 2017) decouples weight decay from the
gradient update by simply shrinking ``θ`` *before* the adaptive step.

We also implement a **linear warm-up + cosine decay** learning-rate schedule,
which is the default for almost every modern LLM:

- For the first ``warmup_steps`` iterations, the LR ramps linearly from 0 to
  ``lr_max``.
- After that, it decays as a half cosine until ``lr_min`` at ``total_steps``.

Run this file to train a 2-parameter linear regression and verify that the
loss reaches near-zero.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch


class Adam:
    """A minimal AdamW. Built for readability, not speed."""

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        # per-parameter running averages of g and g²
        self._m = [torch.zeros_like(p) for p in self.params]
        self._v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        self.t += 1
        bias1 = 1 - self.b1 ** self.t
        bias2 = 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self._m, self._v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            if self.wd:
                p.mul_(1 - self.lr * self.wd)
            p.addcdiv_(m / bias1, (v / bias2).sqrt().add_(self.eps),
                       value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


def cosine_warmup_lr(step: int,
                     warmup_steps: int,
                     total_steps: int,
                     lr_max: float,
                     lr_min: float = 0.0) -> float:
    """Linear warm-up then cosine decay. See module docstring."""
    if warmup_steps > 0 and step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    if step >= total_steps:
        return lr_min
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Demo: fit y = 2 x1 - 3 x2 with our home-grown optimiser
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(256, 2)
    y_true = x @ torch.tensor([2.0, -3.0])
    layer = torch.nn.Linear(2, 1, bias=False)
    opt = Adam(layer.parameters(), lr=5e-2)
    total = 200
    warmup = 20
    for step in range(total):
        opt.lr = cosine_warmup_lr(step, warmup, total, 5e-2)
        pred = layer(x).squeeze(-1)
        loss = (pred - y_true).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            print(f"step {step:>3}  lr {opt.lr:.4f}  loss {loss.item():.6f}")
    print("learned weights:", layer.weight.detach().squeeze().tolist())
    print("(expected ≈ [2.0, -3.0])")
    print("\nNext file: 07_train.py — putting it all together on Shakespeare.")
