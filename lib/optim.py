"""
DuckGPT — optimisation primitives from scratch.

Includes:
  - Adam (with decoupled weight decay, i.e. AdamW)
  - warmup + cosine learning-rate schedule

No reliance on `torch.optim.*`; only on torch tensor ops.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch


class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2017).

    Re-implements `torch.optim.AdamW` in ~30 lines for transparency.
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 lr: float = 3e-4,
                 betas: tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8,
                 weight_decay: float = 0.1):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # Each entry is a tuple (param, exp_avg, exp_avg_sq).
        # We materialise the state lazily to mirror torch.optim behaviour.
        self._params = list(params)
        self._state: dict[int, dict[str, torch.Tensor]] = {}
        self.step_count = 0

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        b1, b2 = self.betas
        bias1 = 1.0 - b1 ** self.step_count
        bias2 = 1.0 - b2 ** self.step_count

        for p in self._params:
            if p.grad is None:
                continue
            g = p.grad
            state = self._state.setdefault(id(p), {})
            if "m" not in state:
                state["m"] = torch.zeros_like(p)
                state["v"] = torch.zeros_like(p)
            m, v = state["m"], state["v"]

            # in-place updates of the running first and second moments
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)

            m_hat = m / bias1
            v_hat = v / bias2

            # decoupled weight decay: shrink params before the adaptive step
            if self.weight_decay != 0:
                p.mul_(1 - self.lr * self.weight_decay)

            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self._params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.detach_()
                p.grad.zero_()

    # ----- LR control (so a scheduler can just set `optimizer.lr = ...`) -----
    @property
    def learning_rate(self) -> float:
        return self.lr


def cosine_warmup_lr(step: int,
                     warmup_steps: int,
                     total_steps: int,
                     lr_max: float,
                     lr_min: float = 0.0) -> float:
    """Linear warm-up to `lr_max`, then cosine decay down to `lr_min`."""
    if warmup_steps > 0 and step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    if step >= total_steps:
        return lr_min
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def constant_lr(step: int, lr: float) -> float:
    return lr
