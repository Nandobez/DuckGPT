"""DuckGPT — core lesson 05: a minimal GPT.

Now that we have the pieces (tokeniser, attention, block) we wire them into the
smallest reasonable GPT. ~150 lines, written for readability:

- one ``Block`` class containing pre-norm attention + MLP
- a ``GPT`` module that owns:
    - a token embedding ``wte``
    - a position embedding ``wpe``
    - ``n_layer`` blocks
    - a final LayerNorm
    - a tied LM head (re-uses ``wte.weight``)
- a ``forward(x, targets=None)`` that returns ``(logits, loss)``

We use ``F.scaled_dot_product_attention`` here too — it does the same maths as
``core/03_attention.py`` and is fast on GPUs, but you can swap it back for the
manual implementation if you want to inspect the attention weights themselves.

Run this file to instantiate the model and forward a random batch.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config (a tiny dataclass keeps hyperparameters in one place)
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 256       # tiny — for demo
    block_size: int = 64        # max sequence length
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """One multi-head, causal self-attention layer.

    See ``core/03_attention.py`` for the manual numpy version. Here we let
    PyTorch's ``scaled_dot_product_attention`` do the heavy lifting.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.n_embd // cfg.n_head
        # one fused (Q, K, V) linear, then a final output linear
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, T, n_head, d_head) -> (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # back to (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    """Two-layer feed-forward with GELU. Hidden dim is 4 × n_embd, as in GPT-2."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.act = nn.GELU()
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))


class Block(nn.Module):
    """Pre-norm transformer block: x → x + attn(LN(x)) → y + mlp(LN(y))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class GPT(nn.Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)   # positional embeddings
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying: input embedding and output projection share weights.
        self.wte.weight = self.lm_head.weight
        # GPT-2 paper init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence too long"
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Run as a script for a sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=128, block_size=32, n_layer=2, n_head=4, n_embd=64)
    model = GPT(cfg)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))   # fake batch
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(idx, targets)
    print(f"logits shape: {logits.shape}")
    print(f"loss        : {loss.item():.4f}")
    print("\nNext file: 06_optim.py — Adam + cosine LR schedule from scratch.")
