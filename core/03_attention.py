"""DuckGPT — core lesson 03: scaled dot-product attention.

We build one *single-headed*, non-batched attention layer from first principles,
then multi-head, then add a causal mask so the model cannot peek at the future.

The formula (Vaswani et al., 2017):

.. math::

    \\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(
        \\frac{Q K^\\top}{\\sqrt{d_k}}
    \\right) V

What the pieces are:

- ``Q`` (queries): "what am I looking for?"
- ``K`` (keys):    "what do I advertise about myself?"
- ``V`` (values):  "what information do I carry?"
- Scaling by ``√d_k`` keeps dot-products small so the softmax does not saturate.

Run this file to print shapes at every step + a heatmap of the attention
weights (saved to ``core/attn.png`` if matplotlib is installed).
"""
import math

import numpy as np

rng = np.random.default_rng(0)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ----- 1. single-headed attention ------------------------------------------
print("=== 1. Single head attention ===")
T, d = 5, 8                                  # 5 tokens, embedding dim 8
X = rng.standard_normal((T, d))

# learned projections (here, random — in a real model these are trained)
Wq = rng.standard_normal((d, d))
Wk = rng.standard_normal((d, d))
Wv = rng.standard_normal((d, d))

Q = X @ Wq                                   # (T, d)
K = X @ Wk
V = X @ Wv

scores = Q @ K.T / math.sqrt(d)              # (T, T)  – similarity
attn = softmax(scores, axis=-1)              # (T, T)  – row stochastic
out  = attn @ V                              # (T, d)

print(f"X.shape:      {X.shape}")
print(f"Q/K/V.shape:  {Q.shape}")
print(f"scores.shape: {scores.shape}")
print(f"attn.shape:   {attn.shape}  (rows sum to 1)")
print(f"out.shape:    {out.shape}\n")

# ----- 2. causal mask ------------------------------------------------------
print("=== 2. Causal mask (decoder-style) ===")
mask = np.tril(np.ones((T, T), dtype=bool))  # lower-triangular True
masked_scores = np.where(mask, scores, -1e9)
masked_attn = softmax(masked_scores, axis=-1)
print(f"row 0 attn weights: {masked_attn[0].round(3)}   (only token 0 visible)")
print(f"row 4 attn weights: {masked_attn[4].round(3)}   (all 5 tokens visible)\n")


# ----- 3. multi-head attention --------------------------------------------
print("=== 3. Multi-head attention ===")
H = 4                                          # 4 heads
d_head = d // H

def split_heads(x):
    return x.reshape(T, H, d_head).transpose(1, 0, 2)     # (H, T, d_head)

def combine_heads(x):
    return x.transpose(1, 0, 2).reshape(T, d)             # (T, d)

q = split_heads(Q)                            # (H, T, d_head)
k = split_heads(K)
v = split_heads(V)

scores_mh = np.einsum("htd,hsd->hts", q, k) / math.sqrt(d_head)   # (H, T, T)
attn_mh = softmax(scores_mh, axis=-1)
out_mh = combine_heads(np.einsum("hts,hsd->htd", attn_mh, v))
print(f"per-head attn shape: {attn_mh.shape}")
print(f"multi-head output:    {out_mh.shape}\n")


# ----- 4. optional: visualise the attention --------------------------------
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(masked_attn, cmap="viridis"); ax[0].set_title("causal attn (single head)")
    ax[0].set_xlabel("key"); ax[0].set_ylabel("query")
    ax[1].imshow(attn_mh[0], cmap="viridis"); ax[1].set_title("head 0 (unmasked)")
    for a in ax: a.set_xticks(range(T)); a.set_yticks(range(T))
    plt.tight_layout()
    plt.savefig("core/attn.png", dpi=120)
    print("Saved attention heatmap to core/attn.png")
except ModuleNotFoundError:
    print("(matplotlib not installed — skipped heatmap)")

print("\nNext file: 04_transformer_block.py — wire attention + MLP into a block.")
