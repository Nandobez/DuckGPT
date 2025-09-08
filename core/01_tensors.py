"""DuckGPT — core lesson 01: tensors, matmul, softmax.

Goal
====
Before we can build a transformer we need to be confident with three ideas:

1. **Tensors** are just multi-dimensional arrays. PyTorch tensors and NumPy
   arrays share almost the same API; the difference is that PyTorch tensors
   know how to compute gradients ("autograd").
2. **Matrix multiplication** is the workhorse of every layer. For
   `Y = X @ W`, every output value is a dot-product of one row of `X` with
   one column of `W`.
3. **Softmax** turns a vector of scores into a probability distribution,
   used both to weight attention and to pick the next token. Formula:

   .. math:: \\mathrm{softmax}(x)_i = \\frac{e^{x_i}}{\\sum_j e^{x_j}}

Run this file with ``python core/01_tensors.py`` — every step prints what it
just computed, plus the equivalent PyTorch result, so you can convince yourself
both versions agree.
"""
import math

import numpy as np
import torch

# ----- 1. tensors -----------------------------------------------------------
print("=== 1. Tensors ===")
x = np.arange(6.0).reshape(2, 3)            # shape (2, 3)
y = torch.arange(6.0).reshape(2, 3)
print(f"numpy x:\n{x}")
print(f"torch y:\n{y}")
print(f"same numbers ? {np.allclose(x, y.numpy())}\n")

# ----- 2. matrix multiplication --------------------------------------------
print("=== 2. Matrix multiplication ===")
A = np.random.randn(4, 5)                   # (4, 5)
B = np.random.randn(5, 3)                   # (5, 3)
C = A @ B                                   # (4, 3)
print(f"A shape: {A.shape}, B shape: {B.shape}, C = A @ B shape: {C.shape}")

# manual loop equivalent of `A @ B`
manual = np.zeros((4, 3))
for i in range(4):
    for j in range(3):
        for k in range(5):
            manual[i, j] += A[i, k] * B[k, j]
print(f"matches manual loop ? {np.allclose(C, manual)}\n")

# ----- 3. softmax -----------------------------------------------------------
print("=== 3. Softmax ===")

def softmax(x):
    """Numerically stable softmax over the last dimension."""
    x = x - x.max(axis=-1, keepdims=True)   # subtract max for stability
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"logits: {logits}")
print(f"softmax: {probs}    sum: {probs.sum():.6f}")

ref = torch.softmax(torch.tensor(logits), dim=-1).numpy()
print(f"matches torch.softmax ? {np.allclose(probs, ref)}\n")

# ----- 4. why subtract the max? --------------------------------------------
print("=== 4. Stability of softmax ===")
big_logits = np.array([1000.0, 999.0, 998.0])
# naive: exp(1000) overflows to inf
naive = np.exp(big_logits) / np.exp(big_logits).sum()
print(f"naive softmax of {big_logits}: {naive}")
print(f"stable softmax of {big_logits}: {softmax(big_logits)}")
print("…always subtract the max before exp()\n")

# ----- 5. tiny matmul exercise ----------------------------------------------
print("=== 5. Mini exercise: project then attend ===")
n, d = 4, 8           # 4 tokens, embedding dim 8
X = np.random.randn(n, d)
Wq = np.random.randn(d, d)
Wk = np.random.randn(d, d)

Q = X @ Wq            # queries  (n, d)
K = X @ Wk            # keys     (n, d)
scores = Q @ K.T / math.sqrt(d)            # (n, n) scaled dot-product
attn = softmax(scores)                     # row-stochastic matrix
print(f"scores shape: {scores.shape}")
print(f"attn weights row sums (should be 1.0): {attn.sum(axis=1)}")
print("\nNext file: 02_tokenizer.py — turning text into integer ids.")
