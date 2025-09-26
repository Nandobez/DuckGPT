"""DuckGPT — core lesson 04: one transformer block.

A transformer decoder block (the kind a GPT stacks N times) has four pieces:

1. **LayerNorm** — normalise each token vector to zero mean / unit variance.
2. **Causal self-attention** — what we just built in ``03_attention.py``,
   wrapped in a residual connection.
3. **LayerNorm** again.
4. **MLP** — two linear layers with GELU in between, wrapped in another
   residual connection.

Pseudo-code:

.. code:: python

    z = x + attn(layer_norm(x))
    y = z + mlp(layer_norm(z))

We use *pre-LayerNorm* (norm before sub-block) — this is what GPT-2 / modern
LLMs use, and it makes training more stable than the original post-norm.

Run the file: it forwards a small random batch through a block, printing the
shape after every step. You should see the shape unchanged: ``(B, T, d_model)``
goes in and comes out, while context flows through attention.
"""
import math

import numpy as np

rng = np.random.default_rng(0)
B, T, d, H = 2, 6, 32, 4
d_head = d // H


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def gelu(x):
    """Approximate GELU activation (Hendrycks & Gimpel 2016)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def causal_attention(x, Wq, Wk, Wv, Wo):
    Bsz, Tlen, D = x.shape
    Q = (x @ Wq).reshape(Bsz, Tlen, H, d_head).transpose(0, 2, 1, 3)  # (B, H, T, d_head)
    K = (x @ Wk).reshape(Bsz, Tlen, H, d_head).transpose(0, 2, 1, 3)
    V = (x @ Wv).reshape(Bsz, Tlen, H, d_head).transpose(0, 2, 1, 3)
    scores = np.einsum("bhtd,bhsd->bhts", Q, K) / math.sqrt(d_head)
    mask = np.tril(np.ones((Tlen, Tlen), dtype=bool))
    scores = np.where(mask, scores, -1e9)
    attn = softmax(scores)
    ctx = np.einsum("bhts,bhsd->bhtd", attn, V)               # (B, H, T, d_head)
    ctx = ctx.transpose(0, 2, 1, 3).reshape(Bsz, Tlen, D)     # (B, T, D)
    return ctx @ Wo


def mlp(x, W1, W2):
    return gelu(x @ W1) @ W2


def block(x, params):
    # pre-norm self-attention + residual
    h = x + causal_attention(layer_norm(x), *params["attn"])
    # pre-norm MLP + residual
    return h + mlp(layer_norm(h), *params["mlp"])


# ----- initialise parameters -----------------------------------------------
def make_params():
    return {
        "attn": (
            rng.standard_normal((d, d)),     # Wq
            rng.standard_normal((d, d)),     # Wk
            rng.standard_normal((d, d)),     # Wv
            rng.standard_normal((d, d)),     # Wo
        ),
        "mlp": (
            rng.standard_normal((d, 4 * d)),  # W1
            rng.standard_normal((4 * d, d)),  # W2
        ),
    }


# ----- forward pass --------------------------------------------------------
print("=== Transformer block forward pass ===")
x = rng.standard_normal((B, T, d))
params = make_params()
y = block(x, params)
print(f"input   : {x.shape}")
print(f"output  : {y.shape}")
print(f"shape preserved? {x.shape == y.shape}")

print("\nNext file: 05_model.py — stack blocks into a GPT.")
