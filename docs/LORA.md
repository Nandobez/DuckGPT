# LoRA fine-tuning guide

LoRA (Low-Rank Adaptation, Hu et al. 2021) freezes a pretrained model and
learns a *small* low-rank correction on top. Typical numbers:

| Model | Full fine-tune trainable | LoRA `r=8` trainable |
|---|---|---|
| 124M GPT-2 | 124M | ~0.4M |
| 350M GPT-2 medium | 350M | ~1.0M |

So you can adapt a model to a new domain (e.g. PT-BR literature) with a
3–5 MB adapter file instead of duplicating the whole checkpoint.

## End-to-end recipe

```bash
# 1) train a base model (e.g. on Shakespeare or a multilingual corpus)
python lib/train.py config/train_shakespeare_char.py

# 2) prepare the *target* domain dataset (PT-BR literature)
python data/literatura_ptbr/prepare.py

# 3) fine-tune with a LoRA adapter
python lib/lora.py finetune \
    --base-ckpt out-shakespeare-char/ckpt.pt \
    --dataset data/literatura_ptbr \
    --out adapters/ptbr.pt \
    --r 8 --alpha 16 \
    --iters 1000 \
    --lr 1e-3
```

## What gets patched

By default DuckGPT injects LoRA into the two linear layers per transformer
block that matter most:

- `c_attn` — fused (Q, K, V) projection
- `c_proj` — output projection of attention

You can override:

```bash
python lib/lora.py finetune --targets c_attn,c_proj,fc,proj …
```

To see what was inserted:

```bash
python lib/lora.py info --adapter adapters/ptbr.pt
```

## Sampling with the adapter

```bash
# attach the adapter on the fly (no merge needed)
python - <<'PY'
import torch, sys; sys.path.append("lib")
from model import GPT, GPTConfig
from lora import inject_lora, load_lora

ck = torch.load("out-shakespeare-char/ckpt.pt", weights_only=False)
cfg = GPTConfig(**{k:v for k,v in ck['model_args'].items() if k in GPTConfig.__dataclass_fields__})
model = GPT(cfg)
model.load_state_dict(ck['model'])
inject_lora(model, r=8, alpha=16)
load_lora(model, "adapters/ptbr.pt")

# now `model` behaves like the fine-tuned variant
print(sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable params")
PY
```

For zero-overhead inference, merge the adapter back into the base weights:

```bash
python lib/lora.py merge \
    --base-ckpt out-shakespeare-char/ckpt.pt \
    --adapter adapters/ptbr.pt \
    --out out-merged/ckpt.pt
```

The merged checkpoint is a normal GPT checkpoint and runs at full speed via
`lib/sample.py` / `lib/server.py`.

## Choosing `r` and `alpha`

- `r` (rank) — capacity of the adapter. Common values: 4, 8, 16. Bigger
  ⇒ more expressive but more parameters and easier to overfit.
- `alpha` — scaling factor. Effective update is `(alpha / r) * B @ A`. A
  popular rule of thumb is `alpha = 2 * r` (`r=8 → alpha=16`).
- `dropout` — regularisation on the LoRA path itself. Defaults to 0 because
  the base model already has dropout.

## When *not* to use LoRA

- Tiny base models (≤ 5M params) where full fine-tuning is already cheap.
- Drastically new vocabulary or modality — the base embeddings can't be
  rotated by a low-rank update alone.

For those cases run a full fine-tune via `lib/train.py --init_from=resume`.
