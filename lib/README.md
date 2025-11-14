# `lib/` — production-grade DuckGPT

This is the *working* implementation of DuckGPT. Designed for actual training runs,
not for reading top-to-bottom. If you are here to *learn* how a transformer works,
start in [`../core/`](../core/README.md) instead — it walks through every concept
file by file with heavy comments and tiny examples.

## What lives here

| File | Purpose | Notable optimisations |
|---|---|---|
| `tokenizer.py` | byte-level BPE | vectorised pair counts (numpy `np.unique`), batched merges, special tokens |
| `model.py` | Transformer decoder | weight tying, scaled init, `F.scaled_dot_product_attention` (FlashAttention when available), causal mask cache |
| `optim.py` | hand-rolled AdamW + cosine warm-up | in-place tensor ops, decoupled weight decay (Loshchilov & Hutter 2017) |
| `train.py` | training loop | auto device pick (CUDA / MPS / CPU), AMP/bf16 autocast, gradient clipping, checkpointing |
| `sample.py` | generation | top-k + temperature, optional `torch.compile`, autocast inference |
| `bench.py` | micro-benchmarks for the forward/backward step | |
| `configurator.py` | tiny config loader that turns `config/foo.py` into globals | |

## Why each optimisation?

- **Vectorised BPE pair counting** — pure-Python `Counter` over the corpus is
  O(N) per merge; the numpy version pre-computes all adjacent pairs in one
  vectorised pass and is ~50× faster on Shakespeare.
- **Weight tying (`wte.weight = lm_head.weight`)** — saves ≈30 % of parameters
  and usually improves generalisation (Press & Wolf 2017).
- **`F.scaled_dot_product_attention`** — PyTorch routes this to FlashAttention
  when running on CUDA + bf16 / fp16; otherwise falls back to the math kernel.
  Either way it avoids materialising the `(B, H, T, T)` tensor that the naive
  implementation in `core/` builds.
- **Auto device pick** — `train.py` and `sample.py` default to the best
  available backend (CUDA > MPS > CPU). Override with `device=cpu` from the CLI.
- **AMP / bf16 autocast** — halves memory and roughly doubles throughput on
  Ampere+ GPUs while keeping the optimiser state in fp32.
- **Decoupled weight decay** — applying `θ ← (1 − ηλ)θ` *before* the adaptive
  step decouples regularisation from the moment-normalised gradient direction,
  which empirically helps Transformers.

## Quickstart

```bash
# 1) train a tokenizer (≈ 1–5 min on the Shakespeare corpus)
python lib/tokenizer.py train --input data/shakespeare/input.txt \
    --vocab-size 4096 --out models/bpe

# 2) prepare a dataset (encodes corpus → train.bin / val.bin)
python data/shakespeare/prepare.py

# 3) train (auto picks CUDA if present)
python lib/train.py config/smoke.py        # 100-iter smoke test, ~2 min on CPU
python lib/train.py config/train_shakespeare_char.py   # ~30–60 min CPU / 2–5 min GPU

# 4) sample
python lib/sample.py --out_dir=out-smoke --start="ROMEO:" --max_new_tokens=400
```

## Going further

Once you're comfortable, the natural next steps are wired in as `notebooks/`:
- `notebooks/05_core_to_lib.ipynb` — diff between the didactic version and this
  one, with a wall-clock benchmark for each optimisation.

For production deployment, `server.py` is on the roadmap — a thin FastAPI wrapper
exposing `/generate` and `/stream` endpoints, swappable between any checkpoint
on disk.
