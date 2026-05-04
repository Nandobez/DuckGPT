# Training guide

Three honest scenarios for training DuckGPT, ordered by how long each one takes.

## 1. Smoke test — 2 minutes on CPU

Use this to verify the pipeline end-to-end before committing real compute.

```bash
# 1) prepare a char-level dataset (no BPE needed)
python data/shakespeare_char/prepare.py

# 2) train the tiny "smoke" config (100 iters, 0.8M params)
python lib/train.py config/smoke.py

# 3) sample
python lib/sample.py --out_dir=out-smoke --start="ROMEO:" \
    --max_new_tokens 100 --stream
```

Expected: training loss falls from ≈4.6 to ≈2.5 in about 100 iterations. Generated
text will be character-level gibberish — that's fine, the goal here is to confirm
that `train.py` → `ckpt.pt` → `sample.py` runs without errors.

## 2. Real Shakespeare-char run — 30 min on CPU / 2-5 min on GPU

The classic Karpathy-style "tiny GPT learns Shakespeare" run. Produces text that
*looks* like Shakespeare.

```bash
python data/shakespeare_char/prepare.py
python lib/train.py config/train_shakespeare_char.py
python lib/sample.py --out_dir=out-shakespeare-char --start="ROMEO:" \
    --max_new_tokens 400 --top_p 0.9 --repetition_penalty 1.15 --stream
```

The default config trains a 6-layer / 6-head / 384-dim model for 5000 iterations,
which fits on any laptop GPU and even a modern CPU.

## 3. Serious bilingual run — hours / days

To get coherent EN + PT-BR output you need a real corpus and real training time.

```bash
# 1) train a BPE tokeniser (~30 min for vocab 16k on Shakespeare; 1-2 h on Wikipedia)
python lib/tokenizer.py train \
    --input data/literatura_ptbr/raw \
    --vocab-size 16000 \
    --out models/bpe

# 2) build the trilingual corpus (downloads Wikipedia PT + WikiText + OPUS-100 en-pt)
python data/multilang_en_pt/prepare.py

#    smoke version (5k docs/source) for end-to-end testing
python data/multilang_en_pt/prepare.py --max-per-source 5000

# 3) train the 8L/8H/512d model (20k iterations)
python lib/train.py config/train_multilang.py
```

Real-world timing for `config/train_multilang.py`:

| Hardware | Approx. wall-clock |
|---|---|
| CPU (modern desktop) | several days |
| RTX 3060 / 3070 (12 GB) | 8–16 h |
| RTX 4090 / A6000 | 2–4 h |
| A100 / H100 | 1–2 h |

## Writing your own config

A config is just a Python file imported by `lib/configurator.py`. Override any
global variable you see at the top of `lib/train.py`:

```python
# config/my_run.py
out_dir = 'out-my-run'
dataset = 'literatura_ptbr'         # name of folder under data/
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
batch_size = 32
gradient_accumulation_steps = 4
max_iters = 8000
learning_rate = 3e-4
```

Run with `python lib/train.py config/my_run.py`.

## Picking sensible hyperparameters

- **block_size** — should be ≤ the longest natural document you want the model
  to handle. 256 for char-level, 512–1024 for BPE.
- **batch_size × gradient_accumulation_steps** — effective batch. Aim for a few
  thousand tokens per step (e.g. 32 × 4 × 256 ≈ 32k).
- **learning_rate** — 3e-4 is a strong default for AdamW + cosine warm-up.
  Halve it if you go below 1024 tokens per step; double it for very large batches.
- **weight_decay** — 0.1 by default; turn off (0.0) for tiny models.
- **dropout** — 0.0 for tiny runs, 0.1 for medium, up to 0.2 for char-level
  Shakespeare where overfit is the main risk.

## Resuming / fine-tuning

```bash
python lib/train.py config/my_run.py --init_from=resume
```

The script will read `out_dir/ckpt.pt` and continue from the last iteration.

For fine-tuning a pretrained model on a new dataset, see
[`docs/LORA.md`](LORA.md) — adapters are usually a better choice than full
fine-tuning for small datasets.
