# `core/` — didactic DuckGPT

A walkthrough of every piece of a GPT, in the order I learned it. Each file is
short, heavily commented, and references the relevant paper / equation. The
code is intentionally **slow and naive** — clarity trumps speed. Once you have
read everything here, look at [`../lib/`](../lib/README.md) for the optimised
counterpart.

## Suggested reading order

| Step | File | What you'll learn |
|---|---|---|
| 1 | `01_tensors.py` | Tensors, matrix multiplication, softmax — built from numpy and verified against torch |
| 2 | `02_tokenizer.py` | Byte-level BPE, printing each merge as it is learned |
| 3 | `03_attention.py` | Scaled dot-product attention with manual broadcasting, plus a heatmap |
| 4 | `04_transformer_block.py` | One full encoder/decoder block, with shape annotations |
| 5 | `05_model.py` | A minimal GPT (~200 lines), no tricks |
| 6 | `06_optim.py` | Adam from scratch + a cosine LR schedule |
| 7 | `07_train.py` | A single-process training loop on Shakespeare |
| 8 | `08_sample.py` | Greedy + temperature sampling |

Each file can be **run directly**:

```bash
python core/01_tensors.py
python core/02_tokenizer.py
…
```

They print intermediate tensors, expected shapes and (where useful) compare
their output to PyTorch's built-ins so you can convince yourself the maths is
right.

## Defaults

Every example uses **tiny tensors and few iterations** — they all run on CPU in
seconds. You won't get a publication-quality model here. The point is for you
to be able to step through the math with `pdb` or in a notebook without
waiting.

When you want speed and a real training run, move on to [`../lib/`](../lib/).
