# Sampling guide

`lib/sample.py` is a CLI wrapper around a single function — `stream_generate`
(or `beam_search`) — that turns a checkpoint plus a prompt into text.

## Quick reference

```bash
python lib/sample.py --out_dir=out-foo --start="Era uma vez" \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_p 0.9 \
    --top_k 40 \
    --repetition_penalty 1.15 \
    --stream
```

## Knobs

| Flag | What it does |
|---|---|
| `--start` | Initial prompt. Use `FILE:path.txt` to load it from a file. |
| `--max_new_tokens` | Hard upper bound on tokens to generate. |
| `--temperature` | Divide logits by this. `< 1.0` sharpens (more deterministic), `> 1.0` flattens (more chaotic). |
| `--top_k` | Keep only the top-k most likely tokens. `0` disables. |
| `--top_p` | Nucleus sampling: keep the smallest set with cumulative probability ≥ p. `0.0` disables. Combine with `--top_k` for "double filter". |
| `--repetition_penalty` | Divide the logit of any previously-generated token by this factor (CTRL-style). 1.0 = off, 1.1–1.3 = typical. |
| `--stop` | One or more stop strings; generation halts as soon as any appears in the decoded buffer. |
| `--stream` | Print tokens as they are produced. |
| `--beam_size` | If > 0, use beam search instead of stochastic sampling. Length-normalised score. |
| `--num_samples` | Repeat the entire generation N times with different RNG seeds. |
| `--device` | `cuda` / `mps` / `cpu`. Defaults to the best available. |
| `--seed` | Reproducible sampling. |

## Choosing settings

- **Pure greedy** — `--temperature 1.0 --top_k 1` or `--beam_size 4`. Best for
  factual / deterministic output. Beam picks higher-quality but more boring text.
- **Nucleus (default for chat-style)** — `--top_p 0.9 --temperature 0.8`.
  Good balance of coherence and creativity.
- **Very creative** — `--temperature 1.2 --top_p 0.95 --repetition_penalty 1.1`.
  Useful for fiction / brainstorming. Expect more rambling.
- **Avoid loops** — bump `--repetition_penalty` to 1.15–1.3 and/or add a
  `--stop` string for whatever your model loves to repeat.

## Streaming from Python

```python
from lib.sample import stream_generate, _load_checkpoint, _resolve_codec
import torch

model, ck = _load_checkpoint("out-foo/ckpt.pt", "cuda")
encode, decode_one = _resolve_codec(ck)
ids = torch.tensor([encode("Era uma vez")], device="cuda")

for tok, ch in stream_generate(
    model, ids, decode_one,
    max_new_tokens=200, temperature=0.8,
    top_k=None, top_p=0.9, repetition_penalty=1.15, stop=[],
):
    print(ch, end="", flush=True)
```

## Beam search

```bash
python lib/sample.py --out_dir=out-foo --start="Hello, world." \
    --max_new_tokens 50 --beam_size 4
```

The script prints the best beam and its log-probability score. Length penalty
is fixed at 1.0; tweak it in `lib/sample.py::beam_search` if you want longer or
shorter outputs.
