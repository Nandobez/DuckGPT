<div align="center">

<pre>
██████╗ ██╗   ██╗ ██████╗██╗  ██╗ ██████╗ ██████╗ ████████╗
██╔══██╗██║   ██║██╔════╝██║ ██╔╝██╔════╝ ██╔══██╗╚══██╔══╝
██║  ██║██║   ██║██║     █████╔╝ ██║  ███╗██████╔╝   ██║
██║  ██║██║   ██║██║     ██╔═██╗ ██║   ██║██╔═══╝    ██║
██████╔╝╚██████╔╝╚██████╗██║  ██╗╚██████╔╝██║        ██║
╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝        ╚═╝
</pre>

### A GPT-2 style language model — built from scratch, bilingual (EN + PT-BR)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](./LICENSE)

</div>

DuckGPT is a GPT-2 style language model trained end-to-end on **English + Brazilian Portuguese**. Everything that is not the training data is written from scratch — model, tokeniser, optimiser, training loop and sampling.

It ships in **two parallel layers**:

| Layer | Folder | Purpose |
|---|---|---|
| 🎓 Didactic | [`core/`](./core) | One concept per file, heavy comments, runs on CPU in seconds. Start here to understand how a transformer works. |
| 🏎️ Production | [`lib/`](./lib) | Vectorised BPE, FlashAttention via `sdpa`, AMP/bf16, DDP-ready training. **Plus**: nucleus + beam sampling, LoRA fine-tuning, evaluation suite, FastAPI server, distillation. |
| 🚀 Deploy | [`deploy/`](./deploy) | Dockerfile + docker-compose for the FastAPI inference server. |

There are also [`notebooks/`](./notebooks) — bridge notebooks that load both layers side by side and benchmark the differences.

## Suggested journey

```
1. core/01_tensors.py            ← tensors, matmul, softmax
2. core/02_tokenizer.py          ← byte-level BPE, printing every merge
3. core/03_attention.py          ← scaled dot-product attention + causal mask
4. core/04_transformer_block.py  ← pre-norm block, shapes annotated
5. core/05_model.py              ← minimal GPT (~150 lines)
6. core/06_optim.py              ← Adam + cosine warm-up
7. core/07_train.py              ← full training loop on Shakespeare
8. core/08_sample.py             ← greedy + temperature sampling
↓
notebooks/01..05                 ← visualisations, benchmarks, diffs
↓
lib/                             ← run a real GPT-2 124M training
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy datasets tqdm matplotlib

# 1. Read & run the didactic scripts (each takes < 30 s)
python core/01_tensors.py
python core/02_tokenizer.py
python core/03_attention.py
python core/04_transformer_block.py
python core/05_model.py
python core/06_optim.py

# 2. Prepare a tiny Shakespeare-char dataset
python data/shakespeare_char/prepare.py

# 3. Run the didactic training loop (a few minutes on CPU)
python core/07_train.py
python core/08_sample.py --start "ROMEO:" --max-new-tokens 200

# 4. When you want a real run, switch to lib/
python lib/tokenizer.py train --input data/literatura_ptbr/raw \
    --vocab-size 16000 --out models/bpe
python data/literatura_ptbr/prepare.py
python lib/train.py config/train_shakespeare_char.py
python lib/sample.py --out_dir=out-shakespeare-char --start="Era uma vez" \
    --top_p 0.9 --repetition_penalty 1.15 --stream
```

## Extra capabilities

### LoRA fine-tuning

```bash
# fine-tune a Shakespeare-trained model on PT-BR literature
python lib/lora.py finetune --base-ckpt out-shakespeare/ckpt.pt \
    --dataset data/literatura_ptbr --out adapters/ptbr.pt \
    --r 8 --alpha 16 --iters 1000

# inspect
python lib/lora.py info --adapter adapters/ptbr.pt

# merge for zero-overhead inference
python lib/lora.py merge --base-ckpt out-shakespeare/ckpt.pt \
    --adapter adapters/ptbr.pt --out out-merged/ckpt.pt
```

### Evaluation

```bash
python lib/eval/run.py --checkpoint out/ckpt.pt \
    --val-bin data/wikipedia_ptbr/val.bin
# prints a markdown table: perplexity · HellaSwag-mini · PT-Grammar · BLEU-4
```

### Distillation (teacher → student)

```bash
python lib/distill.py --teacher-ckpt out-large/ckpt.pt \
    --dataset data/wikipedia_ptbr --out out-student/ckpt.pt \
    --student-config '{"n_layer":4,"n_head":4,"n_embd":256}' \
    --alpha 0.8 --temperature 2.0 --iters 5000
```

### Inference server (FastAPI)

```bash
pip install fastapi 'uvicorn[standard]'
export DUCKGPT_CKPT=$PWD/out-shakespeare-char/ckpt.pt
python -m uvicorn lib.server:app --host 0.0.0.0 --port 8000

# or via Docker:
docker compose -f deploy/docker-compose.yml up --build
```

Endpoints: `GET /health`, `POST /generate`, `WebSocket /stream`.

## Datasets (only external piece)

| Folder | Source | Language |
|---|---|---|
| `data/shakespeare` | Project Gutenberg #100 (BPE) | English |
| `data/shakespeare_char` | Project Gutenberg #100 (char-level) | English |
| `data/literatura_ptbr` | Project Gutenberg (Machado, Eça, Alencar, Camões…) | PT-BR |
| `data/wikipedia_ptbr` | wikimedia/wikipedia 20231101.pt | PT-BR |
| `data/multilang_en_pt` | WikiText-103 + Wikipedia PT (interleaved) | EN + PT-BR |
| `data/openwebtext` | OpenWebText (HF datasets) | English |

## Repo layout

```
DuckGPT/
├── README.md
├── LICENSE
├── core/                   # didactic — read in numeric order
│   ├── 01_tensors.py
│   ├── 02_tokenizer.py
│   ├── 03_attention.py
│   ├── 04_transformer_block.py
│   ├── 05_model.py
│   ├── 06_optim.py
│   ├── 07_train.py
│   ├── 08_sample.py
│   └── README.md
├── lib/                    # production — DDP, AMP, FlashAttention, …
│   ├── tokenizer.py        # BPE (vectorised)
│   ├── model.py            # GPT-2 style transformer
│   ├── optim.py            # hand-rolled AdamW + cosine warm-up
│   ├── train.py            # DDP-ready training loop with AMP
│   ├── sample.py           # top-k / top-p / repetition penalty / streaming / beam
│   ├── lora.py             # LoRA fine-tuning (inject, save, merge)
│   ├── distill.py          # KD: train a small student to mimic a large teacher
│   ├── server.py           # FastAPI: /generate (REST) + /stream (WebSocket)
│   ├── eval/               # PPL, HellaSwag-mini, PT-Grammar, BLEU
│   ├── bench.py
│   ├── configurator.py
│   └── README.md
├── deploy/                 # Dockerfile + docker-compose for `lib/server.py`
├── notebooks/              # visualisations + benchmarks tying core ↔ lib
│   ├── 01_bpe_walkthrough.ipynb
│   ├── 02_attention_heatmap.ipynb
│   ├── 03_block_shapes.ipynb
│   ├── 04_training_dynamics.ipynb
│   └── 05_core_to_lib.ipynb
├── config/                 # training configs
└── data/                   # dataset prep scripts
```

## Documentation

Deeper guides live under [`docs/`](./docs):

- [TRAINING.md](./docs/TRAINING.md) — three scenarios (smoke / Shakespeare / trilingual), hyperparameters
- [SAMPLING.md](./docs/SAMPLING.md) — every CLI flag of `lib/sample.py`, plus a Python client snippet
- [LORA.md](./docs/LORA.md) — fine-tuning workflow + merge for zero-overhead inference
- [EVAL.md](./docs/EVAL.md) — running the full eval suite, swapping in larger splits
- [SERVER.md](./docs/SERVER.md) — FastAPI endpoints, curl examples, Docker
- [DISTILLATION.md](./docs/DISTILLATION.md) — teacher → student knowledge distillation

## Author

Fernando Bezerra — [@Nandobez](https://github.com/Nandobez)

## License

MIT — see [`LICENSE`](./LICENSE).
