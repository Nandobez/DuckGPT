# Evaluation suite

`lib/eval/` ships four small evaluations plus a top-level runner that prints
a markdown table. They are deliberately tiny so you can run the whole suite
on a CPU in a minute; replace the JSON inputs with bigger official splits
when you need a publishable number.

## Run the full suite

```bash
python lib/eval/run.py --checkpoint out-foo/ckpt.pt \
    --val-bin data/wikipedia_ptbr/val.bin data/shakespeare/val.bin
```

Output (example):

```
## DuckGPT eval — ckpt.pt

| Metric | Value | Notes |
|---|---|---|
| PPL · val (wiki-pt) | 24.31 | 380,512 tokens |
| PPL · val (shakespeare) | 18.72 | 36,059 tokens |
| HellaSwag-mini | 50.0% | 3/6 |
| PT-Grammar | 62.5% | 5/8 |
| BLEU-4 (EN↔PT mini) | 0.0034 | 6 pairs |
```

Pipe directly into a release note or commit message.

## Individual evals

### `lib/eval/perplexity.py`

Runs the model over a stream of token ids (a `.bin` file produced by any
`data/*/prepare.py`) and reports `exp(mean NLL)`.

```bash
python lib/eval/perplexity.py --checkpoint out-foo/ckpt.pt \
    --bin data/wikipedia_ptbr/val.bin
```

### `lib/eval/hellaswag_mini.py`

Tiny multiple-choice eval. Each item has a context and N continuations; the
script picks the one with the highest length-normalised log-likelihood.

The default dataset lives at `lib/eval/data/hellaswag_mini.json` and contains
6 bilingual items. Swap it for the official HellaSwag split for a serious
benchmark:

```bash
python lib/eval/hellaswag_mini.py --checkpoint out-foo/ckpt.pt \
    --data my_hellaswag.json
```

JSON format:

```json
[
  {
    "context": "He picked up a hammer and started",
    "choices": ["…", "…", "…"],
    "answer": 0
  }
]
```

### `lib/eval/pt_grammar.py`

8 Brazilian-Portuguese sentence pairs covering noun agreement, verb
agreement, regency, crasis, and orthography. The model picks the
higher-likelihood sentence; we count how often it picks the grammatical one.

Easy to extend: drop new items into `lib/eval/data/pt_grammar.json`.

### `lib/eval/bleu.py`

Greedy translation EN↔PT against a small reference list, scored with
BLEU-4 on whitespace-tokenised text.

```bash
python lib/eval/bleu.py --checkpoint out-foo/ckpt.pt --max-new 40
```

## Adding your own metric

Drop a new module under `lib/eval/`, write a `main()` that takes
`--checkpoint`, and import / call it from `lib/eval/run.py`. The shared
helpers in `lib/eval/_loader.py` cover:

- `load_checkpoint(path)` — restores model + tokenizer correctly for both
  char-level and BPE checkpoints.
- `sequence_logprob(model, ids, device)` — sum of log-probabilities of a
  sequence under the model. Most multiple-choice evals are just that.
