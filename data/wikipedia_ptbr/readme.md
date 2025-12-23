# Wikipedia PT-BR

Brazilian-Portuguese Wikipedia dump tokenised with GPT-2 BPE. Public-domain corpus
(~1.1M articles, ~600M tokens after BPE) — good base for a Portuguese language model.

## Prepare

```bash
pip install datasets tiktoken tqdm numpy
python data/wikipedia_ptbr/prepare.py
# smoke test on a small slice:
python data/wikipedia_ptbr/prepare.py --max-articles 5000
```

Outputs `train.bin` and `val.bin` in this folder.

## Train

Use the matching config: `config/train_wikipedia_ptbr.py`.

```bash
python train.py config/train_wikipedia_ptbr.py
```
