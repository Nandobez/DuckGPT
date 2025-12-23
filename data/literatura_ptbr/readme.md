# Literatura PT-BR

Public-domain Brazilian / Portuguese literature corpus (Machado de Assis,
Eça de Queirós, José de Alencar, Camões). Small but stylistically rich — useful
for fine-tuning a Portuguese language model on classical prose.

## Prepare

```bash
pip install tiktoken numpy
python data/literatura_ptbr/prepare.py
```

Outputs `train.bin` and `val.bin` (uint16 GPT-2 BPE tokens) in this folder.
Cached raw texts go into `raw/`.

## Train

```bash
python train.py config/train_literatura_ptbr.py
```
