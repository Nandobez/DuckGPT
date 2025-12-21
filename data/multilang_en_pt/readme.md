# Multilingual EN + PT corpus

Combines three sources so the model picks up encyclopedic *and* conversational
patterns in both languages:

| Stream | Source | Register |
|---|---|---|
| English encyclopedic | WikiText-103 | formal prose |
| Portuguese encyclopedic | wikimedia/wikipedia 20231101.pt | formal prose |
| EN↔PT dialogue | Helsinki-NLP/opus-100 en-pt (subtitles + EuroParl + books) | conversational, parallel sentences |

All three streams are tokenised with the DuckGPT BPE checkpoint at
`models/bpe.*` and concatenated with the `<|endoftext|>` token between
documents.

## Prepare

```bash
# 1) ensure a tokeniser is trained
python lib/tokenizer.py train --input data/literatura_ptbr/raw \
    --vocab-size 16000 --out models/bpe

# 2) build the bilingual + dialogue corpus
python data/multilang_en_pt/prepare.py            # uses 200k docs per source
python data/multilang_en_pt/prepare.py --max-per-source 5000   # smoke test
```

Produces `train.bin` + `val.bin` in this folder.

## Train

```bash
python lib/train.py config/train_multilang.py
```
