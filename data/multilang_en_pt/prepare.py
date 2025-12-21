"""
DuckGPT — bilingual EN + PT corpus.

Combines three sources so the model sees encyclopedic text *and* dialogue
in both languages:

- **English encyclopedic**: WikiText-103 (Salesforce, CC BY-SA)
- **Portuguese encyclopedic**: wikimedia/wikipedia 20231101.pt
- **Bilingual dialogue / spoken**: Helsinki-NLP/opus-100 en-pt parallel
  sentence pairs (subtitles, EuroParl, books — conversational language)

Each document is tokenised with the in-house DuckGPT BPE, separated by the
``<|endoftext|>`` token. We interleave documents at the source level so the
model alternates between languages and registers throughout training.

Usage:
    python data/multilang_en_pt/prepare.py [--max-per-source N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[1]))
from tokenizer import BPETokenizer  # noqa: E402
TOK_PREFIX = HERE.parents[1] / "models" / "bpe"


def tokenise_stream(dataset, text_field, enc, eot):
    for row in dataset:
        text = row[text_field] if isinstance(text_field, str) else row.get(text_field[0]) or row.get(text_field[1])
        if not text:
            continue
        ids = enc.encode_ordinary(text)
        ids.append(eot)
        yield ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-source", type=int, default=200_000,
                        help="Cap the number of documents pulled from each source.")
    parser.add_argument("--val-tokens", type=int, default=200_000)
    args = parser.parse_args()

    if not Path(str(TOK_PREFIX) + ".merges").exists():
        sys.exit(
            f"No tokenizer at {TOK_PREFIX}.merges — train one first via "
            f"data/literatura_ptbr/prepare.py or `python tokenizer.py train ...`"
        )
    enc = BPETokenizer()
    enc.load(str(TOK_PREFIX))
    eot = enc.special_tokens.get("<|endoftext|>") or enc.add_special("<|endoftext|>")

    print("Loading WikiText-103 (encyclopedic, English)…")
    en = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    if args.max_per_source:
        en = en.select(range(min(args.max_per_source, len(en))))

    print("Loading Wikipedia PT (encyclopedic, Portuguese)…")
    pt = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train")
    if args.max_per_source:
        pt = pt.select(range(min(args.max_per_source, len(pt))))

    print("Loading OPUS-100 EN↔PT (dialogue / falas)…")
    opus = load_dataset("Helsinki-NLP/opus-100", "en-pt", split="train")
    if args.max_per_source:
        opus = opus.select(range(min(args.max_per_source, len(opus))))

    print("Tokenising + interleaving the three streams…")
    en_iter = tokenise_stream(en, "text", enc, eot)
    pt_iter = tokenise_stream(pt, "text", enc, eot)

    # Each opus row is {"translation": {"en": "...", "pt": "..."}}.
    def opus_iter():
        for row in opus:
            for lang in ("en", "pt"):
                text = row["translation"].get(lang)
                if not text:
                    continue
                ids = enc.encode_ordinary(text)
                ids.append(eot)
                yield ids

    op = opus_iter()
    en_done = pt_done = op_done = False
    buf: list[int] = []
    while not (en_done and pt_done and op_done):
        for label, src in (("en", en_iter), ("pt", pt_iter), ("opus", op)):
            try:
                buf.extend(next(src))
            except StopIteration:
                if label == "en":
                    en_done = True
                elif label == "pt":
                    pt_done = True
                else:
                    op_done = True

    arr = np.array(buf, dtype=np.uint16)
    print(f"  total tokens: {len(arr):,}")

    split = max(0, len(arr) - args.val_tokens)
    arr[:split].tofile(HERE / "train.bin")
    arr[split:].tofile(HERE / "val.bin")
    print(f"  train.bin: {split:,} tokens   val.bin: {len(arr)-split:,} tokens")


if __name__ == "__main__":
    main()
