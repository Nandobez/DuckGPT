"""
DuckGPT — character-level Shakespeare dataset prep.

Maps every unique character to an integer (no BPE). Useful for the
"char-level" smoke-test config. Saves train.bin/val.bin + meta.pkl.
"""
import pickle
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input.txt"
DATA_URL = "https://www.gutenberg.org/files/100/100-0.txt"

def main():
    if not INPUT.exists():
        print(f"Downloading {DATA_URL}…")
        with urllib.request.urlopen(DATA_URL, timeout=60) as r:
            INPUT.write_bytes(r.read())

    text = INPUT.read_text(encoding="utf-8", errors="ignore")
    print(f"corpus: {len(text):,} chars")

    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"unique chars: {vocab_size}")
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    n = len(text)
    train_text = text[: int(0.9 * n)]
    val_text = text[int(0.9 * n):]
    train_ids = [stoi[c] for c in train_text]
    val_ids = [stoi[c] for c in val_text]
    print(f"train: {len(train_ids):,}   val: {len(val_ids):,}")

    np.array(train_ids, dtype=np.uint16).tofile(HERE / "train.bin")
    np.array(val_ids, dtype=np.uint16).tofile(HERE / "val.bin")

    with open(HERE / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

if __name__ == "__main__":
    main()
