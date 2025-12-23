"""
DuckGPT — prepare the Shakespeare corpus.

Downloads the complete works from Project Gutenberg (public domain), tokenises
with the DuckGPT BPE checkpoint (trained ahead of time if needed), and writes
``train.bin`` / ``val.bin`` as uint16 arrays.

If no tokenizer is found at ``models/bpe``, one is trained on the corpus first.
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np

# allow `import tokenizer` from the project root no matter where this is run
sys.path.append(str(Path(__file__).resolve().parents[2]))
from tokenizer import BPETokenizer  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_URL = "https://www.gutenberg.org/files/100/100-0.txt"
INPUT = HERE / "input.txt"
TOK_PREFIX = Path(__file__).resolve().parents[2] / "models" / "bpe"

def maybe_download():
    if INPUT.exists():
        return
    print(f"Downloading {DATA_URL}…")
    with urllib.request.urlopen(DATA_URL, timeout=60) as r:
        INPUT.write_bytes(r.read())

def maybe_train_tokenizer(text: str, vocab_size: int = 8192):
    if Path(str(TOK_PREFIX) + ".merges").exists():
        return
    print(f"Training BPE tokenizer on Shakespeare corpus (vocab={vocab_size})…")
    tok = BPETokenizer(special_tokens={"<|endoftext|>": -1})
    tok.train(text, vocab_size=vocab_size, verbose=True)
    TOK_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(TOK_PREFIX))

def main():
    maybe_download()
    text = INPUT.read_text(encoding="utf-8", errors="ignore")
    print(f"corpus: {len(text):,} chars")

    maybe_train_tokenizer(text)
    tok = BPETokenizer()
    tok.load(str(TOK_PREFIX))

    n = len(text)
    train_text = text[: int(0.9 * n)]
    val_text = text[int(0.9 * n):]
    train_ids = tok.encode(train_text)
    val_ids = tok.encode(val_text)
    print(f"train: {len(train_ids):,} tokens   val: {len(val_ids):,} tokens")

    np.array(train_ids, dtype=np.uint16).tofile(HERE / "train.bin")
    np.array(val_ids, dtype=np.uint16).tofile(HERE / "val.bin")

if __name__ == "__main__":
    main()
