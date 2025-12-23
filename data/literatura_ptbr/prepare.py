"""
DuckGPT — public-domain Brazilian / Portuguese literature corpus.

Pulls a curated set of public-domain works from Project Gutenberg (Machado de Assis,
Eça de Queirós, José de Alencar, Camões, etc.), strips Gutenberg headers/footers,
concatenates everything, tokenises with GPT-2 BPE and writes train.bin / val.bin.

Usage:
    python data/literatura_ptbr/prepare.py
"""
import re
import urllib.request
from pathlib import Path

import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[1]))
from tokenizer import BPETokenizer  # noqa: E402
TOK_PREFIX = HERE.parents[1] / "models" / "bpe"

# Project Gutenberg book IDs (all public domain, PT or PT-BR).
BOOKS = [
    # Machado de Assis
    (55752, "machado_dom_casmurro"),
    (54829, "machado_memorias_postumas"),
    (52961, "machado_quincas_borba"),
    (54829, "machado_helena"),
    # Eça de Queirós
    (45330, "eca_o_crime_do_padre_amaro"),
    (16384, "eca_os_maias"),
    # José de Alencar
    (38496, "alencar_iracema"),
    (37001, "alencar_o_guarani"),
    # Camões
    (3333,  "camoes_os_lusiadas"),
]

START_PATTERNS = [
    re.compile(r"\*\*\* START OF (?:THIS|THE) PROJECT GUTENBERG.*?\*\*\*", re.I),
    re.compile(r"\*END\*THE SMALL PRINT.*?\*END\*", re.I | re.S),
]
END_PATTERNS = [
    re.compile(r"\*\*\* END OF (?:THIS|THE) PROJECT GUTENBERG.*?\*\*\*", re.I),
    re.compile(r"End of (?:the|this) Project Gutenberg.*", re.I),
]

def strip_gutenberg_chrome(text: str) -> str:
    """Remove the Gutenberg legal headers and footers."""
    for pat in START_PATTERNS:
        m = pat.search(text)
        if m:
            text = text[m.end():]
            break
    for pat in END_PATTERNS:
        m = pat.search(text)
        if m:
            text = text[:m.start()]
            break
    return text.strip()

def fetch(book_id: int) -> str:
    for suffix in ("-0.txt", ".txt"):
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}{suffix}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception:
            continue
    raise RuntimeError(f"could not download book {book_id}")

def main():
    raw_dir = HERE / "raw"
    raw_dir.mkdir(exist_ok=True)
    pieces = []
    for book_id, name in BOOKS:
        cache = raw_dir / f"{book_id}.txt"
        if not cache.exists():
            print(f"  downloading {name} (#{book_id})…")
            cache.write_text(fetch(book_id), encoding="utf-8")
        text = strip_gutenberg_chrome(cache.read_text(encoding="utf-8"))
        if not text:
            print(f"  WARN: {name} stripped to empty, skipping")
            continue
        pieces.append(text)
        print(f"  {name}: {len(text):,} chars")
    corpus = "\n\n".join(pieces)
    print(f"\nTotal corpus: {len(corpus):,} chars")

    if not Path(str(TOK_PREFIX) + ".merges").exists():
        print(f"Training DuckGPT BPE on corpus (vocab=16000)…")
        tok = BPETokenizer(special_tokens={"<|endoftext|>": -1})
        tok.train(corpus, vocab_size=16000, verbose=True)
        TOK_PREFIX.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(TOK_PREFIX))
    else:
        tok = BPETokenizer()
        tok.load(str(TOK_PREFIX))
    eot = tok.special_tokens.get("<|endoftext|>") or tok.add_special("<|endoftext|>")
    ids = tok.encode_ordinary(corpus)
    ids.append(eot)
    arr = np.array(ids, dtype=np.uint16)

    split = int(0.999 * len(arr))
    arr[:split].tofile(HERE / "train.bin")
    arr[split:].tofile(HERE / "val.bin")
    print(f"train.bin: {split:,} tokens   val.bin: {len(arr)-split:,} tokens")

if __name__ == "__main__":
    main()
