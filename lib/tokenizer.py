"""
DuckGPT — byte-level BPE tokenizer, implemented from scratch.

Single-file implementation. No external tokeniser libraries.

Workflow
--------
1. `BPETokenizer().train(text, vocab_size)` — learn merges from a corpus.
2. `tok.save("models/bpe")` — write merges to disk.
3. `tok.load("models/bpe")` — reload later.
4. `tok.encode(string)` / `tok.decode(ids)` — round-trip text.

Pre-tokenisation uses a GPT-2-style regex limited to characters the `re`
module can match natively (ASCII + Latin Extended), which is enough for
English + Brazilian Portuguese. Internally the tokeniser operates on bytes,
so any UTF-8 input round-trips correctly.

CLI:
    python tokenizer.py train  --input data/literatura_ptbr/raw  --vocab-size 8192  --out models/bpe
    python tokenizer.py encode --model models/bpe --text "Olá mundo"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

try:  # numpy makes pair counting ~50x faster, but the tokenizer still works without it
    import numpy as _np
except ModuleNotFoundError:
    _np = None

# ----------------------------------------------------------------------------
# Pre-tokenisation
# ----------------------------------------------------------------------------
# GPT-2 inspired but written using the standard `re` module (no `regex` dep).
# Groups, in priority order:
#   1. English contractions ('s, 't, 're, 've, 'm, 'll, 'd)
#   2. Optional leading space + letters (ASCII + Latin Extended)
#   3. Optional leading space + digits
#   4. Optional leading space + run of non-letter/non-digit (punctuation)
#   5. Trailing whitespace / standalone whitespace
PRETOK_RE = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-zÀ-ÿ]+| ?\d+| ?[^A-Za-z0-9\s\xC0-\xFF]+|\s+(?!\S)|\s+"
)

# ----------------------------------------------------------------------------
# Core BPE operations
# ----------------------------------------------------------------------------

def _get_pair_counts(token_lists, counts):
    """Tally how often each (a, b) pair occurs across all pre-tokenised chunks.

    Vectorised with numpy when available (~50× faster on long corpora) and
    falls back to a pure-Python implementation otherwise.
    """
    if _np is not None:
        return _get_pair_counts_np(token_lists, counts)
    pair_counts: Counter = Counter()
    for ids, mult in zip(token_lists, counts):
        for a, b in zip(ids, ids[1:]):
            pair_counts[(a, b)] += mult
    return pair_counts


def _get_pair_counts_np(token_lists, counts):
    """Vectorised pair counting. Concatenates all chunks into one int32 array
    and uses np.unique on (left, right) pairs, weighted by chunk multiplicity.
    Chunk boundaries are excluded by masking the seam between consecutive chunks.
    """
    seqs = [_np.asarray(ids, dtype=_np.int32) for ids in token_lists]
    arr = _np.concatenate(seqs)
    # repeat each chunk's multiplicity for every of its tokens
    mult = _np.concatenate([_np.full(len(s), m, dtype=_np.int64)
                            for s, m in zip(seqs, counts)])
    if arr.size < 2:
        return Counter()
    # pair = (arr[i], arr[i+1]); weight = mult[i] (apply to left token)
    left = arr[:-1]
    right = arr[1:]
    weight = mult[:-1].copy()
    # zero-out weights where i is the last position of a chunk
    seam_idx = _np.cumsum([len(s) for s in seqs]) - 1   # last index of each chunk
    seam_idx = seam_idx[seam_idx < len(weight)]
    weight[seam_idx] = 0
    # pack pairs into a single int64 key for np.unique
    keys = left.astype(_np.int64) * (1 << 32) + right.astype(_np.int64)
    uniq, inv = _np.unique(keys, return_inverse=True)
    tally = _np.bincount(inv, weights=weight, minlength=len(uniq))
    pair_counts: Counter = Counter()
    for k, c in zip(uniq.tolist(), tally.tolist()):
        if c <= 0:
            continue
        a = (k >> 32) & 0xFFFFFFFF
        b = k & 0xFFFFFFFF
        pair_counts[(int(a), int(b))] = int(c)
    return pair_counts

def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every adjacent (a, b) in `ids` with `new_id`."""
    out: list[int] = []
    i = 0
    a, b = pair
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

# ----------------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------------

class BPETokenizer:
    def __init__(self, special_tokens: dict[str, int] | None = None):
        # vocab[id] -> raw bytes for that token. 0..255 are initialised lazily.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # ordered list of merges so encode() reproduces training-time choices
        self.merges: list[tuple[int, int]] = []
        self.merge_index: dict[tuple[int, int], int] = {}
        self.special_tokens: dict[str, int] = special_tokens or {}
        self._special_re: re.Pattern | None = None
        self._build_special_re()

    # ----- training -----
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256, "vocab_size must be at least 256 (one per byte)"
        num_merges = vocab_size - 256 - len(self.special_tokens)
        if num_merges <= 0:
            return

        # 1. Pre-tokenise + dedupe so we count each unique chunk once.
        chunk_counts: Counter = Counter(m.group(0) for m in PRETOK_RE.finditer(text))
        token_lists = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        counts = [chunk_counts[chunk] for chunk in chunk_counts]

        next_id = 256
        for step in range(num_merges):
            pair_counts = _get_pair_counts(token_lists, counts)
            if not pair_counts:
                break
            pair = max(pair_counts, key=pair_counts.get)
            freq = pair_counts[pair]
            new_id = next_id
            next_id += 1

            token_lists = [_merge(ids, pair, new_id) for ids in token_lists]
            self.merges.append(pair)
            self.merge_index[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose and (step + 1) % 200 == 0:
                print(f"  merge {step+1:>5}/{num_merges}: "
                      f"{self._render(pair[0])!r} + {self._render(pair[1])!r} "
                      f"-> id {new_id}  (freq={freq})")

        # finally register special tokens above the merged range
        for tok, _ in list(self.special_tokens.items()):
            self.special_tokens[tok] = next_id
            self.vocab[next_id] = tok.encode("utf-8")
            next_id += 1
        self._build_special_re()

    # ----- encoding -----
    def encode_ordinary(self, text: str) -> list[int]:
        """Encode without honouring special tokens."""
        out: list[int] = []
        for m in PRETOK_RE.finditer(text):
            out.extend(self._encode_chunk(m.group(0)))
        return out

    def encode(self, text: str) -> list[int]:
        """Encode while preserving special tokens as atomic units."""
        if not self.special_tokens or self._special_re is None:
            return self.encode_ordinary(text)
        out: list[int] = []
        pos = 0
        for m in self._special_re.finditer(text):
            if m.start() > pos:
                out.extend(self.encode_ordinary(text[pos:m.start()]))
            out.append(self.special_tokens[m.group(0)])
            pos = m.end()
        if pos < len(text):
            out.extend(self.encode_ordinary(text[pos:]))
        return out

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # find the pair with the smallest merge index (i.e. earliest learned)
            best_pair = None
            best_rank = None
            for a, b in zip(ids, ids[1:]):
                rank = self.merge_index.get((a, b))
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = (a, b)
            if best_pair is None:
                break
            ids = _merge(ids, best_pair, self.merge_index[best_pair])
        return ids

    # ----- decoding -----
    def decode(self, ids: Iterable[int]) -> str:
        pieces = b"".join(self.vocab[i] for i in ids)
        return pieces.decode("utf-8", errors="replace")

    # ----- specials -----
    def add_special(self, name: str) -> int:
        next_id = 256 + len(self.merges) + len(self.special_tokens)
        self.special_tokens[name] = next_id
        self.vocab[next_id] = name.encode("utf-8")
        self._build_special_re()
        return next_id

    def _build_special_re(self) -> None:
        if not self.special_tokens:
            self._special_re = None
            return
        # Longest-first so e.g. "<|endoftext|>" wins over "<|".
        names = sorted(self.special_tokens.keys(), key=len, reverse=True)
        self._special_re = re.compile("|".join(re.escape(n) for n in names))

    # ----- helpers -----
    def _render(self, token_id: int) -> str:
        return self.vocab[token_id].decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    # ----- persistence -----
    def save(self, prefix: str) -> None:
        prefix = str(prefix)
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
        with open(prefix + ".merges", "w", encoding="utf-8") as f:
            f.write("# DuckGPT BPE merges — one (left, right) pair per line.\n")
            for a, b in self.merges:
                f.write(f"{a} {b}\n")
        meta = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
        }
        with open(prefix + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, prefix: str) -> None:
        prefix = str(prefix)
        with open(prefix + ".json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.special_tokens = meta.get("special_tokens", {}) or {}
        self.merges = []
        self.merge_index = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256
        with open(prefix + ".merges", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = (int(x) for x in line.split())
                self.merges.append((a, b))
                self.merge_index[(a, b)] = next_id
                self.vocab[next_id] = self.vocab[a] + self.vocab[b]
                next_id += 1
        for tok, _ in list(self.special_tokens.items()):
            self.special_tokens[tok] = next_id
            self.vocab[next_id] = tok.encode("utf-8")
            next_id += 1
        self._build_special_re()

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _read_corpus(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        chunks = []
        for sub in sorted(p.glob("**/*")):
            if sub.is_file() and sub.suffix.lower() in {".txt", ".md"}:
                chunks.append(sub.read_text(encoding="utf-8", errors="ignore"))
        return "\n".join(chunks)
    return p.read_text(encoding="utf-8", errors="ignore")

def _cli() -> None:
    parser = argparse.ArgumentParser(description="DuckGPT BPE tokenizer.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train BPE merges on a corpus.")
    p_train.add_argument("--input", required=True, help="Text file or directory.")
    p_train.add_argument("--vocab-size", type=int, default=8192)
    p_train.add_argument("--out", default="models/bpe")
    p_train.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    p_train.add_argument("--verbose", action="store_true")

    p_enc = sub.add_parser("encode", help="Encode a string and print ids.")
    p_enc.add_argument("--model", required=True)
    p_enc.add_argument("--text", required=True)

    p_dec = sub.add_parser("decode", help="Decode a comma-separated id list.")
    p_dec.add_argument("--model", required=True)
    p_dec.add_argument("--ids", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        tok = BPETokenizer(special_tokens={name: -1 for name in args.special})
        corpus = _read_corpus(args.input)
        if not corpus.strip():
            sys.exit(f"empty corpus at {args.input}")
        tok.train(corpus, vocab_size=args.vocab_size, verbose=args.verbose)
        tok.save(args.out)
        print(f"trained vocab_size={tok.vocab_size}, saved to {args.out}.merges/.json")
    elif args.cmd == "encode":
        tok = BPETokenizer()
        tok.load(args.model)
        ids = tok.encode(args.text)
        print(ids)
        print(f"({len(ids)} tokens)")
    elif args.cmd == "decode":
        tok = BPETokenizer()
        tok.load(args.model)
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
        print(tok.decode(ids))

if __name__ == "__main__":
    _cli()
