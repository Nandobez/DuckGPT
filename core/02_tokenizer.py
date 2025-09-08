"""DuckGPT — core lesson 02: byte-level BPE, the short version.

This is a *teaching* tokenizer. It does the same thing as ``lib/tokenizer.py``
but with:

- pure Python (no numpy vectorisation),
- a tiny corpus,
- a print statement at every merge so you can see how the vocabulary grows.

Byte-pair encoding (Sennrich et al., 2016):

1. Start with every byte as its own token (256 tokens).
2. Count adjacent token pairs across the corpus.
3. Merge the most-frequent pair into a *new* token id (next free integer).
4. Repeat ``num_merges`` times.

Encoding new text: pre-tokenise, then keep applying merges in the order they
were learned, picking the earliest-learned pair available each step.

Run this file directly to watch a 12-merge example train on a tiny PT/EN
corpus and round-trip the result back to text.
"""
from collections import Counter

# 1. tiny multilingual corpus -----------------------------------------------
CORPUS = (
    "olá mundo. olá amigos. o mundo é grande. "
    "hello world. hello friends. the world is big. "
) * 10

VOCAB_SIZE = 268   # 256 bytes + 12 merges

# 2. initial state ----------------------------------------------------------
# Treat each chunk as a list of byte values (0..255).
ids = list(CORPUS.encode("utf-8"))
print(f"corpus: {len(ids)} bytes")
print(f"unique starting tokens: {len(set(ids))}\n")

# We keep a flat token list here, but `lib/tokenizer.py` splits the corpus
# into pre-tokenised chunks first so merges can't cross word boundaries.

# 3. learn merges -----------------------------------------------------------
vocab = {i: bytes([i]) for i in range(256)}
merges: list[tuple[int, int]] = []

def most_common_pair(ids):
    return Counter(zip(ids, ids[1:])).most_common(1)[0]

def merge(ids, pair, new_id):
    out = []
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

num_merges = VOCAB_SIZE - 256
for step in range(num_merges):
    pair, freq = most_common_pair(ids)
    new_id = 256 + step
    ids = merge(ids, pair, new_id)
    merges.append(pair)
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    left = vocab[pair[0]].decode("utf-8", errors="replace")
    right = vocab[pair[1]].decode("utf-8", errors="replace")
    print(f"merge {step:>2}: '{left}' + '{right}'  -> id {new_id}  (freq={freq})")

print(f"\nfinal vocab size: {len(vocab)}  (256 bytes + {num_merges} merges)\n")

# 4. encode / decode --------------------------------------------------------
def encode(text):
    out = list(text.encode("utf-8"))
    while len(out) >= 2:
        # find earliest-learned merge applicable
        best_pair = None
        best_rank = None
        for i, pair in enumerate(zip(out, out[1:])):
            if pair in merges:
                rank = merges.index(pair)
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair
        if best_pair is None:
            break
        out = merge(out, best_pair, 256 + best_rank)
    return out

def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

sample = "olá amigos do mundo"
ids_enc = encode(sample)
print(f"encode {sample!r} -> {ids_enc}   (len {len(ids_enc)})")
print(f"decode  -> {decode(ids_enc)!r}")
assert decode(ids_enc) == sample, "roundtrip failed!"
print("roundtrip OK\n")

print("Next file: 03_attention.py — how a single attention head works.")
