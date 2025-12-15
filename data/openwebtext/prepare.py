"""
DuckGPT — OpenWebText preparation.

Pulls OpenWebText via the Hugging Face ``datasets`` loader (the corpus itself
is external; the *encoding* is done with the in-house DuckGPT BPE tokenizer).
"""
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from tokenizer import BPETokenizer  # noqa: E402

HERE = Path(__file__).resolve().parent
TOK_PREFIX = Path(__file__).resolve().parents[2] / "models" / "bpe"
NUM_PROC = max(1, (os.cpu_count() or 4) // 2)

def main():
    if not Path(str(TOK_PREFIX) + ".merges").exists():
        sys.exit(
            f"No tokenizer at {TOK_PREFIX}.merges — train one first with\n"
            f"  python tokenizer.py train --input data/shakespeare/input.txt "
            f"--vocab-size 32000 --out models/bpe"
        )
    tok = BPETokenizer()
    tok.load(str(TOK_PREFIX))
    eot = tok.special_tokens.get("<|endoftext|>")
    if eot is None:
        eot = tok.add_special("<|endoftext|>")
        tok.save(str(TOK_PREFIX))

    print("Loading OpenWebText…")
    ds = load_dataset("openwebtext", num_proc=NUM_PROC)
    splits = ds["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    splits["val"] = splits.pop("test")

    def encode(example):
        ids = tok.encode_ordinary(example["text"])
        ids.append(eot)
        return {"ids": ids, "len": len(ids)}

    enc = splits.map(encode, remove_columns=["text"], num_proc=NUM_PROC,
                     desc="encoding")

    for name, dset in enc.items():
        total = int(np.sum(dset["len"], dtype=np.uint64))
        out = HERE / f"{name}.bin"
        print(f"  → {out}  ({total:,} tokens)")
        arr = np.memmap(out, dtype=np.uint16, mode="w+", shape=(total,))
        idx = 0
        for shard_id in tqdm(range(1024), desc=f"writing {name}"):
            shard = dset.shard(num_shards=1024, index=shard_id, contiguous=True).with_format("numpy")
            chunk = np.concatenate(shard["ids"])
            arr[idx : idx + len(chunk)] = chunk
            idx += len(chunk)
        arr.flush()

if __name__ == "__main__":
    main()
