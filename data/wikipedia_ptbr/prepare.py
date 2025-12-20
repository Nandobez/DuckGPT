"""
DuckGPT — Wikipedia PT-BR dataset preparation.

Downloads the latest Brazilian-Portuguese Wikipedia dump from Hugging Face,
tokenises every article with GPT-2 BPE (works fine on UTF-8 PT text), and
writes train.bin / val.bin (uint16) so training reads from disk via mmap.

Usage:
    python data/wikipedia_ptbr/prepare.py [--max-articles N]
"""
import argparse
import os
from pathlib import Path

import sys
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[1]))
from tokenizer import BPETokenizer  # noqa: E402
TOK_PREFIX = HERE.parents[1] / "models" / "bpe"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Cap the number of articles (useful for smoke tests).")
    parser.add_argument("--val-fraction", type=float, default=0.0005)
    parser.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = parser.parse_args()

    print("Loading wikimedia/wikipedia (20231101.pt)…")
    ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train")
    if args.max_articles:
        ds = ds.select(range(min(args.max_articles, len(ds))))
    print(f"  articles: {len(ds):,}")

    split = ds.train_test_split(test_size=args.val_fraction, seed=2357, shuffle=True)
    split["val"] = split.pop("test")

    if not Path(str(TOK_PREFIX) + ".merges").exists():
        sys.exit(
            f"No tokenizer at {TOK_PREFIX}.merges — train one first with\n"
            f"  python tokenizer.py train --input data/literatura_ptbr/raw "
            f"--vocab-size 16000 --out models/bpe"
        )
    enc = BPETokenizer()
    enc.load(str(TOK_PREFIX))
    eot = enc.special_tokens.get("<|endoftext|>") or enc.add_special("<|endoftext|>")

    def tokenise(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(eot)  # delimits articles
        return {"ids": ids, "len": len(ids)}

    tokenised = split.map(tokenise, remove_columns=["text"], num_proc=args.num_proc,
                          desc="tokenising")

    for name, dset in tokenised.items():
        total = int(np.sum(dset["len"], dtype=np.uint64))
        out = HERE / f"{name}.bin"
        print(f"  → {out}  ({total:,} tokens)")
        arr = np.memmap(out, dtype=np.uint16, mode="w+", shape=(total,))
        idx = 0
        # write in batches to keep memory bounded
        for shard_id in tqdm(range(1024), desc=f"writing {name}"):
            batch = dset.shard(num_shards=1024, index=shard_id, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    print("Done.")

if __name__ == "__main__":
    main()
