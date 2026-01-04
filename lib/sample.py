"""
DuckGPT — production-grade sampling.

Features added on top of the basic loop in ``core/08_sample.py``:

- ``--top_k``    : restrict to the top-k most likely tokens at each step.
- ``--top_p``    : nucleus sampling (Holtzman et al., 2020) — keep the smallest
                   set whose cumulative probability is ``top_p``.
- ``--repetition_penalty`` : divide logits of previously-generated tokens by
                   the given factor (Keskar et al., CTRL 2019).
- ``--stop``     : one or more "stop strings" — generation ends as soon as any
                   of them appears in the decoded output.
- ``--stream``   : print tokens as they're generated, character by character.
- ``--beam_size``: beam search instead of stochastic sampling.

Example:

.. code:: bash

    python lib/sample.py --out_dir out-shakespeare-char \\
        --start "ROMEO:" --max_new_tokens 200 \\
        --top_p 0.9 --repetition_penalty 1.2 --stream
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(HERE))
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


# ---------------------------------------------------------------------------
# Logit shaping
# ---------------------------------------------------------------------------

def apply_repetition_penalty(logits: torch.Tensor,
                             generated: torch.Tensor,
                             penalty: float) -> torch.Tensor:
    """Divide logits of already-generated token ids by ``penalty`` (>1)."""
    if penalty == 1.0:
        return logits
    for tok in set(generated.tolist()):
        if logits[0, tok] > 0:
            logits[0, tok] = logits[0, tok] / penalty
        else:
            logits[0, tok] = logits[0, tok] * penalty
    return logits


def filter_top_k(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    if not k:
        return logits
    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    return logits.masked_fill(logits < v[:, [-1]], float('-inf'))


def filter_top_p(logits: torch.Tensor, p: float | None) -> torch.Tensor:
    """Keep the smallest set of tokens with cumulative prob ≥ p (nucleus)."""
    if not p or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumulative > p
    # shift right so we always keep at least one token
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits.masked_fill_(mask, float('-inf'))
    out = torch.full_like(logits, float('-inf'))
    out.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return out


# ---------------------------------------------------------------------------
# Generation primitives
# ---------------------------------------------------------------------------

def stream_generate(model: GPT,
                    ids: torch.Tensor,
                    decode_one: Callable[[int], str],
                    max_new_tokens: int,
                    temperature: float,
                    top_k: int | None,
                    top_p: float | None,
                    repetition_penalty: float,
                    stop: list[str]) -> Iterator[tuple[int, str]]:
    """Yield (token_id, decoded_str) one step at a time. Stops on first stop string."""
    buffer = ""
    block = model.config.block_size
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = ids if ids.size(1) <= block else ids[:, -block:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            logits = apply_repetition_penalty(logits, ids[0], repetition_penalty)
            logits = filter_top_k(logits, top_k)
            logits = filter_top_p(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, nxt], dim=1)
            ch = decode_one(int(nxt.item()))
            buffer += ch
            yield int(nxt.item()), ch
            if stop and any(s in buffer for s in stop):
                return


def beam_search(model: GPT,
                ids: torch.Tensor,
                max_new_tokens: int,
                beam_size: int,
                length_penalty: float = 1.0,
                stop_ids: set[int] | None = None) -> tuple[list[int], float]:
    """Return (token_ids, log_prob) of the best beam after `max_new_tokens` steps."""
    block = model.config.block_size
    beams: list[tuple[list[int], float]] = [(ids[0].tolist(), 0.0)]
    finished: list[tuple[list[int], float]] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            candidates: list[tuple[list[int], float]] = []
            for seq, score in beams:
                t = torch.tensor([seq[-block:]], device=ids.device, dtype=torch.long)
                logits, _ = model(t)
                logp = F.log_softmax(logits[0, -1], dim=-1)
                topv, topi = torch.topk(logp, beam_size)
                for v, i in zip(topv.tolist(), topi.tolist()):
                    new_seq = seq + [i]
                    new_score = score + v
                    if stop_ids and i in stop_ids:
                        finished.append((new_seq, new_score))
                    else:
                        candidates.append((new_seq, new_score))
            if not candidates:
                break
            candidates.sort(key=lambda sb: sb[1] / (len(sb[0]) ** length_penalty),
                            reverse=True)
            beams = candidates[:beam_size]
        finished.extend(beams)
    finished.sort(key=lambda sb: sb[1] / (len(sb[0]) ** length_penalty), reverse=True)
    return finished[0]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _load_checkpoint(ckpt: str, device: str):
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = obj.get('model_args') or obj.get('config')
    if isinstance(cfg, dict):
        cfg = GPTConfig(**{k: v for k, v in cfg.items()
                           if k in GPTConfig.__dataclass_fields__})
    model = GPT(cfg)
    state = obj['model']
    for k in list(state.keys()):
        if k.startswith('_orig_mod.'):
            state[k[len('_orig_mod.'):]] = state.pop(k)
    model.load_state_dict(state)
    model.eval().to(device)
    return model, obj


def _resolve_codec(ck: dict):
    """Pick the right encode / decode pair for the trained checkpoint."""
    dataset = (ck.get('config') or {}).get('dataset')
    if dataset:
        meta_path = ROOT / 'data' / dataset / 'meta.pkl'
        if meta_path.exists():
            meta = pickle.load(open(meta_path, 'rb'))
            stoi, itos = meta['stoi'], meta['itos']
            return ((lambda s: [stoi[c] for c in s if c in stoi]),
                    (lambda i: itos[i]))
    prefix = ROOT / 'models' / 'bpe'
    if Path(str(prefix) + '.merges').exists():
        tok = BPETokenizer()
        tok.load(str(prefix))
        return (lambda s: tok.encode(s)), (lambda i: tok.decode([i]))
    raise SystemExit("No tokenizer found (no meta.pkl, no models/bpe).")


def main():
    p = argparse.ArgumentParser(description="DuckGPT sampling CLI")
    p.add_argument('--out_dir', default='out')
    p.add_argument('--ckpt', default=None)
    p.add_argument('--start', default='\n')
    p.add_argument('--max_new_tokens', type=int, default=200)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=0)
    p.add_argument('--top_p', type=float, default=0.0)
    p.add_argument('--repetition_penalty', type=float, default=1.0)
    p.add_argument('--stop', nargs='*', default=[])
    p.add_argument('--stream', action='store_true')
    p.add_argument('--beam_size', type=int, default=0)
    p.add_argument('--num_samples', type=int, default=1)
    p.add_argument('--device', default=_auto_device())
    p.add_argument('--seed', type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ckpt_path = args.ckpt or os.path.join(args.out_dir, 'ckpt.pt')
    print(f"loading {ckpt_path} on {args.device}…")
    model, ck = _load_checkpoint(ckpt_path, args.device)
    encode, decode_one = _resolve_codec(ck)

    start_text = args.start
    if start_text.startswith('FILE:'):
        start_text = Path(start_text[5:]).read_text(encoding='utf-8')
    start_ids = encode(start_text) or [0]
    ids0 = torch.tensor([start_ids], dtype=torch.long, device=args.device)

    for _ in range(args.num_samples):
        print("─" * 60)
        if args.beam_size:
            seq, score = beam_search(model, ids0, args.max_new_tokens, args.beam_size)
            print("".join(decode_one(i) for i in seq))
            print(f"\n[beam log-prob: {score:.3f}]")
            continue

        t0 = time.perf_counter()
        if args.stream:
            print(start_text, end='', flush=True)
        generated: list[int] = []
        for tok, ch in stream_generate(
            model, ids0.clone(), decode_one,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k or None,
            top_p=args.top_p or None,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop,
        ):
            generated.append(tok)
            if args.stream:
                print(ch, end='', flush=True)
        if not args.stream:
            print(start_text + "".join(decode_one(i) for i in generated))
        dt = time.perf_counter() - t0
        print(f"\n[{len(generated)} tokens in {dt:.2f}s — {len(generated)/max(dt,1e-9):.1f} tok/s]")
    print("─" * 60)


if __name__ == '__main__':
    main()
