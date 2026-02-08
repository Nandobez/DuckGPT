"""Shared helpers for DuckGPT eval scripts."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Callable, Tuple

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.append(str(HERE.parent))   # so we can `from model import GPT`

from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def auto_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_checkpoint(ckpt_path: str, device: str = None) -> Tuple[GPT, dict]:
    device = device or auto_device()
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = obj.get('model_args') or obj.get('config')
    if isinstance(cfg, dict):
        cfg = GPTConfig(**{k: v for k, v in cfg.items()
                           if k in GPTConfig.__dataclass_fields__})
    model = GPT(cfg)
    sd = obj['model']
    for k in list(sd.keys()):
        if k.startswith('_orig_mod.'):
            sd[k[len('_orig_mod.'):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, obj


def resolve_codec(ck: dict) -> Tuple[Callable[[str], list], Callable[[list], str]]:
    dataset = (ck.get('config') or {}).get('dataset')
    if dataset:
        meta_path = ROOT.parent / 'data' / dataset / 'meta.pkl'
        if meta_path.exists():
            meta = pickle.load(open(meta_path, 'rb'))
            stoi, itos = meta['stoi'], meta['itos']
            return ((lambda s: [stoi[c] for c in s if c in stoi]),
                    (lambda L: ''.join(itos[i] for i in L)))
    prefix = ROOT.parent / 'models' / 'bpe'
    if Path(str(prefix) + '.merges').exists():
        tok = BPETokenizer()
        tok.load(str(prefix))
        return (lambda s: tok.encode(s)), (lambda L: tok.decode(L))
    raise SystemExit("No tokenizer found (no meta.pkl, no models/bpe).")


@torch.no_grad()
def sequence_logprob(model: GPT, ids: list[int], device: str) -> float:
    """Sum of log-probabilities the model assigns to ``ids[1:]`` conditioned
    on the preceding tokens. Used for likelihood-based multiple-choice eval.
    """
    if len(ids) < 2:
        return 0.0
    block = model.config.block_size
    ids_t = torch.tensor([ids[:block]], dtype=torch.long, device=device)
    logits, _ = model(ids_t)
    log_probs = torch.log_softmax(logits, dim=-1)[0]            # (T, V)
    target = ids_t[0, 1:]                                       # (T-1,)
    selected = log_probs[:-1].gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(selected.sum().item())
