"""
DuckGPT — FastAPI inference server.

Endpoints
=========

GET  /health
    Returns model metadata: device, vocab_size, block_size, parameter count.

POST /generate
    JSON body:
        {
          "prompt": "...",
          "max_tokens": 200,
          "temperature": 0.8,
          "top_k": 40,
          "top_p": 0.9,
          "repetition_penalty": 1.1,
          "stop": ["</s>"]
        }
    Returns:
        {
          "text": "...",
          "tokens": <int>,
          "time_ms": <int>
        }

WebSocket /stream
    Send the same JSON. Receives one JSON message per token::

        {"token": "...", "id": 123}

    and a final ``{"done": true, "text": "...", "tokens": N, "time_ms": M}``.

Run with
========

.. code:: bash

    pip install fastapi uvicorn
    python -m uvicorn lib.server:app --host 0.0.0.0 --port 8000

or via the Docker setup in ``deploy/`` (see ``deploy/Dockerfile``).
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from model import GPT, GPTConfig
from sample import (
    stream_generate, _load_checkpoint, _resolve_codec, _auto_device,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to continue.")
    max_tokens: int = Field(200, ge=1, le=4096)
    temperature: float = Field(0.8, ge=0.0)
    top_k: Optional[int] = Field(None, ge=0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=1.0)
    stop: list[str] = Field(default_factory=list)
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    text: str
    tokens: int
    time_ms: int


# ---------------------------------------------------------------------------
# Lifespan: load the model once at startup
# ---------------------------------------------------------------------------

class State:
    model: GPT
    encode: callable
    decode_one: callable
    device: str
    block_size: int
    vocab_size: int
    n_params: int


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = os.environ.get("DUCKGPT_CKPT")
    if not ckpt:
        raise RuntimeError(
            "Set DUCKGPT_CKPT=/path/to/ckpt.pt before starting the server."
        )
    device = os.environ.get("DUCKGPT_DEVICE", _auto_device())
    model, ck = _load_checkpoint(ckpt, device)
    encode, decode_one = _resolve_codec(ck)
    state.model = model
    state.encode = encode
    state.decode_one = decode_one
    state.device = device
    state.block_size = model.config.block_size
    state.vocab_size = model.config.vocab_size
    state.n_params = sum(p.numel() for p in model.parameters())
    print(f"[duckgpt] loaded {ckpt} on {device}  ({state.n_params:,} params)")
    yield


app = FastAPI(title="DuckGPT", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": state.device,
        "block_size": state.block_size,
        "vocab_size": state.vocab_size,
        "n_params": state.n_params,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if req.seed is not None:
        torch.manual_seed(req.seed)
    ids0 = torch.tensor([state.encode(req.prompt) or [0]],
                        dtype=torch.long, device=state.device)
    t0 = time.perf_counter()
    out_tokens = []
    for tok, _ in stream_generate(
        state.model, ids0, state.decode_one,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        stop=req.stop,
    ):
        out_tokens.append(tok)
    dt = (time.perf_counter() - t0) * 1000
    text = req.prompt + "".join(state.decode_one(i) for i in out_tokens)
    return GenerateResponse(text=text, tokens=len(out_tokens), time_ms=int(dt))


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        req_data = await ws.receive_json()
        req = GenerateRequest(**req_data)
        if req.seed is not None:
            torch.manual_seed(req.seed)
        ids0 = torch.tensor([state.encode(req.prompt) or [0]],
                            dtype=torch.long, device=state.device)
        t0 = time.perf_counter()
        out_text = ""
        n = 0
        for tok, ch in stream_generate(
            state.model, ids0, state.decode_one,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            stop=req.stop,
        ):
            n += 1
            out_text += ch
            await ws.send_json({"id": int(tok), "token": ch})
        await ws.send_json({"done": True, "text": req.prompt + out_text,
                            "tokens": n, "time_ms": int((time.perf_counter() - t0) * 1000)})
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_json({"error": str(e)})
    finally:
        await ws.close()
