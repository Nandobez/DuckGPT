# Inference server

`lib/server.py` is a FastAPI app that exposes a trained checkpoint over HTTP +
WebSocket. It picks up the checkpoint to load from the `DUCKGPT_CKPT`
environment variable.

## Run directly

```bash
pip install fastapi 'uvicorn[standard]'
export DUCKGPT_CKPT=$PWD/out-shakespeare-char/ckpt.pt
export DUCKGPT_DEVICE=cpu          # or cuda / mps

PYTHONPATH=$PWD python -m uvicorn lib.server:app --host 0.0.0.0 --port 8000
```

## Run via Docker

```bash
docker compose -f deploy/docker-compose.yml up --build
```

The compose file bind-mounts `out-shakespeare-char/` to `/app/checkpoints/`
inside the container — adjust the path to whichever checkpoint you want to
serve.

## Endpoints

### `GET /health`

Returns metadata about the loaded model.

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

```json
{
  "status": "ok",
  "device": "cpu",
  "block_size": 128,
  "vocab_size": 100,
  "n_params": 816768
}
```

### `POST /generate`

Synchronous completion. Returns the prompt + the generated text plus timing.

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Era uma vez",
    "max_tokens": 200,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.15
  }' | python -m json.tool
```

Body fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | str | — | Required. |
| `max_tokens` | int | 200 | 1 ≤ max_tokens ≤ 4096 |
| `temperature` | float | 0.8 | ≥ 0.0 |
| `top_k` | int? | null | null disables. |
| `top_p` | float? | null | null disables. Combine with top_k for a tighter filter. |
| `repetition_penalty` | float | 1.0 | ≥ 1.0 |
| `stop` | list[str] | [] | Generation stops at the first match. |
| `seed` | int? | null | Reproducible sampling. |

### `WebSocket /stream`

Token-by-token streaming. Send the same JSON body and you'll receive one
message per token plus a final summary.

```bash
# requires `wscat` or similar
wscat -c ws://localhost:8000/stream
> {"prompt":"ROMEO:","max_tokens":80,"temperature":0.7,"top_p":0.9}
< {"id":34,"token":"R"}
< {"id":110,"token":"O"}
…
< {"done":true,"text":"ROMEO:…","tokens":80,"time_ms":210}
```

## Python client

```python
import requests
r = requests.post("http://localhost:8000/generate", json={
    "prompt": "Era uma vez",
    "max_tokens": 200,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.15,
})
print(r.json()["text"])
```

For streaming:

```python
import asyncio, websockets, json

async def stream():
    async with websockets.connect("ws://localhost:8000/stream") as ws:
        await ws.send(json.dumps({"prompt":"Era uma vez","max_tokens":50}))
        async for msg in ws:
            data = json.loads(msg)
            if data.get("done"):
                print("\n[done]", data["tokens"], "tokens")
                break
            print(data["token"], end="", flush=True)

asyncio.run(stream())
```

## Environment variables

| Variable | Required | Notes |
|---|---|---|
| `DUCKGPT_CKPT` | yes | Path to the `.pt` file. |
| `DUCKGPT_DEVICE` | no | `cuda`, `mps`, or `cpu`. Auto-detected if unset. |

## Production notes

- The model is loaded once at startup via FastAPI's `lifespan` — there's no
  hot-reload of weights. Restart the server to switch checkpoints.
- Token generation is **single-threaded** through the model. For higher
  throughput on GPU, run multiple uvicorn workers or use NVIDIA Triton.
- The server enables CORS for `*` so a local UI can hit it during development;
  restrict this in production.
