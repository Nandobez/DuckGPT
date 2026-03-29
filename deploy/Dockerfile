FROM python:3.12-slim

WORKDIR /app

# System dependencies (only what torch + numpy actually need)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage the build cache.
COPY deploy/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
       "torch==2.*" \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy the project (we deliberately leave training data + datasets out — bind-mount them).
COPY README.md LICENSE /app/
COPY lib /app/lib
COPY models /app/models
COPY config /app/config

ENV PYTHONPATH=/app
ENV DUCKGPT_CKPT=/app/checkpoints/ckpt.pt
ENV DUCKGPT_DEVICE=cpu

EXPOSE 8000
CMD ["uvicorn", "lib.server:app", "--host", "0.0.0.0", "--port", "8000"]
