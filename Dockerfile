# Ultra-light Dockerfile for Railway Free Tier
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/app/.cache/huggingface

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Xenova ONNX model during build
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('Xenova/all-MiniLM-L6-v2', 'onnx/model.onnx'); \
    hf_hub_download('Xenova/all-MiniLM-L6-v2', 'tokenizer.json')"

# Copy app
COPY app ./app

# Railway uses PORT env var - don't hardcode
EXPOSE 8000

# Use shell form to expand $PORT at runtime
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
