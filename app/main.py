import os

# Thread limits for Railway's small containers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
import numpy as np
from pathlib import Path
from typing import List, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Verdex Embedding Service",
    description="Ultra-light ONNX embedding service for Railway free tier",
    version="1.0.0"
)

# Use all-MiniLM-L6-v2 ONNX - good quality, small size
MODEL_ID = os.getenv("MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_DIR = Path("/tmp/model")

session: ort.InferenceSession = None
tokenizer: Tokenizer = None


class EmbedRequest(BaseModel):
    input: Union[str, List[str]]


class EmbedResponse(BaseModel):
    embedding: Union[List[float], List[List[float]]]


def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean pooling - take mean of token embeddings weighted by attention mask"""
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
    sum_embeddings = np.sum(model_output * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-9, a_max=None)


def download_model():
    """Download ONNX model and tokenizer from HuggingFace"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Download ONNX model
    onnx_path = MODEL_DIR / "model.onnx"
    if not onnx_path.exists():
        logger.info(f"Downloading ONNX model from {MODEL_ID}...")
        downloaded = hf_hub_download(
            repo_id=MODEL_ID,
            filename="onnx/model.onnx",
            local_dir=MODEL_DIR
        )
        # Move to expected location
        import shutil
        shutil.move(downloaded, onnx_path)

    # Download tokenizer
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    if not tokenizer_path.exists():
        logger.info("Downloading tokenizer...")
        downloaded = hf_hub_download(
            repo_id=MODEL_ID,
            filename="tokenizer.json",
            local_dir=MODEL_DIR
        )
        if Path(downloaded) != tokenizer_path:
            import shutil
            shutil.move(downloaded, tokenizer_path)

    return onnx_path, tokenizer_path


@app.on_event("startup")
async def load_model():
    """Load ONNX model at startup"""
    global session, tokenizer

    try:
        onnx_path, tokenizer_path = download_model()

        # Load ONNX session with optimizations
        logger.info("Loading ONNX model...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        session = ort.InferenceSession(
            str(onnx_path),
            sess_options,
            providers=["CPUExecutionProvider"]
        )

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        tokenizer.enable_truncation(max_length=512)

        logger.info("Model loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health")
async def health():
    """Health check for Railway"""
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": MODEL_ID, "runtime": "onnx"}


@app.get("/")
async def root():
    return {
        "service": "Verdex Embedding Service",
        "model": MODEL_ID,
        "runtime": "onnx",
        "endpoints": ["/health", "/embed"]
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    """Generate embeddings using ONNX runtime"""
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Handle single or batch input
        is_single = isinstance(req.input, str)
        texts = [req.input] if is_single else req.input

        # Tokenize
        encoded = tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        # Run inference
        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
        )

        # Pool and normalize
        embeddings = mean_pooling(outputs[0], attention_mask)
        embeddings = normalize(embeddings)

        if is_single:
            return {"embedding": embeddings[0].tolist()}
        return {"embedding": [e.tolist() for e in embeddings]}

    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
