import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.set_num_threads(1)

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

MODEL_NAME = os.getenv("MODEL_NAME", "paraphrase-MiniLM-L3-v2")
model = None

class EmbedRequest(BaseModel):
    input: str

@app.on_event("startup")
def load_model():
    global model
    model = SentenceTransformer(MODEL_NAME)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
def embed(req: EmbedRequest):
    vector = model.encode(req.input, normalize_embeddings=True)
    return {"embedding": vector.tolist()}
