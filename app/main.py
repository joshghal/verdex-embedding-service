from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os

app = FastAPI(title="Verdex Embedding Service")

MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

model = SentenceTransformer(MODEL_NAME)

class EmbedRequest(BaseModel):
    input: str

class EmbedResponse(BaseModel):
    embedding: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed", response_model=EmbedResponse)
def embed(
    req: EmbedRequest,
):
    vector = model.encode(
        req.input,
        normalize_embeddings=True
    )

    return {"embedding": vector.tolist()}
