# core_tools/embedding_engine.py
"""
Embedding Engine:
- Preferred local sentence-transformers for embeddings (fast offline)
- Fallback to OpenAI embeddings if OPENAI_API_KEY is available and local model not present
"""

from typing import List
import os
import numpy as np
from config.settings import EMBEDDING_BACKEND, EMBEDDING_MODEL_OPENAI, LOCAL_EMBEDDING_MODEL, OPENAI_API_KEY
try:
    from sentence_transformers import SentenceTransformer
    _has_sbert = True
except Exception:
    _has_sbert = False

# If openai present
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    _openai_client = None

class EmbeddingEngine:
    def __init__(self, backend: str = "auto"):
        self.backend = backend or EMBEDDING_BACKEND or "auto"
        if self.backend == "auto":
            self.backend = "local" if _has_sbert else ("openai" if _openai_client else "local")

        if self.backend == "local":
            if not _has_sbert:
                raise RuntimeError("Local SBERT model requested but sentence-transformers not available.")
            self.model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        else:
            if not _openai_client:
                raise RuntimeError("OpenAI client not available for embeddings.")
            self.model = None  # use openai client

    def encode(self, texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        if self.backend == "local":
            emb = self.model.encode(texts, convert_to_numpy=True)
            return emb.tolist()
        else:
            # OpenAI embeddings
            resp = _openai_client.embeddings.create(model=EMBEDDING_MODEL_OPENAI, input=texts)
            return [d.embedding for d in resp.data]
