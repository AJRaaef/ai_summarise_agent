# agents/semantic_agent.py
"""
Semantic Agent: uses embeddings to map column names and sample values
to common airline concepts (revenue, passengers, delay, cancelled, route, flightno, date, time).
"""

from core_tools.embedding_engine import EmbeddingEngine
from typing import Dict, Any
import numpy as np

# Known concept labels used to match against (can be extended)
CONCEPTS = [
    "revenue", "passengers", "delay", "cancel", "route", "flight number", "date", "time",
    "origin", "destination", "seat_capacity", "load_factor", "fare"
]

class SemanticAgent:
    def __init__(self, backend: str = "auto"):
        self.engine = EmbeddingEngine(backend)

        # precompute concept embeddings
        self.concept_emb = self.engine.encode(CONCEPTS)

    def infer_columns(self, df) -> Dict[str, Any]:
        cols = list(df.columns)
        col_map = {}
        col_samples = {}
        col_embs = self.engine.encode(cols)

        for i, c in enumerate(cols):
            # compute cosine similarity with concept embeddings
            v = np.array(col_embs[i], dtype=float)
            sims = []
            for j, ce in enumerate(self.concept_emb):
                ce_v = np.array(ce, dtype=float)
                sim = float(np.dot(v, ce_v) / (np.linalg.norm(v) * np.linalg.norm(ce_v) + 1e-10))
                sims.append((CONCEPTS[j], sim))
            sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
            col_map[c] = {"best_match": sims_sorted[0][0], "score": sims_sorted[0][1],
                          "top_matches": sims_sorted[:3]}
            # sample value type hint
            sample_vals = df[c].dropna().astype(str).head(5).tolist()
            col_samples[c] = sample_vals
        return {"mapping": col_map, "samples": col_samples}
