"""
AMARA - RAG Pipeline
Builds and queries a FAISS vector store from academic paper abstracts.
Grounds agent outputs in retrievable, verifiable chunks.
"""
from __future__ import annotations
import os
import pickle
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import VECTOR_STORE_PATH, MAX_CHUNKS


class RAGPipeline:
    """
    Simple FAISS-backed RAG pipeline using sentence-transformers embeddings.
    Falls back gracefully if FAISS is not installed.
    """

    EMBED_MODEL = "all-MiniLM-L6-v2"

    def __init__(self):
        self.embedder   = SentenceTransformer(self.EMBED_MODEL)
        self.chunks: List[Dict]   = []
        self.embeddings: np.ndarray | None = None
        self.index      = None
        self._faiss_ok  = False
        self._try_load_faiss()

    # ── FAISS Setup ───────────────────────────────────────────────────────────
    def _try_load_faiss(self):
        try:
            import faiss  # noqa: F401
            self._faiss_ok = True
        except ImportError:
            print("[RAG] FAISS not available – using numpy cosine similarity fallback.")

    def _build_faiss_index(self, embeddings: np.ndarray):
        import faiss
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype("float32"))
        return index

    # ── Chunking ──────────────────────────────────────────────────────────────
    def _chunk_paper(self, paper: Dict, chunk_size: int = 300) -> List[Dict]:
        """Split a paper's abstract into overlapping chunks."""
        text   = paper.get("abstract", "") or paper.get("summary", "")
        words  = text.split()
        chunks = []
        step   = chunk_size // 2  # 50% overlap

        for i in range(0, max(1, len(words) - chunk_size + 1), step):
            chunk_text = " ".join(words[i : i + chunk_size])
            chunks.append({
                "text":    chunk_text,
                "title":   paper["title"],
                "authors": paper["authors"],
                "year":    paper["year"],
                "url":     paper["url"],
                "source":  paper["source"],
            })
            if len(chunks) >= 3:  # max 3 chunks per paper
                break

        if not chunks:  # empty abstract fallback
            chunks.append({
                "text":    paper["title"],
                "title":   paper["title"],
                "authors": paper["authors"],
                "year":    paper["year"],
                "url":     paper["url"],
                "source":  paper["source"],
            })
        return chunks

    # ── Index Building ────────────────────────────────────────────────────────
    def build_index(self, papers: List[Dict]) -> None:
        """Embed all paper chunks and build the vector index."""
        self.chunks = []
        for paper in papers:
            self.chunks.extend(self._chunk_paper(paper))

        texts            = [c["text"] for c in self.chunks]
        self.embeddings  = self.embedder.encode(texts, show_progress_bar=False)

        if self._faiss_ok:
            self.index = self._build_faiss_index(self.embeddings)

        # Persist to disk
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        with open(VECTOR_STORE_PATH + ".pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

        print(f"[RAG] Index built: {len(self.chunks)} chunks from {len(papers)} papers.")

    def load_index(self) -> bool:
        """Load a previously saved index from disk."""
        path = VECTOR_STORE_PATH + ".pkl"
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks     = data["chunks"]
        self.embeddings = data["embeddings"]
        if self._faiss_ok:
            self.index = self._build_faiss_index(self.embeddings)
        print(f"[RAG] Loaded index: {len(self.chunks)} chunks.")
        return True

    # ── Query ─────────────────────────────────────────────────────────────────
    def query(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve the most relevant chunks for a query."""
        top_k = top_k or MAX_CHUNKS
        if not self.chunks:
            return []

        q_emb = self.embedder.encode([query])

        if self._faiss_ok and self.index is not None:
            import faiss  # noqa: F401
            _, indices = self.index.search(q_emb.astype("float32"), top_k)
            return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        else:
            # Cosine similarity fallback
            q_norm    = q_emb / (np.linalg.norm(q_emb) + 1e-8)
            emb_norm  = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
            scores    = (emb_norm @ q_norm.T).squeeze()
            top_idx   = np.argsort(scores)[::-1][:top_k]
            return [self.chunks[i] for i in top_idx]

    # ── Convenience ───────────────────────────────────────────────────────────
    def get_context_string(self, query: str, top_k: int = None) -> str:
        """Return retrieved chunks as a formatted string for LLM context."""
        chunks = self.query(query, top_k)
        lines  = []
        for i, c in enumerate(chunks, 1):
            lines.append(
                f"[Source {i}] {c['title']} ({', '.join(c['authors'])}, {c['year']})\n{c['text']}"
            )
        return "\n\n".join(lines)
