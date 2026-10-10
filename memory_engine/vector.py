from __future__ import annotations

import hashlib
import math
import os
from typing import Optional

import requests

from .store import tokenize


class EmbeddingProvider:
    """
    优先使用 Ollama（可配置 OLLAMA_EMBEDDING_MODEL）。
    Ollama 不可用时使用稳定 hash embedding，保证本地测试不依赖外部服务。
    """

    def __init__(self, dim=384):
        self.model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:4b")
        self.url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/embeddings")
        self.dim = dim
        self.use_ollama = os.getenv("USE_OLLAMA_EMBEDDING", "1") == "1"

    def embed(self, text: str) -> list[float]:
        """Embed text safely even when a memory is longer than the model context.

        Long inputs are split into bounded overlapping windows. Duplicate windows
        are embedded once and weighted by their frequency, which keeps repetitive
        inputs cheap without discarding their contribution to the final vector.
        """
        text = str(text or "")
        max_chars = max(512, int(os.getenv("OLLAMA_EMBED_MAX_CHARS", "6000")))
        overlap = min(max_chars // 4, max(0, int(os.getenv("OLLAMA_EMBED_OVERLAP_CHARS", "200"))))

        if len(text) <= max_chars:
            return self._embed_one(text)

        chunks = self._split_text(text, max_chars=max_chars, overlap=overlap)
        counts: dict[str, int] = {}
        for chunk in chunks:
            counts[chunk] = counts.get(chunk, 0) + 1

        vectors = [(self._embed_one(chunk), count) for chunk, count in counts.items()]
        if not vectors:
            return self._hash_embed(text)

        # Embedding dimensions should be stable for a given model. Use the
        # shared prefix defensively in case a backend returns inconsistent sizes.
        dim = min(len(vector) for vector, _ in vectors)
        if dim == 0:
            return self._hash_embed(text)
        total_weight = sum(count for _, count in vectors)
        pooled = [
            sum(vector[i] * count for vector, count in vectors) / total_weight
            for i in range(dim)
        ]
        return self._normalize(pooled)

    @staticmethod
    def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
        """Split on character boundaries, preferring whitespace when available."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + max_chars * 2 // 3, end)
                if boundary > start:
                    end = boundary
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
        return chunks

    def _embed_one(self, text: str) -> list[float]:
        if self.use_ollama:
            try:
                r = requests.post(
                    self.url,
                    json={"model": self.model, "prompt": text},
                    timeout=15,
                )
                r.raise_for_status()
                vec = r.json().get("embedding")
                if vec:
                    return self._normalize([float(x) for x in vec])
            except Exception:
                # A failed Ollama request falls back per chunk. This prevents one
                # oversized document from breaking the complete Add operation.
                pass
        return self._hash_embed(text)

    def _hash_embed(self, text: str):
        v = [0.0] * self.dim
        toks = tokenize(text)
        if not toks:
            return v
        for tok in toks:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            v[idx] += sign
        return self._normalize(v)

    @staticmethod
    def _normalize(v):
        n = math.sqrt(sum(x*x for x in v))
        return [x/n for x in v] if n else v


def cosine(a,b):
    if not a or not b:
        return 0.0
    n = min(len(a),len(b))
    dot = sum(a[i]*b[i] for i in range(n))
    na = math.sqrt(sum(x*x for x in a[:n]))
    nb = math.sqrt(sum(x*x for x in b[:n]))
    return dot / (na*nb) if na and nb else 0.0
