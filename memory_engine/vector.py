from __future__ import annotations

import hashlib
import math
import os
import time
from typing import Optional

import requests

from .store import tokenize


class EmbeddingProvider:
    """
    Ollama embedding provider.

    Different embedding spaces must never be mixed silently. Hash fallback is
    therefore opt-in via OLLAMA_ALLOW_HASH_FALLBACK=1.
    """

    def __init__(self, dim=384):
        self.model = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
        self.url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/embeddings")
        self.dim = dim
        self.use_ollama = os.getenv("USE_OLLAMA_EMBEDDING", "1") == "1"
        self.allow_hash_fallback = os.getenv("OLLAMA_ALLOW_HASH_FALLBACK", "0") == "1"
        self.timeout = float(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "30"))
        self.retries = max(0, int(os.getenv("OLLAMA_EMBEDDING_RETRIES", "1")))
        self.embedding_dim: Optional[int] = None

    def embed(self, text: str) -> list[float]:
        """Embed long messages in bounded overlapping chunks without mixing vector spaces."""
        text = str(text or "")
        if not self.use_ollama:
            return self._hash_embed(text)

        max_chars = max(512, int(os.getenv("OLLAMA_EMBED_MAX_CHARS", "6000")))
        overlap = min(max_chars // 4, max(0, int(os.getenv("OLLAMA_EMBED_OVERLAP_CHARS", "200"))))
        if len(text) <= max_chars:
            return self._embed_one(text)

        chunks = self._split_text(text, max_chars=max_chars, overlap=overlap)
        counts: dict[str, int] = {}
        for chunk in chunks:
            counts[chunk] = counts.get(chunk, 0) + 1

        vectors = [
            (self._embed_chunk_adaptive(chunk, min_chars=256), count)
            for chunk, count in counts.items()
        ]
        if not vectors:
            raise ValueError("Cannot embed empty chunked input")
        dims = {len(vector) for vector, _ in vectors}
        if len(dims) != 1:
            raise RuntimeError(f"embedding dimension changed across chunks: {sorted(dims)}")
        total_weight = sum(count for _, count in vectors)
        pooled = [
            sum(vector[i] * count for vector, count in vectors) / total_weight
            for i in range(len(vectors[0][0]))
        ]
        return self._normalize(pooled)

    def _embed_chunk_adaptive(self, text: str, min_chars: int = 256) -> list[float]:
        """Embed a chunk, shrinking it only when Ollama rejects the chunk.

        Some local Ollama configurations expose a much smaller effective
        embedding context than the nominal model context. We therefore keep
        shrinking a rejected chunk down to a small, still meaningful window
        before surfacing the real Ollama error. No alternate embedding space
        is introduced and failures are never silently converted to vectors.
        """
        try:
            return self._embed_one(text)
        except Exception as exc:
            if len(text) <= min_chars:
                raise exc

            midpoint = len(text) // 2
            boundary = text.rfind(" ", min_chars // 2, midpoint)
            if boundary <= min_chars // 2:
                boundary = midpoint

            left = text[:boundary].strip()
            right = text[boundary:].strip()
            if not left or not right:
                raise exc

            left_vector = self._embed_chunk_adaptive(left, min_chars=min_chars)
            right_vector = self._embed_chunk_adaptive(right, min_chars=min_chars)
            if len(left_vector) != len(right_vector):
                raise RuntimeError(
                    f"embedding dimension changed across adaptive chunks: "
                    f"{len(left_vector)} vs {len(right_vector)}"
                )
            pooled = [
                (left_vector[i] + right_vector[i]) / 2.0
                for i in range(len(left_vector))
            ]
            return self._normalize(pooled)

    @staticmethod
    def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
        """Split text into bounded windows, preferring whitespace boundaries."""
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
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(
                    self.url,
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout,
                )
                r.raise_for_status()
                vec = r.json().get("embedding")
                if not vec:
                    raise ValueError("Ollama returned an empty embedding")
                values = [float(x) for x in vec]
                actual_dim = len(values)
                if self.embedding_dim is None:
                    self.embedding_dim = actual_dim
                elif actual_dim != self.embedding_dim:
                    raise ValueError(
                        f"embedding dimension changed: expected {self.embedding_dim}, got {actual_dim}"
                    )
                return self._normalize(values)
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.2 * (attempt + 1))

        if self.allow_hash_fallback:
            fallback_dim = self.embedding_dim or self.dim
            return self._hash_embed(text, dim=fallback_dim)

        raise RuntimeError(
            f"Ollama embedding failed after {self.retries + 1} attempt(s): "
            f"model={self.model}, url={self.url}, error={last_error}"
        ) from last_error

    def _hash_embed(self, text: str, dim: Optional[int] = None):
        dim = int(dim or self.dim)
        v = [0.0] * dim
        toks = tokenize(text)
        if not toks:
            return v
        for tok in toks:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % dim
            sign = 1.0 if h[4] & 1 else -1.0
            v[idx] += sign
        return self._normalize(v)

    @staticmethod
    def _normalize(v):
        n = math.sqrt(sum(x*x for x in v))
        return [x/n for x in v] if n else v


def cosine(a,b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb) if na and nb else 0.0
