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
        self.model = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:4b")
        self.url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/embeddings")
        self.dim = dim
        self.use_ollama = os.getenv("USE_OLLAMA_EMBEDDING", "1") == "1"
        self.allow_hash_fallback = os.getenv("OLLAMA_ALLOW_HASH_FALLBACK", "0") == "1"
        self.timeout = float(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "30"))
        self.retries = max(0, int(os.getenv("OLLAMA_EMBEDDING_RETRIES", "1")))
        self.embedding_dim: Optional[int] = None

    def embed(self, text: str) -> list[float]:
        if not self.use_ollama:
            return self._hash_embed(text)

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
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na*nb) if na and nb else 0.0
