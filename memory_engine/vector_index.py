from __future__ import annotations

import threading
from typing import Iterable

import numpy as np


class MemoryVectorIndex:
    """
    Process-level dense vector index.

    SQLite remains the durable source of truth. Every vector for one user must
    use exactly the same embedding dimension.
    """

    def __init__(self, store):
        self.store = store
        self._lock = threading.RLock()
        self._loaded_users: set[str] = set()
        self._vectors: dict[str, dict[str, np.ndarray]] = {}
        self._matrix: dict[str, np.ndarray] = {}
        self._ids: dict[str, list[str]] = {}
        self._dims: dict[str, int] = {}

    def _ensure_user(self, user_id: str) -> None:
        with self._lock:
            if user_id in self._loaded_users:
                return

            rows = self.store.embeddings(user_id)
            vectors: dict[str, np.ndarray] = {}
            dims: set[int] = set()
            for memory_id, vector in rows:
                arr = np.asarray(vector, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    continue
                dims.add(int(arr.size))
                norm = float(np.linalg.norm(arr))
                if norm > 0:
                    arr = arr / norm
                vectors[memory_id] = arr

            if len(dims) > 1:
                raise ValueError(
                    f"mixed embedding dimensions for user {user_id}: {sorted(dims)}"
                )

            self._vectors[user_id] = vectors
            if dims:
                self._dims[user_id] = next(iter(dims))
            self._rebuild_matrix(user_id)
            self._loaded_users.add(user_id)

    def _rebuild_matrix(self, user_id: str) -> None:
        vectors = self._vectors.setdefault(user_id, {})
        ids = list(vectors.keys())
        self._ids[user_id] = ids
        if not ids:
            self._matrix[user_id] = np.empty((0, 0), dtype=np.float32)
            return

        dims = {len(vectors[mid]) for mid in ids}
        if len(dims) != 1:
            raise ValueError(
                f"mixed embedding dimensions in in-memory index for user {user_id}: {sorted(dims)}"
            )

        dim = next(iter(dims))
        expected = self._dims.get(user_id)
        if expected is not None and dim != expected:
            raise ValueError(
                f"embedding dimension mismatch for user {user_id}: expected {expected}, got {dim}"
            )
        self._dims[user_id] = dim
        self._matrix[user_id] = np.vstack(
            [vectors[mid] for mid in ids]
        ).astype(np.float32, copy=False)

    def add(self, user_id: str, memory_id: str, vector: list[float]) -> None:
        self._ensure_user(user_id)

        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError("embedding is empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError("embedding contains non-finite values")
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm

        with self._lock:
            expected = self._dims.get(user_id)
            if expected is not None and len(arr) != expected:
                raise ValueError(
                    f"embedding dimension mismatch for user {user_id}: expected {expected}, got {len(arr)}"
                )
            if expected is None:
                self._dims[user_id] = len(arr)
            self._vectors.setdefault(user_id, {})[memory_id] = arr
            self._rebuild_matrix(user_id)

    def search(
        self,
        user_id: str,
        query_vector: list[float],
        ids: Iterable[str] | None = None,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        self._ensure_user(user_id)

        with self._lock:
            matrix = self._matrix.get(user_id)
            all_ids = self._ids.get(user_id, [])
            if matrix is None or matrix.size == 0:
                return []

            q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
            expected = self._dims.get(user_id)
            if expected is not None and len(q) != expected:
                raise ValueError(
                    f"query embedding dimension mismatch for user {user_id}: index={expected}, query={len(q)}"
                )
            qnorm = float(np.linalg.norm(q))
            if qnorm <= 0:
                return []
            q = q / qnorm

            allowed = set(ids) if ids is not None else None
            positions = [
                i for i, mid in enumerate(all_ids)
                if allowed is None or mid in allowed
            ]
            if not positions:
                return []

            sub = matrix[positions]
            scores = sub @ q
            limit = min(max(1, top_k), len(positions))
            if limit < len(positions):
                local = np.argpartition(-scores, limit - 1)[:limit]
                local = local[np.argsort(-scores[local])]
            else:
                local = np.argsort(-scores)

            return [
                (all_ids[positions[int(i)]], float(scores[int(i)]))
                for i in local
            ]
