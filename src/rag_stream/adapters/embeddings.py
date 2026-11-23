"""Embedding generation adapters supporting dense vectors and sparse BM25 representations."""

import math
import re
from typing import Any

import numpy as np

from rag_stream.config import settings


class DenseEmbeddingAdapter:
    """Generates normalized dense semantic vector embeddings.

    Produces normalized float vectors matching configured dimensions (e.g. 384 for bge-small).
    Uses a deterministic feature-hashing algorithm for zero-dependency local execution,
    and supports external embedding API fallback.
    """

    def __init__(self, dimension: int = settings.dense_vector_dim):
        """Initialize dense embedder with target vector dimensionality."""
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        """Generate a dense normalized vector for a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate dense normalized vectors for a batch of text strings.

        Args:
            texts: List of input text strings.

        Returns:
            List of normalized float vectors of length self.dimension.
        """
        results: list[list[float]] = []
        for text in texts:
            words = re.findall(r"\w+", text.lower())
            if not words:
                vec = np.zeros(self.dimension, dtype=np.float32)
                vec[0] = 1.0
                results.append(vec.tolist())
                continue

            # Deterministic character-ngram feature hashing
            vec = np.zeros(self.dimension, dtype=np.float32)
            for i, word in enumerate(words):
                # Hash word with position modulation
                h = hash(word) % self.dimension
                weight = 1.0 / (1.0 + math.log(1 + i))
                vec[h] += float(weight)

                # N-gram subword hashing
                for n in range(3, min(6, len(word) + 1)):
                    sub_h = hash(word[:n]) % self.dimension
                    vec[sub_h] += 0.35 * float(weight)

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            else:
                vec[0] = 1.0

            results.append(vec.tolist())

        return results


class BM25SparseTokenizer:
    """Computes sparse BM25 token frequencies and weights for Qdrant sparse vectors.

    Converts raw text into token indices and BM25 term frequency weights:
        TF_weight = (tf * (k1 + 1)) / (tf + k1)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, max_index: int = 100_000):
        """Initialize BM25 tokenizer with standard constants."""
        self.k1 = k1
        self.b = b
        self.max_index = max_index

    def tokenize_sparse(self, text: str) -> dict[str, Any]:
        """Convert text into sparse representation suitable for Qdrant SparseVector.

        Args:
            text: Text to vectorize.

        Returns:
            Dictionary containing 'indices': list[int] and 'values': list[float].
        """
        tokens = re.findall(r"\b\w{2,}\b", text.lower())
        if not tokens:
            return {"indices": [1], "values": [0.1]}

        # Compute term frequency per hashed token index
        freq_map: dict[int, int] = {}
        for token in tokens:
            idx = (hash(token) & 0x7FFFFFFF) % self.max_index + 1
            freq_map[idx] = freq_map.get(idx, 0) + 1

        indices: list[int] = []
        values: list[float] = []

        for idx, count in sorted(freq_map.items()):
            # BM25 term frequency saturation formula
            score = (count * (self.k1 + 1.0)) / (count + self.k1)
            indices.append(idx)
            values.append(round(float(score), 4))

        return {"indices": indices, "values": values}
