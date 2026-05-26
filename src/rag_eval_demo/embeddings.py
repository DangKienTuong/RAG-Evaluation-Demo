from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol

from .text_utils import tokenize


class Embedder(Protocol):
    model_name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class LocalHashEmbedder:
    """Deterministic offline embedder for smoke tests; OpenAI remains the default."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions
        self.model_name = f"local-hash-{dimensions}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            sign = -1.0 if value & 1 else 1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbedder:
    def __init__(self, model_name: str) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY chưa được cấu hình. Tạo .env từ .env.example "
                "hoặc dùng --local-embeddings chỉ cho smoke test offline."
            )
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend([item.embedding for item in ordered])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def make_embedder(provider: str, model_name: str) -> Embedder:
    if provider == "local":
        return LocalHashEmbedder()
    if provider == "openai":
        return OpenAIEmbedder(model_name)
    raise ValueError(f"Embedding provider không hỗ trợ: {provider}")
