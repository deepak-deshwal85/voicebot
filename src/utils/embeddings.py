from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None
        self.enabled = bool(os.getenv("OPENAI_API_KEY"))

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.enabled:
            logger.warning("OPENAI_API_KEY is not set; skipping embedding generation.")
            return []

        client = self._get_client()
        response = await client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> list[float]:
        embeddings = await self.embed_texts([query])
        return embeddings[0] if embeddings else []


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if not left_values or not right_values:
        return 0.0
    if len(left_values) != len(right_values):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)
