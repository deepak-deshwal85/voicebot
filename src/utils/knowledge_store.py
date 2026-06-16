from __future__ import annotations

import json
import logging
from typing import Any

from utils.config import AgentConfig, load_agent_config
from utils.embeddings import EmbeddingService, cosine_similarity

logger = logging.getLogger(__name__)


class KnowledgeStore:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or load_agent_config()
        self.documents: list[dict[str, Any]] = []
        self.embeddings = EmbeddingService(model=self.config.embedding_model)

    async def initialize(self) -> None:
        path = self.config.knowledge_path
        if not path.exists():
            logger.warning("Knowledge base missing: %s", path)
            self.documents = []
            return

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                self.documents = payload
            elif isinstance(payload, dict):
                self.documents = payload.get("documents", [])
            else:
                self.documents = []
        except Exception as exc:
            logger.error("Failed to load knowledge base %s: %s", path, exc)
            self.documents = []

        logger.info(
            "Loaded %s chunks from %s",
            len(self.documents),
            path.name,
        )

    async def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.documents:
            return []

        if any(doc.get("embedding") for doc in self.documents):
            results = await self._search_embeddings(query, top_k)
            if results:
                return results

        return self._search_keywords(query, top_k)

    async def _search_embeddings(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_embedding = await self.embeddings.embed_query(query)
        if not query_embedding:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self.documents:
            embedding = doc.get("embedding")
            if not embedding:
                continue
            score = cosine_similarity(query_embedding, embedding)
            scored.append(
                (
                    score,
                    {
                        "text": doc["text"],
                        "source": doc.get("metadata", {}).get("source", ""),
                        "score": score,
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k] if doc.get("score", 0) > 0.2]

    def _search_keywords(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_words = set(query.lower().split())
        scored: list[tuple[int, dict[str, Any]]] = []

        for doc in self.documents:
            doc_words = set(doc["text"].lower().split())
            score = len(query_words.intersection(doc_words))
            if score > 0:
                scored.append(
                    (
                        score,
                        {
                            "text": doc["text"],
                            "source": doc.get("metadata", {}).get("source", ""),
                        },
                    )
                )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
