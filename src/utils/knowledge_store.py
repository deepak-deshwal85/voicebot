from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from utils.config import AgentConfig, load_agent_config
from utils.embeddings import EmbeddingService, cosine_similarity

logger = logging.getLogger(__name__)


def _load_documents(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("documents", [])
    return []


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
            self.documents = await asyncio.to_thread(_load_documents, path)
        except Exception as exc:
            logger.error("Failed to load knowledge base %s: %s", path, exc)
            self.documents = []

        logger.info(
            "Loaded %s chunks from %s",
            len(self.documents),
            path.name,
        )

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.documents:
            return []

        if any(doc.get("embedding") for doc in self.documents):
            return await self._search_embeddings_balanced(query, top_k)

        return self._search_keywords_balanced(query, top_k)

    async def _search_embeddings_balanced(
        self, query: str, top_k: int
    ) -> list[dict[str, Any]]:
        candidates = await self._search_embeddings(
            query, top_k=min(top_k * 4, 20), min_score=0.2
        )
        if not candidates:
            return []

        pdf_hits = [doc for doc in candidates if doc.get("source") == "pdf"]
        web_hits = [doc for doc in candidates if doc.get("source") == "website"]

        merged: list[dict[str, Any]] = []
        if web_hits:
            merged.append(web_hits[0])
        if pdf_hits and (not merged or pdf_hits[0].get("score", 0) >= 0.25):
            merged.append(pdf_hits[0])

        for doc in candidates:
            if doc not in merged:
                merged.append(doc)
            if len(merged) >= top_k:
                break

        return merged[:top_k]

    def _search_keywords_balanced(self, query: str, top_k: int) -> list[dict[str, Any]]:
        pdf_docs = [
            doc for doc in self.documents if doc.get("metadata", {}).get("source") == "pdf"
        ]
        web_docs = [
            doc
            for doc in self.documents
            if doc.get("metadata", {}).get("source") == "website"
        ]

        merged: list[dict[str, Any]] = []
        for pool in (web_docs, pdf_docs):
            merged.extend(self._search_keywords_pool(query, pool, max(1, top_k // 2)))

        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for doc in sorted(merged, key=lambda item: item.get("score", 0), reverse=True):
            text = doc.get("text", "")
            if text in seen:
                continue
            seen.add(text)
            unique.append(doc)
            if len(unique) >= top_k:
                break
        return unique

    def _search_keywords_pool(
        self, query: str, documents: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        query_words = set(query.lower().split())
        scored: list[tuple[int, dict[str, Any]]] = []

        for doc in documents:
            doc_words = set(doc["text"].lower().split())
            score = len(query_words.intersection(doc_words))
            if score > 0:
                scored.append(
                    (
                        score,
                        {
                            "text": doc["text"],
                            "source": doc.get("metadata", {}).get("source", ""),
                            "filename": doc.get("metadata", {}).get("filename", ""),
                            "score": float(score),
                        },
                    )
                )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    async def _search_embeddings(
        self, query: str, top_k: int, min_score: float = 0.2
    ) -> list[dict[str, Any]]:
        query_embedding = await self.embeddings.embed_query(query)
        if not query_embedding:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self.documents:
            embedding = doc.get("embedding")
            if not embedding:
                continue
            score = cosine_similarity(query_embedding, embedding)
            metadata = doc.get("metadata", {})
            scored.append(
                (
                    score,
                    {
                        "text": doc["text"],
                        "source": metadata.get("source", ""),
                        "filename": metadata.get("filename", ""),
                        "score": score,
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored if doc.get("score", 0) > min_score][:top_k]
