from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Literal

from utils.config import AgentConfig, load_agent_config
from utils.embeddings import EmbeddingService, cosine_similarity
from utils.knowledge_router import KnowledgeSource

logger = logging.getLogger(__name__)

SourceName = Literal["website", "pdf"]
_SNIPPET_MAX_CHARS = 320
_TOP_K_DEFAULT = 3


def _load_documents(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("documents", [])
    return []


def _truncate_snippet(text: str, max_chars: int = _SNIPPET_MAX_CHARS) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _format_results(results: list[dict[str, Any]], source: SourceName) -> str:
    if not results:
        return ""

    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        excerpt = _truncate_snippet(result.get("text", ""))
        if not excerpt:
            continue
        if source == "website":
            url = result.get("url", "")
            label = f"Website {index}"
            if url:
                label = f"{label} ({url})"
        else:
            filename = result.get("filename") or "document"
            label = f"PDF {index} ({filename})"
        score = result.get("score")
        score_text = f" score={score:.2f}" if isinstance(score, float) else ""
        lines.append(f"{label}{score_text}: {excerpt}")
    return "\n".join(lines)


class _SourcePool:
    def __init__(self, path: Path, source: SourceName, embedding_model: str) -> None:
        self.path = path
        self.source = source
        self.embeddings = EmbeddingService(model=embedding_model)
        self.documents: list[dict[str, Any]] = []
        self._loaded = False
        self._loading = False

    async def ensure_loaded(self) -> bool:
        if self._loaded:
            return bool(self.documents)

        while self._loading:
            await asyncio.sleep(0.05)
            if self._loaded:
                return bool(self.documents)

        self._loading = True
        try:
            self.documents = await asyncio.to_thread(_load_documents, self.path)
            self._loaded = True
            logger.info(
                "Loaded %s %s chunks from %s",
                len(self.documents),
                self.source,
                self.path.name,
            )
            return bool(self.documents)
        except Exception as exc:
            logger.error(
                "Failed to load %s knowledge from %s: %s", self.source, self.path, exc
            )
            self.documents = []
            self._loaded = True
            return False
        finally:
            self._loading = False

    async def search(
        self, query: str, top_k: int = _TOP_K_DEFAULT
    ) -> list[dict[str, Any]]:
        if not await self.ensure_loaded() or not self.documents:
            return []

        if any(doc.get("embedding") for doc in self.documents):
            results = await self._search_embeddings(query, top_k=top_k)
            if results:
                return results
            logger.warning(
                "Embedding search returned no %s results; falling back to keywords.",
                self.source,
            )

        return self._search_keywords(query, top_k=top_k)

    def _search_keywords(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_words = set(query.lower().split())
        scored: list[tuple[int, dict[str, Any]]] = []

        for doc in self.documents:
            doc_words = set(doc["text"].lower().split())
            score = len(query_words.intersection(doc_words))
            if score <= 0:
                continue
            metadata = doc.get("metadata", {})
            scored.append(
                (
                    score,
                    {
                        "text": doc["text"],
                        "source": self.source,
                        "filename": metadata.get("filename", ""),
                        "url": metadata.get("url", ""),
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
                        "source": self.source,
                        "filename": metadata.get("filename", ""),
                        "url": metadata.get("url", ""),
                        "score": score,
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored if doc.get("score", 0) > min_score][:top_k]


class KnowledgeStore:
    """Lazy split knowledge store: website and PDF indexes load on demand."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or load_agent_config()
        self.website = _SourcePool(
            self.config.website_knowledge_path,
            "website",
            self.config.embedding_model,
        )
        self.pdf = _SourcePool(
            self.config.pdf_knowledge_path,
            "pdf",
            self.config.embedding_model,
        )

    @property
    def documents(self) -> list[dict[str, Any]]:
        return [*self.website.documents, *self.pdf.documents]

    async def initialize(self) -> None:
        return None

    async def preload_pdf(self) -> bool:
        """Load the PDF index early."""
        return await self.pdf.ensure_loaded()

    async def preload_website(self) -> bool:
        """Load the website index early."""
        return await self.website.ensure_loaded()

    async def search_website(self, query: str, top_k: int = _TOP_K_DEFAULT) -> str:
        results = await self.website.search(query, top_k=top_k)
        return _format_results(results, "website")

    async def search_pdf(self, query: str, top_k: int = _TOP_K_DEFAULT) -> str:
        results = await self.pdf.search(query, top_k=top_k)
        return _format_results(results, "pdf")

    async def search_routed(
        self,
        query: str,
        source: KnowledgeSource,
        top_k: int = _TOP_K_DEFAULT,
    ) -> str:
        if source == "website":
            return await self.search_website(query, top_k=top_k)
        if source == "pdf":
            return await self.search_pdf(query, top_k=top_k)

        website_results, pdf_results = await asyncio.gather(
            self.website.search(query, top_k=max(1, top_k // 2)),
            self.pdf.search(query, top_k=max(1, top_k // 2)),
        )
        merged = [*website_results, *pdf_results]
        merged.sort(key=lambda item: item.get("score", 0), reverse=True)
        return _format_results(merged[:top_k], "website")

    async def search(
        self, query: str, top_k: int = _TOP_K_DEFAULT
    ) -> list[dict[str, Any]]:
        from utils.knowledge_router import route_knowledge_source

        source = route_knowledge_source(query)
        if source == "website":
            return await self.website.search(query, top_k=top_k)
        if source == "pdf":
            return await self.pdf.search(query, top_k=top_k)

        website_results, pdf_results = await asyncio.gather(
            self.website.search(query, top_k=max(1, top_k // 2)),
            self.pdf.search(query, top_k=max(1, top_k // 2)),
        )
        merged = [*website_results, *pdf_results]
        merged.sort(key=lambda item: item.get("score", 0), reverse=True)
        return merged[:top_k]
