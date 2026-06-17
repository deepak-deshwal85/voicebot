from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from utils.config import AgentConfig, load_agent_config
from utils.embeddings import EmbeddingService, cosine_similarity
from utils.resume_sections import SECTION_SEARCH_TERMS

logger = logging.getLogger(__name__)

_SNIPPET_MAX_CHARS = 320
_TOP_K_DEFAULT = 3
_MIN_KEYWORD_SCORE_MULTI = 2
_EMBEDDING_MIN_SCORE = 0.18
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "with",
    }
)


def _meaningful_words(query: str) -> set[str]:
    return _tokenize(query) - _STOP_WORDS


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _load_documents(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("documents", [])
    return []


def _doc_search_text(doc: dict[str, Any]) -> str:
    return str(doc.get("search_text") or doc.get("text", ""))


def _topic_terms(doc: dict[str, Any]) -> set[str]:
    metadata = doc.get("metadata", {})
    terms: set[str] = set()
    for keyword in metadata.get("keywords", []):
        terms.update(_tokenize(str(keyword)))
    section = str(metadata.get("section", ""))
    terms.update(SECTION_SEARCH_TERMS.get(section, frozenset()))
    return terms


def _truncate_snippet(text: str, max_chars: int = _SNIPPET_MAX_CHARS) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _format_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""

    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        excerpt = _truncate_snippet(result.get("text", ""))
        if not excerpt:
            continue
        filename = result.get("filename") or "resume"
        section = result.get("section")
        section_label = f", {section}" if section else ""
        label = f"Resume {index} ({filename}{section_label})"
        score = result.get("score")
        score_text = f" score={score:.2f}" if isinstance(score, float) else ""
        lines.append(f"{label}{score_text}: {excerpt}")
    return "\n".join(lines)


class KnowledgeStore:
    """Lazy resume knowledge store loaded from config/{client}-resume.json."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or load_agent_config()
        self.path = self.config.resume_knowledge_path
        self.embeddings = EmbeddingService(model=self.config.embedding_model)
        self.documents: list[dict[str, Any]] = []
        self._loaded = False
        self._loading = False

    async def initialize(self) -> None:
        return None

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
                "Loaded %s resume chunks from %s",
                len(self.documents),
                self.path.name,
            )
            return bool(self.documents)
        except Exception as exc:
            logger.error("Failed to load resume knowledge from %s: %s", self.path, exc)
            self.documents = []
            self._loaded = True
            return False
        finally:
            self._loading = False

    async def preload(self) -> bool:
        return await self.ensure_loaded()

    async def search(self, query: str, top_k: int = _TOP_K_DEFAULT) -> str:
        results = await self._search_documents(query, top_k=top_k)
        return _format_results(results)

    async def _search_documents(
        self, query: str, top_k: int = _TOP_K_DEFAULT
    ) -> list[dict[str, Any]]:
        if not await self.ensure_loaded() or not self.documents:
            return []

        keyword_results = self._search_keywords(query, top_k=top_k)
        if keyword_results:
            logger.debug(
                "Resume keyword search matched %s chunk(s) for %r",
                len(keyword_results),
                query,
            )
            return keyword_results

        if any(doc.get("embedding") for doc in self.documents):
            results = await self._search_embeddings(query, top_k=top_k)
            if results:
                logger.debug(
                    "Resume embedding search matched %s chunk(s) for %r",
                    len(results),
                    query,
                )
                return results
            logger.warning(
                "Embedding search returned no resume results for query %r.",
                query,
            )

        return []

    def _search_keywords(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_words = _meaningful_words(query)
        if not query_words:
            return []

        min_score = 1 if len(query_words) == 1 else _MIN_KEYWORD_SCORE_MULTI

        scored: list[tuple[int, dict[str, Any]]] = []
        for doc in self.documents:
            doc_words = _meaningful_words(_doc_search_text(doc))
            text_score = len(query_words.intersection(doc_words))
            topic_score = len(query_words.intersection(_topic_terms(doc)))
            score = text_score + topic_score
            if score <= 0:
                continue
            metadata = doc.get("metadata", {})
            scored.append(
                (
                    score,
                    {
                        "text": doc["text"],
                        "filename": metadata.get("filename", ""),
                        "section": metadata.get("section", ""),
                        "score": float(score),
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored or scored[0][0] < min_score:
            return []
        return [doc for _, doc in scored[:top_k]]

    async def _search_embeddings(
        self, query: str, top_k: int, min_score: float = _EMBEDDING_MIN_SCORE
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
                        "filename": metadata.get("filename", ""),
                        "section": metadata.get("section", ""),
                        "score": score,
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored if doc.get("score", 0) > min_score][:top_k]
