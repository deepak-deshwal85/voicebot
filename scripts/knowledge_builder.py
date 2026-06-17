from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from utils.config import AgentConfig, client_resume_knowledge_path
from utils.embeddings import EmbeddingService
from utils.resume_sections import (
    build_search_text,
    section_keywords,
    split_resume_sections,
)

logger = logging.getLogger(__name__)


class KnowledgeBuilder:
    STORE_VERSION = 3

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.documents: list[dict[str, Any]] = []
        self.embeddings = EmbeddingService(model=config.embedding_model)

    async def build(self) -> Path:
        self.documents = []
        await self._ingest_resumes()
        await self._compute_embeddings()

        output = client_resume_knowledge_path(self.config.client_id)
        payload = {
            "version": self.STORE_VERSION,
            "client_id": self.config.client_id,
            "source": "resume",
            "embedding_model": self.config.embedding_model,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "documents": self.documents,
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote %s resume chunks to %s", len(self.documents), output)
        return output

    async def _ingest_resumes(self) -> None:
        folder = self.config.resume_folder
        if not folder.exists():
            logger.warning("Resume folder missing: %s", folder)
            return

        for pdf_path in sorted(folder.glob("*.pdf")):
            text = await self._extract_pdf_text(pdf_path)
            if not text.strip():
                continue
            self._add_section_chunks(text, filename=pdf_path.name, path=str(pdf_path))
            logger.info("Ingested resume PDF %s", pdf_path.name)

    def _add_section_chunks(self, text: str, *, filename: str, path: str) -> None:
        for section, section_text in split_resume_sections(text):
            if len(section_text.strip()) < 40:
                continue
            keywords = section_keywords(section)
            for index, chunk in enumerate(self._split_text(section_text)):
                self.documents.append(
                    {
                        "text": chunk,
                        "search_text": build_search_text(section, chunk),
                        "metadata": {
                            "source": "resume",
                            "type": "resume_content",
                            "section": section,
                            "keywords": keywords,
                            "filename": filename,
                            "path": path,
                            "chunk": index,
                        },
                    }
                )

    async def _compute_embeddings(self) -> None:
        if not self.embeddings.enabled:
            logger.warning("OPENAI_API_KEY missing; skipping embeddings")
            return

        texts = [doc.get("search_text") or doc["text"] for doc in self.documents]
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = await self.embeddings.embed_texts(batch)
            for offset, vector in enumerate(vectors):
                self.documents[start + offset]["embedding"] = vector

    async def _extract_pdf_text(self, pdf_path: Path) -> str:
        def read_pdf() -> str:
            reader = PdfReader(str(pdf_path))
            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages
                if (page.extract_text() or "").strip()
            )

        try:
            return await asyncio.get_event_loop().run_in_executor(None, read_pdf)
        except Exception as exc:
            logger.error("Error reading %s: %s", pdf_path, exc)
            return ""

    def _split_text(self, text: str) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if len(candidate) <= self.config.chunk_size:
                current.append(word)
            else:
                if current:
                    chunks.append(" ".join(current))
                current = [word]
        if current:
            chunks.append(" ".join(current))
        return chunks
