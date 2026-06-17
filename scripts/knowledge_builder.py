from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from utils.config import (
    AgentConfig,
    client_pdf_knowledge_path,
    client_website_knowledge_path,
)
from utils.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class KnowledgeBuilder:
    STORE_VERSION = 2

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.documents: list[dict[str, Any]] = []
        self.scraped_pages: set[str] = set()
        self.embeddings = EmbeddingService(model=config.embedding_model)

    async def build(self, *, max_pages: int | None = None) -> tuple[Path, Path]:
        page_limit = max_pages or self.config.max_pages
        self.documents = []

        await self._scrape_website(max_pages=page_limit)
        await self._ingest_pdfs()
        await self._compute_embeddings()

        website_docs = [
            doc
            for doc in self.documents
            if doc.get("metadata", {}).get("source") == "website"
        ]
        pdf_docs = [
            doc
            for doc in self.documents
            if doc.get("metadata", {}).get("source") == "pdf"
        ]
        built_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            "version": self.STORE_VERSION,
            "client_id": self.config.client_id,
            "website_url": self.config.website_url,
            "embedding_model": self.config.embedding_model,
            "built_at": built_at,
        }
        website_path = client_website_knowledge_path(self.config.client_id)
        pdf_path = client_pdf_knowledge_path(self.config.client_id)
        website_path.write_text(
            json.dumps(
                {**metadata, "source": "website", "documents": website_docs}, indent=2
            ),
            encoding="utf-8",
        )
        pdf_path.write_text(
            json.dumps({**metadata, "source": "pdf", "documents": pdf_docs}, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Wrote website=%s (%s chunks) pdf=%s (%s chunks)",
            website_path,
            len(website_docs),
            pdf_path,
            len(pdf_docs),
        )
        return website_path, pdf_path

    async def _scrape_website(self, max_pages: int) -> None:
        urls = [self.config.website_url]
        scraped = 0

        while urls and scraped < max_pages:
            current_url = urls.pop(0)
            if current_url in self.scraped_pages:
                continue

            html, text = await self._scrape_page(current_url)
            if text:
                self._add_chunks(text, source="website", url=current_url)
                self.scraped_pages.add(current_url)
                scraped += 1
                for link in self._extract_urls(html, current_url):
                    if link not in self.scraped_pages and len(urls) < max_pages * 2:
                        urls.append(link)

        logger.info("Scraped %s website page(s)", scraped)

    async def _ingest_pdfs(self) -> None:
        folder = self.config.pdf_folder
        if not folder.exists():
            logger.warning("PDF folder missing: %s", folder)
            return

        pdf_files = sorted(folder.glob("*.pdf"))
        for pdf_path in pdf_files:
            text = await self._extract_pdf_text(pdf_path)
            if not text.strip():
                continue
            self._add_chunks(
                text,
                source="pdf",
                filename=pdf_path.name,
                path=str(pdf_path),
            )
            logger.info("Ingested PDF %s", pdf_path.name)

    def _add_chunks(
        self,
        text: str,
        *,
        source: str,
        url: str | None = None,
        filename: str | None = None,
        path: str | None = None,
    ) -> None:
        if len(text.strip()) < 100:
            return

        for index, chunk in enumerate(self._split_text(text)):
            metadata: dict[str, Any] = {"source": source, "chunk": index}
            if source == "website":
                metadata.update({"type": "website_content", "url": url})
            else:
                metadata.update(
                    {"type": "pdf_content", "filename": filename, "path": path}
                )
            self.documents.append({"text": chunk, "metadata": metadata})

    async def _compute_embeddings(self) -> None:
        if not self.embeddings.enabled:
            logger.warning("OPENAI_API_KEY missing; skipping embeddings")
            return

        texts = [doc["text"] for doc in self.documents]
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

    async def _scrape_page(self, url: str) -> tuple[str, str]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(url, headers=headers, timeout=10),
            )
            if response.status_code != 200:
                return "", ""

            soup = BeautifulSoup(response.content, "lxml")
            for element in soup(["script", "style"]):
                element.decompose()
            text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True)).strip()
            return response.text, text
        except Exception as exc:
            logger.error("Error scraping %s: %s", url, exc)
            return "", ""

    def _extract_urls(self, html: str, base_url: str) -> list[str]:
        if not html:
            return []
        soup = BeautifulSoup(html, "lxml")
        base_netloc = urlparse(self.config.website_url).netloc
        urls = []
        for link in soup.find_all("a", href=True):
            absolute = urljoin(base_url, link["href"])
            if urlparse(absolute).netloc == base_netloc:
                urls.append(absolute)
        return list(set(urls))

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
