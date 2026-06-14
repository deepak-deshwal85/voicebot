from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from utils.config import AgentConfig, load_agent_config
from utils.embeddings import EmbeddingService, cosine_similarity

logger = logging.getLogger(__name__)


class KnowledgeStore:
    STORE_VERSION = 2

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or load_agent_config()
        self.data_path = self.config.knowledge_data_path
        self.website_url = self.config.website_url
        self.documents: list[dict[str, Any]] = []
        self.scraped_pages: set[str] = set()
        self.embedding_model = self.config.embedding_model
        self.embeddings = EmbeddingService(model=self.embedding_model)

    async def initialize(self) -> None:
        """Load the pre-built knowledge base. No scraping at runtime."""
        logger.info("Initializing knowledge store from %s", self.data_path)
        await self._load_or_create_store()
        logger.info(
            "Knowledge store initialized with %s documents", len(self.documents)
        )

    async def rebuild(
        self,
        max_pages: int | None = None,
        force_refresh: bool = True,
        include_website: bool = True,
        include_pdfs: bool = True,
    ) -> None:
        """Build the knowledge base from website content and PDF documents."""
        page_limit = max_pages or self.config.max_pages

        if force_refresh:
            self.documents = []
            self.scraped_pages = set()

        if include_pdfs:
            await self.ingest_pdfs()

        if include_website:
            await self.scrape_website(max_pages=page_limit)

        await self._compute_embeddings()
        await self._save_store()
        logger.info(
            "Knowledge base rebuilt with %s documents from website and PDFs",
            len(self.documents),
        )

    async def _load_or_create_store(self) -> None:
        try:
            if self.data_path.exists():
                with open(self.data_path, encoding="utf-8") as handle:
                    payload = json.load(handle)

                if isinstance(payload, list):
                    self.documents = payload
                elif isinstance(payload, dict):
                    self.documents = payload.get("documents", [])
                    for doc in self.documents:
                        url = doc.get("metadata", {}).get("url")
                        if url:
                            self.scraped_pages.add(url)
                else:
                    self.documents = []
            else:
                self.data_path.parent.mkdir(parents=True, exist_ok=True)
                self.documents = []
                await self._save_store()
        except Exception as exc:
            logger.error("Error loading knowledge store: %s", exc)
            self.documents = []

    async def _save_store(self) -> None:
        payload = {
            "version": self.STORE_VERSION,
            "website_url": self.website_url,
            "embedding_model": self.embedding_model,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "documents": self.documents,
        }
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            logger.error("Error saving knowledge store: %s", exc)

    async def add_document(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        *,
        auto_save: bool = True,
    ) -> None:
        doc = {
            "text": text,
            "metadata": metadata or {},
        }
        self.documents.append(doc)
        if auto_save:
            await self._save_store()

    async def ingest_pdfs(self) -> None:
        pdf_folder = self.config.pdf_folder
        if not pdf_folder.exists():
            logger.warning("PDF folder does not exist: %s", pdf_folder)
            return

        pdf_files = sorted(pdf_folder.glob("*.pdf"))
        if not pdf_files:
            logger.info("No PDF files found in %s", pdf_folder)
            return

        logger.info("Ingesting %s PDF file(s) from %s", len(pdf_files), pdf_folder)
        for pdf_path in pdf_files:
            text = await self._extract_pdf_text(pdf_path)
            if not text.strip():
                logger.warning("No text extracted from %s", pdf_path.name)
                continue

            chunks = self._split_text(text, max_length=self.config.chunk_size)
            for index, chunk in enumerate(chunks):
                self.documents.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "type": "pdf_content",
                            "source": "pdf",
                            "filename": pdf_path.name,
                            "path": str(pdf_path),
                            "chunk": index,
                        },
                    }
                )
            logger.info("Added %s chunk(s) from %s", len(chunks), pdf_path.name)

    async def _extract_pdf_text(self, pdf_path: Path) -> str:
        def read_pdf() -> str:
            reader = PdfReader(str(pdf_path))
            pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)
            return "\n".join(pages)

        try:
            return await asyncio.get_event_loop().run_in_executor(None, read_pdf)
        except Exception as exc:
            logger.error("Error reading PDF %s: %s", pdf_path, exc)
            return ""

    async def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.documents:
            logger.warning("Knowledge base is empty")
            return []

        if self._has_embeddings():
            results = await self._search_embeddings(query, top_k)
            if results:
                return results

        return self._search_keywords(query, top_k)

    async def search_with_fallback(
        self, query: str, top_k: int = 3
    ) -> list[dict[str, Any]]:
        results = await self.search(query, top_k)
        if results or not self.config.runtime_scraping_enabled:
            return results

        logger.info(
            "No indexed content found for '%s'; runtime scraping is enabled",
            query,
        )
        return await self.search_website(query, top_k)

    def _has_embeddings(self) -> bool:
        return any(doc.get("embedding") for doc in self.documents)

    async def _search_embeddings(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_embedding = await self.embeddings.embed_query(query)
        if not query_embedding:
            return []

        scored_docs: list[tuple[float, dict[str, Any]]] = []
        for doc in self.documents:
            embedding = doc.get("embedding")
            if not embedding:
                continue
            score = cosine_similarity(query_embedding, embedding)
            scored_docs.append(
                (
                    score,
                    {
                        "text": doc["text"],
                        "type": doc.get("metadata", {}).get("type", ""),
                        "source": doc.get("metadata", {}).get("source", ""),
                        "score": score,
                    },
                )
            )

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k] if doc.get("score", 0) > 0.2]

    def _search_keywords(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_words = set(query.lower().split())
        scored_docs: list[tuple[int, dict[str, Any]]] = []

        for doc in self.documents:
            doc_words = set(doc["text"].lower().split())
            score = len(query_words.intersection(doc_words))
            if score > 0:
                scored_docs.append(
                    (
                        score,
                        {
                            "text": doc["text"],
                            "type": doc.get("metadata", {}).get("type", ""),
                            "source": doc.get("metadata", {}).get("source", ""),
                        },
                    )
                )

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    async def search_website(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        try:
            if not await self._can_scrape_website():
                logger.warning("Cannot scrape website according to robots.txt")
                return []

            await self.scrape_website(max_pages=min(self.config.max_pages, 10))
            await self._compute_embeddings()
            await self._save_store()
            return await self.search(query, top_k)
        except Exception as exc:
            logger.error("Error searching website: %s", exc)
            return []

    async def _compute_embeddings(self) -> None:
        if not self.embeddings.enabled:
            logger.warning(
                "Embeddings were not generated because OPENAI_API_KEY is missing"
            )
            return

        texts = [doc["text"] for doc in self.documents]
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = await self.embeddings.embed_texts(batch)
            for offset, vector in enumerate(vectors):
                self.documents[start + offset]["embedding"] = vector

    async def _can_scrape_website(self) -> bool:
        try:
            robots_url = urljoin(self.website_url, "/robots.txt")
            robot_parser = RobotFileParser()
            robot_parser.set_url(robots_url)
            await asyncio.get_event_loop().run_in_executor(None, robot_parser.read)
            return robot_parser.can_fetch("*", self.website_url)
        except Exception as exc:
            logger.warning("Could not check robots.txt: %s", exc)
            return True

    async def scrape_website(self, max_pages: int = 10) -> None:
        try:
            logger.info("Starting website scraping of %s", self.website_url)
            urls_to_scrape = [self.website_url]
            scraped_count = 0

            while urls_to_scrape and scraped_count < max_pages:
                current_url = urls_to_scrape.pop(0)
                if current_url in self.scraped_pages:
                    continue

                try:
                    page_html, page_text = await self._scrape_page(current_url)
                    if page_text:
                        await self._add_web_content(current_url, page_text)
                        self.scraped_pages.add(current_url)
                        scraped_count += 1

                        for url in self._extract_urls(page_html, current_url):
                            if (
                                url not in self.scraped_pages
                                and len(urls_to_scrape) < max_pages * 2
                            ):
                                urls_to_scrape.append(url)
                except Exception as exc:
                    logger.error("Error scraping %s: %s", current_url, exc)

            logger.info("Scraped %s page(s) from website", scraped_count)
        except Exception as exc:
            logger.error("Error during website scraping: %s", exc)

    async def _scrape_page(self, url: str) -> tuple[str, str]:
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            }
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(url, headers=headers, timeout=10),
            )

            if response.status_code != 200:
                logger.warning(
                    "Failed to scrape %s: HTTP %s", url, response.status_code
                )
                return "", ""

            soup = BeautifulSoup(response.content, "lxml")
            page_html = response.text

            for element in soup(["script", "style"]):
                element.decompose()

            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            return page_html, text
        except Exception as exc:
            logger.error("Error scraping page %s: %s", url, exc)
            return "", ""

    def _extract_urls(self, html: str, base_url: str) -> list[str]:
        urls: list[str] = []
        if not html:
            return urls

        try:
            soup = BeautifulSoup(html, "lxml")
            for link in soup.find_all("a", href=True):
                absolute_url = urljoin(base_url, link["href"])
                if urlparse(absolute_url).netloc == urlparse(self.website_url).netloc:
                    urls.append(absolute_url)
        except Exception as exc:
            logger.error("Error extracting URLs: %s", exc)

        return list(set(urls))

    async def _add_web_content(self, url: str, content: str) -> None:
        if not content or len(content.strip()) < 100:
            return

        chunks = self._split_text(content, max_length=self.config.chunk_size)
        for index, chunk in enumerate(chunks):
            self.documents.append(
                {
                    "text": chunk,
                    "metadata": {
                        "type": "website_content",
                        "source": "website",
                        "url": url,
                        "chunk": index,
                    },
                }
            )

    def _split_text(self, text: str, max_length: int = 1000) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        current_chunk: list[str] = []

        for word in words:
            candidate = " ".join([*current_chunk, word])
            if len(candidate) <= max_length:
                current_chunk.append(word)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
