import asyncio
import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Single-source knowledge module: crawl website, persist chunks, and query fast."""

    def __init__(self, data_path: str = "data/knowledge_base.json"):
        self.data_path = Path(data_path)
        self.website_url = os.getenv(
            "KNOWLEDGE_WEBSITE_URL", "https://www.fidelityinternational.com/"
        )
        self.documents: list[dict[str, Any]] = []
        self.scraped_pages: set[str] = set()

        self.priority_keywords = [
            "products",
            "fund",
            "investment",
            "retirement",
            "pension",
            "service",
            "support",
            "help",
            "contact",
            "fees",
            "pricing",
            "faq",
            "insight",
            "market",
        ]
        self.deny_keywords = [
            "/login",
            "/sign-in",
            "/register",
            "/account",
            "/auth",
            "/cookie",
            "/privacy",
            "/terms",
            "/legal",
            "facebook.com",
            "linkedin.com",
            "instagram.com",
            "youtube.com",
            "twitter.com",
            "mailto:",
            "tel:",
        ]

    async def initialize(
        self,
        preload_website: bool = False,
        max_pages: int = 50,
        force_refresh: bool = False,
    ) -> None:
        await self._load_or_create_store()

        if preload_website and (force_refresh or not self.documents):
            if force_refresh:
                self.documents = []
                self.scraped_pages.clear()
            await self.scrape_website(max_pages=max_pages)

    async def _load_or_create_store(self) -> None:
        try:
            if self.data_path.exists():
                with open(self.data_path, encoding="utf-8") as f:
                    self.documents = json.load(f)
                return

            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            self.documents = []
            await self._save_store()
        except Exception as e:
            logger.error(f"Error loading store: {e}")
            self.documents = []

    async def _save_store(self) -> None:
        try:
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving store: {e}")

    async def add_document(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self.documents.append({"text": text, "metadata": metadata or {}})
        await self._save_store()

    async def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.documents:
            return []

        query_words = set(query.lower().split())
        # Build prefix keys for longer query words to catch STT mis-transcriptions.
        # e.g. "juniorized"[:6] == "junior" matches "junior isa" docs.
        prefix_len = 6
        query_prefixes = {w[:prefix_len] for w in query_words if len(w) > prefix_len}

        scored_docs: list[tuple[float, dict[str, Any]]] = []

        for doc in self.documents:
            doc_words = set(doc["text"].lower().split())
            # Exact keyword overlap (full weight)
            exact = query_words & doc_words
            score = float(len(exact))

            # Prefix overlap for unmatched words (half weight) — handles STT errors
            if query_prefixes:
                unmatched_prefixes = {
                    w[:prefix_len] for w in query_words - exact if len(w) > prefix_len
                }
                for dw in doc_words:
                    if dw[:prefix_len] in unmatched_prefixes:
                        score += 0.5

            if score > 0:
                scored_docs.append(
                    (
                        score,
                        {
                            "text": doc["text"],
                            "type": doc["metadata"].get("type", ""),
                            "url": doc["metadata"].get("url", ""),
                        },
                    )
                )

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    async def scrape_website(self, max_pages: int = 10) -> None:
        try:
            seed_urls = await self._build_seed_urls(max_pages=max_pages)
            urls_to_scrape = deque(seed_urls)
            queued_urls = set(seed_urls)
            scraped_count = 0

            while urls_to_scrape and scraped_count < max_pages:
                current_url = urls_to_scrape.popleft()
                if current_url in self.scraped_pages:
                    continue

                page_data = await self._scrape_page(current_url)
                if not page_data["text"]:
                    continue

                await self._add_web_content(current_url, page_data["text"])
                self.scraped_pages.add(current_url)
                scraped_count += 1

                new_urls = sorted(
                    page_data["links"],
                    key=lambda u: self._url_priority_score(u),
                    reverse=True,
                )
                for url in new_urls:
                    if (
                        url not in self.scraped_pages
                        and url not in queued_urls
                        and self._should_include_url(url)
                    ):
                        urls_to_scrape.append(url)
                        queued_urls.add(url)

            await self._save_store()
        except Exception as e:
            logger.error(f"Error during website scraping: {e}")

    async def _scrape_page(self, url: str) -> dict[str, Any]:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: requests.get(url, headers=headers, timeout=10)
            )
            if response.status_code != 200:
                return {"text": "", "links": []}

            soup = BeautifulSoup(response.content, "lxml")
            links: list[str] = []
            for link in soup.find_all("a", href=True):
                absolute_url = self._normalize_url(urljoin(url, link["href"]))
                if self._should_include_url(absolute_url):
                    links.append(absolute_url)

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(separator=" ", strip=True)
            text = " ".join(text.split())
            return {"text": text, "links": list(set(links))}
        except Exception as e:
            logger.error(f"Error scraping page {url}: {e}")
            return {"text": "", "links": []}

    async def _build_seed_urls(self, max_pages: int) -> list[str]:
        seeds: list[str] = [self._normalize_url(self.website_url)]

        sitemap_urls = await self._fetch_sitemap_urls(limit=max_pages * 3)
        if sitemap_urls:
            seeds.extend(
                sorted(
                    sitemap_urls,
                    key=lambda u: self._url_priority_score(u),
                    reverse=True,
                )
            )

        section_paths = [
            "products",
            "funds",
            "investments",
            "retirement",
            "pensions",
            "help",
            "support",
            "contact",
            "insights",
            "markets",
            "pricing",
            "fees",
            "faqs",
        ]
        seeds.extend(urljoin(self.website_url, path) for path in section_paths)

        unique: list[str] = []
        seen = set()
        for url in seeds:
            normalized = self._normalize_url(url)
            if normalized and normalized not in seen and self._should_include_url(normalized):
                seen.add(normalized)
                unique.append(normalized)

        return unique[: max(max_pages * 2, 20)]

    async def _fetch_sitemap_urls(self, limit: int = 200) -> list[str]:
        candidates = [
            urljoin(self.website_url, "sitemap.xml"),
            urljoin(self.website_url, "sitemap_index.xml"),
        ]

        urls: list[str] = []
        for sitemap_url in candidates:
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda sitemap_url=sitemap_url: requests.get(sitemap_url, timeout=10),
                )
                if response.status_code != 200:
                    continue

                root = ElementTree.fromstring(response.content)
                ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                loc_nodes = root.findall(".//sm:url/sm:loc", ns) or root.findall(".//url/loc")

                for node in loc_nodes:
                    if not node.text:
                        continue
                    normalized = self._normalize_url(node.text.strip())
                    if normalized and self._should_include_url(normalized):
                        urls.append(normalized)
                        if len(urls) >= limit:
                            return urls
            except Exception:
                continue

        return urls

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return ""

        path = parsed.path.rstrip("/")
        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"

        if parsed.query:
            query = parse_qs(parsed.query, keep_blank_values=True)
            keep = {k: query[k] for k in ("page", "p") if k in query}
            if keep:
                pairs = [f"{k}={v}" for k, values in keep.items() for v in values]
                if pairs:
                    normalized = f"{normalized}?{'&'.join(pairs)}"

        return normalized

    def _url_priority_score(self, url: str) -> int:
        lower = url.lower()
        score = sum(5 for keyword in self.priority_keywords if keyword in lower)
        if lower.count("/") <= 4:
            score += 2
        if "?" not in lower:
            score += 1
        return score

    def _should_include_url(self, url: str) -> bool:
        if not url:
            return False

        parsed = urlparse(url)
        base = urlparse(self.website_url)
        if parsed.netloc != base.netloc:
            return False

        lower = url.lower()
        return all(denied not in lower for denied in self.deny_keywords)

    async def _add_web_content(self, url: str, content: str) -> None:
        if not content or len(content.strip()) < 100:
            return

        chunks = self._split_text(content, max_length=1000)
        for i, chunk in enumerate(chunks):
            await self.add_document(
                text=chunk,
                metadata={
                    "type": "website_content",
                    "source": "website",
                    "url": url,
                    "chunk": i,
                },
            )

    def _split_text(self, text: str, max_length: int = 1000) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        current_chunk: list[str] = []

        for word in words:
            if len(" ".join([*current_chunk, word])) <= max_length:
                current_chunk.append(word)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
