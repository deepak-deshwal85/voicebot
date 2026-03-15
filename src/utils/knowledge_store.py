import asyncio
import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class KnowledgeStore:
    def __init__(self, data_path: str = "data/knowledge_base.json"):
        """Initialize the vector store with a path to the knowledge base file."""
        self.data_path = Path(data_path)
        self.website_url = os.getenv(
            "KNOWLEDGE_WEBSITE_URL", "https://www.fidelityinternational.com/"
        )
        self.documents: list[dict[str, Any]] = []
        self.scraped_pages: set = set()  # Track scraped URLs to avoid duplicates
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
    ):
        """Asynchronously initialize the vector store with optional website pre-loading."""
        logger.info("Initializing vector store...")

        # Load existing knowledge base
        await self._load_or_create_store()

        # Pre-load website content if requested
        if preload_website and (force_refresh or not self.documents):
            if force_refresh:
                self.documents = []
                self.scraped_pages.clear()
            logger.info(f"Pre-loading website content from {self.website_url}...")
            await self.scrape_website(max_pages=max_pages)
            logger.info("Website pre-loading completed")

        logger.info(f"Vector store initialized with {len(self.documents)} documents")

    async def _load_or_create_store(self):
        """Load the knowledge base if it exists, otherwise create an empty one."""
        try:
            if self.data_path.exists():
                with open(self.data_path, encoding="utf-8") as f:
                    self.documents = json.load(f)
            else:
                # Create the directory if it doesn't exist
                self.data_path.parent.mkdir(parents=True, exist_ok=True)
                self.documents = []
                await self._save_store()
        except Exception as e:
            logger.error(f"Error loading store: {e}")
            self.documents = []

    async def _save_store(self):
        """Save the current state of the knowledge base."""
        try:
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving store: {e}")

    async def add_document(self, text: str, metadata: dict[str, Any] | None = None):
        """Add a document to the knowledge base with optional metadata."""
        doc = {"text": text, "metadata": metadata or {}}
        self.documents.append(doc)
        await self._save_store()

    async def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Search the knowledge base for relevant information.
        Returns a list of dicts containing text and metadata.
        """
        if not self.documents:
            logger.warning("Knowledge base is empty - no website content loaded")
            return []

        query_words = set(query.lower().split())

        # Score documents based on word overlap
        scored_docs = []
        for doc in self.documents:
            doc_words = set(doc["text"].lower().split())
            score = len(query_words.intersection(doc_words))
            if score > 0:  # Only include documents with some relevance
                scored_docs.append(
                    (
                        score,
                        {"text": doc["text"], "type": doc["metadata"].get("type", "")},
                    )
                )

        # Sort by score (highest first) and return top k results
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

    async def search_with_fallback(
        self, query: str, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """
        Search the website content for relevant information.
        """
        # Keep runtime queries low-latency: search only preloaded/local content.
        return await self.search(query, top_k)

    async def search_website(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Search the Fidelity International website for relevant information.
        """
        try:
            # Check if we can scrape according to robots.txt
            if not await self._can_scrape_website():
                logger.warning("Cannot scrape website according to robots.txt")
                return []

            # Scrape main pages
            await self.scrape_website(max_pages=10)

            # Search through scraped content
            query_words = set(query.lower().split())
            scored_docs = []

            for doc in self.documents:
                if doc["metadata"].get("source") == "website":
                    doc_words = set(doc["text"].lower().split())
                    score = len(query_words.intersection(doc_words))
                    if score > 0:
                        scored_docs.append(
                            (
                                score,
                                {
                                    "text": doc["text"],
                                    "type": doc["metadata"].get("type", ""),
                                    "source": "website",
                                },
                            )
                        )

            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return [doc for score, doc in scored_docs[:top_k]]

        except Exception as e:
            logger.error(f"Error searching website: {e}")
            return []

    async def _can_scrape_website(self) -> bool:
        """Check if we can scrape the website according to robots.txt."""
        try:
            robots_url = urljoin(self.website_url, "/robots.txt")
            rp = RobotFileParser()
            rp.set_url(robots_url)
            await asyncio.get_event_loop().run_in_executor(None, rp.read)
            return rp.can_fetch("*", self.website_url)
        except Exception as e:
            logger.warning(f"Could not check robots.txt: {e}")
            return True  # Default to allowing if we can't check

    async def scrape_website(self, max_pages: int = 10):
        """Scrape the Fidelity International website for content."""
        try:
            logger.info(f"Starting website scraping of {self.website_url}")

            seed_urls = await self._build_seed_urls(max_pages=max_pages)
            urls_to_scrape = deque(seed_urls)
            queued_urls = set(seed_urls)
            scraped_count = 0

            while urls_to_scrape and scraped_count < max_pages:
                current_url = urls_to_scrape.popleft()

                if current_url in self.scraped_pages:
                    continue

                try:
                    # Scrape the page
                    page_data = await self._scrape_page(current_url)
                    if page_data and page_data["text"]:
                        await self._add_web_content(current_url, page_data["text"])
                        self.scraped_pages.add(current_url)
                        scraped_count += 1

                        # Extract new URLs to scrape (limit to same domain)
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

                except Exception as e:
                    logger.error(f"Error scraping {current_url}: {e}")
                    continue

            logger.info(f"Scraped {scraped_count} pages from website")
            await self._save_store()

        except Exception as e:
            logger.error(f"Error during website scraping: {e}")

    async def _scrape_page(self, url: str) -> dict[str, Any]:
        """Scrape a single page and return cleaned text + discoverable links."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: requests.get(url, headers=headers, timeout=10)
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "lxml")

                links = []
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    absolute_url = self._normalize_url(urljoin(url, href))
                    if self._should_include_url(absolute_url):
                        links.append(absolute_url)

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text content
                text = soup.get_text(separator=" ", strip=True)

                # Clean up whitespace
                import re

                text = re.sub(r"\s+", " ", text).strip()

                return {
                    "text": text,
                    "links": list(set(links)),
                }
            else:
                logger.warning(f"Failed to scrape {url}: HTTP {response.status_code}")
                return {"text": "", "links": []}

        except Exception as e:
            logger.error(f"Error scraping page {url}: {e}")
            return {"text": "", "links": []}

    async def _build_seed_urls(self, max_pages: int) -> list[str]:
        """Build prioritized seed URLs from sitemap + tree-style section paths."""
        seeds: list[str] = [self._normalize_url(self.website_url)]

        sitemap_urls = await self._fetch_sitemap_urls(limit=max_pages * 3)
        if sitemap_urls:
            prioritized_sitemap = sorted(
                sitemap_urls,
                key=lambda u: self._url_priority_score(u),
                reverse=True,
            )
            seeds.extend(prioritized_sitemap)

        seeds.extend(self._build_section_seed_urls())

        unique: list[str] = []
        seen = set()
        for url in seeds:
            normalized = self._normalize_url(url)
            if (
                normalized
                and normalized not in seen
                and self._should_include_url(normalized)
            ):
                seen.add(normalized)
                unique.append(normalized)

        return unique[: max(max_pages * 2, 20)]

    def _build_section_seed_urls(self) -> list[str]:
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
        return [urljoin(self.website_url, path) for path in section_paths]

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
                    lambda sitemap_url=sitemap_url: requests.get(
                        sitemap_url, timeout=10
                    ),
                )
                if response.status_code != 200:
                    continue

                root = ElementTree.fromstring(response.content)
                ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

                loc_nodes = root.findall(".//sm:url/sm:loc", ns)
                if not loc_nodes:
                    loc_nodes = root.findall(".//url/loc")

                for node in loc_nodes:
                    if node.text:
                        normalized = self._normalize_url(node.text.strip())
                        if normalized and self._should_include_url(normalized):
                            urls.append(normalized)
                            if len(urls) >= limit:
                                return urls
            except Exception as e:
                logger.debug(f"Sitemap parse failed for {sitemap_url}: {e}")
                continue

        return urls

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return ""

        path = parsed.path.rstrip("/")
        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"

        # Keep only pagination query params; drop tracking params
        if parsed.query:
            query = parse_qs(parsed.query, keep_blank_values=True)
            keep = {}
            for key in ("page", "p"):
                if key in query:
                    keep[key] = query[key]
            if keep:
                pairs = []
                for key, values in keep.items():
                    for val in values:
                        pairs.append(f"{key}={val}")
                if pairs:
                    normalized = f"{normalized}?{'&'.join(pairs)}"

        return normalized

    def _url_priority_score(self, url: str) -> int:
        lower = url.lower()
        score = 0
        for keyword in self.priority_keywords:
            if keyword in lower:
                score += 5
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

    def _extract_urls(self, content: str, base_url: str) -> list[str]:
        """Extract URLs from HTML content."""
        urls = []
        try:
            soup = BeautifulSoup(content, "lxml")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                absolute_url = urljoin(base_url, href)

                # Only include URLs from the same domain
                if urlparse(absolute_url).netloc == urlparse(self.website_url).netloc:
                    urls.append(absolute_url)

        except Exception as e:
            logger.error(f"Error extracting URLs: {e}")

        return list(set(urls))  # Remove duplicates

    async def _add_web_content(self, url: str, content: str):
        """Add scraped web content to the knowledge base."""
        if not content or len(content.strip()) < 100:  # Skip very short content
            return

        # Split content into manageable chunks
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
        """Split text into chunks of maximum length."""
        words = text.split()
        chunks = []
        current_chunk = []

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
