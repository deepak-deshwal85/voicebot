from typing import List, Dict, Any
from pathlib import Path
import json
import logging
import requests
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)

class KnowledgeStore:
    def __init__(self, data_path: str = "data/knowledge_base.json"):
        """Initialize the vector store with a path to the knowledge base file."""
        self.data_path = Path(data_path)
        self.website_url = "https://www.fidelity.co.uk/"
        self.documents: List[Dict[str, Any]] = []
        self.scraped_pages: set = set()  # Track scraped URLs to avoid duplicates

    async def initialize(self, preload_website: bool = False, max_pages: int = 50):
        """Asynchronously initialize the vector store with optional website pre-loading."""
        logger.info("Initializing vector store...")

        # Load existing knowledge base
        await self._load_or_create_store()

        # Pre-load website content if requested and not already loaded
        if preload_website and not self.documents:
            logger.info(f"Pre-loading website content from {self.website_url}...")
            await self.scrape_website(max_pages=max_pages)
            logger.info("Website pre-loading completed")

        logger.info(f"Vector store initialized with {len(self.documents)} documents")

    async def _load_or_create_store(self):
        """Load the knowledge base if it exists, otherwise create an empty one."""
        try:
            if self.data_path.exists():
                with open(self.data_path, 'r', encoding='utf-8') as f:
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
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving store: {e}")

    async def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """Add a document to the knowledge base with optional metadata."""
        doc = {
            "text": text,
            "metadata": metadata or {}
        }
        self.documents.append(doc)
        await self._save_store()

    async def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
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
                scored_docs.append((
                    score,
                    {
                        "text": doc["text"],
                        "type": doc["metadata"].get("type", "")
                    }
                ))

        # Sort by score (highest first) and return top k results
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

    async def search_with_fallback(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search the website content for relevant information.
        """
        # Search existing website content first
        results = await self.search(query, top_k)

        if not results:
            logger.info(f"No existing content found for '{query}', attempting fresh web search...")
            # If no results from existing content, try fresh web scraping
            web_results = await self.search_website(query, top_k)
            if web_results:
                return web_results

        return results

    async def search_website(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
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
                        scored_docs.append((
                            score,
                            {
                                "text": doc["text"],
                                "type": doc["metadata"].get("type", ""),
                                "source": "website"
                            }
                        ))

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

            # Start with the main page
            urls_to_scrape = [self.website_url]
            scraped_count = 0

            while urls_to_scrape and scraped_count < max_pages:
                current_url = urls_to_scrape.pop(0)

                if current_url in self.scraped_pages:
                    continue

                try:
                    # Scrape the page
                    page_content = await self._scrape_page(current_url)
                    if page_content:
                        await self._add_web_content(current_url, page_content)
                        self.scraped_pages.add(current_url)
                        scraped_count += 1

                        # Extract new URLs to scrape (limit to same domain)
                        new_urls = self._extract_urls(page_content, current_url)
                        for url in new_urls:
                            if url not in self.scraped_pages and len(urls_to_scrape) < max_pages * 2:
                                urls_to_scrape.append(url)

                except Exception as e:
                    logger.error(f"Error scraping {current_url}: {e}")
                    continue

            logger.info(f"Scraped {scraped_count} pages from website")
            await self._save_store()

        except Exception as e:
            logger.error(f"Error during website scraping: {e}")

    async def _scrape_page(self, url: str) -> str:
        """Scrape a single page and return its text content."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: requests.get(url, headers=headers, timeout=10)
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'lxml')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text content
                text = soup.get_text(separator=' ', strip=True)

                # Clean up whitespace
                import re
                text = re.sub(r'\s+', ' ', text).strip()

                return text
            else:
                logger.warning(f"Failed to scrape {url}: HTTP {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"Error scraping page {url}: {e}")
            return ""

    def _extract_urls(self, content: str, base_url: str) -> List[str]:
        """Extract URLs from HTML content."""
        urls = []
        try:
            soup = BeautifulSoup(content, 'lxml')
            for link in soup.find_all('a', href=True):
                href = link['href']
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
                    "chunk": i
                }
            )

    def _split_text(self, text: str, max_length: int = 1000) -> List[str]:
        """Split text into chunks of maximum length."""
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            if len(' '.join(current_chunk + [word])) <= max_length:
                current_chunk.append(word)
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks