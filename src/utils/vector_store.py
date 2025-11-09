from typing import List, Dict, Any
from pathlib import Path
import json
import logging
from .web_scraper import WebScraper

logger = logging.getLogger(__name__)

class WebEnabledVectorStore:
    def __init__(self, data_path: str = "data/knowledge_base.json", website_url: str = None):
        """Initialize the vector store with a path to the knowledge base file."""
        self.data_path = Path(data_path)
        self.resume_data_path = Path("data/resume_data.json")
        self.documents: List[Dict[str, Any]] = []
        self._load_or_create_store()

    def _load_or_create_store(self):
        """Load the knowledge base if it exists, otherwise create an empty one."""
        if self.data_path.exists():
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        else:
            # Create the directory if it doesn't exist
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            self.documents = []
            self._save_store()

    def _save_store(self):
        """Save the current state of the knowledge base."""
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2)

    def scrape_website(self, max_pages: int = 50):
        """Scrape the website and add content to the knowledge base."""
        if not self.scraper:
            logger.error("No website URL configured")
            return

        try:
            # Clear existing documents to ensure fresh content
            self.documents = []
            
            pages = self.scraper.scrape_site(max_pages=max_pages)
            for page in pages:
                # Add title and description
                if page['title']:
                    self.add_document(
                        text=page['title'],
                        metadata={'type': 'title', 'url': page['url']}
                    )
                if page['metadata'].get('description'):
                    self.add_document(
                        text=page['metadata']['description'],
                        metadata={'type': 'description', 'url': page['url']}
                    )

                # Add main content
                for heading in page['headings']:
                    self.add_document(
                        text=heading,
                        metadata={'type': 'heading', 'url': page['url']}
                    )
                
                # Add paragraphs and table content
                for paragraph in page['paragraphs']:
                    if len(paragraph.split()) > 3:  # Include more content
                        self.add_document(
                            text=paragraph,
                            metadata={'type': 'content', 'url': page['url']}
                        )
                
                # Add table content if available
                if 'tables' in page:
                    for table_row in page['tables']:
                        self.add_document(
                            text=table_row,
                            metadata={'type': 'table', 'url': page['url']}
                        )

            logger.info(f"Successfully scraped {len(pages)} pages from {self.website_url}")
        except Exception as e:
            logger.error(f"Error scraping website: {str(e)}")

    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """Add a document to the knowledge base with optional metadata."""
        doc = {
            "text": text,
            "metadata": metadata or {}
        }
        self.documents.append(doc)
        self._save_store()

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant information.
        Returns a list of dicts containing text and source URL.
        """
        if not self.documents:
            logger.warning("Knowledge base is empty - no documents to search")
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
                        "url": doc["metadata"].get("url", ""),
                        "type": doc["metadata"].get("type", "")
                    }
                ))
        
        # Sort by score (highest first) and return top k results
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]
        
        # Sort by score and get top_k results
        scored_docs.sort(reverse=True)
        results = [doc for score, doc in scored_docs[:top_k] if score > 0]
        
        return results