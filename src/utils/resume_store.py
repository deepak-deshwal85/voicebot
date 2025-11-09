from typing import List, Dict, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class ResumeVectorStore:
    def __init__(self, data_path: str = "data/knowledge_base.json"):
        """Initialize the vector store with a path to the knowledge base file."""
        self.data_path = Path(data_path)
        self.resume_data_path = Path("data/resume_data.json")
        self.documents: List[Dict[str, Any]] = []

    async def initialize(self):
        """Asynchronously initialize the vector store."""
        await self._load_or_create_store()
        await self.load_resume_data()

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

    async def load_resume_data(self):
        """Load the static resume data into the knowledge base."""
        try:
            with open(self.resume_data_path, 'r', encoding='utf-8') as f:
                resume_data = json.load(f)

            # Clear existing documents
            self.documents = []

            # Add personal info
            personal = resume_data["personal_info"]
            self.add_document(
                text=f"{personal['name']} - {personal['title']}",
                metadata={"type": "header"}
            )
            self.add_document(
                text=f"Contact: {personal['email']} | {personal['phone']} | {personal['location']}",
                metadata={"type": "contact"}
            )
            self.add_document(
                text=personal['summary'],
                metadata={"type": "summary"}
            )

            # Add education
            for edu in resume_data["education"]:
                self.add_document(
                    text=f"{edu['degree']} from {edu['institution']}, {edu['location']} ({edu['year']})",
                    metadata={"type": "education"}
                )

            # Add experience
            for exp in resume_data["experience"]:
                self.add_document(
                    text=f"{exp['title']} at {exp['company']}, {exp['location']} ({exp['period']})",
                    metadata={"type": "experience_header"}
                )
                for resp in exp["responsibilities"]:
                    self.add_document(
                        text=resp,
                        metadata={"type": "experience_detail"}
                    )

            # Add skills
            for category, skills in resume_data["skills"].items():
                self.add_document(
                    text=f"{category.replace('_', ' ').title()}: {', '.join(skills)}",
                    metadata={"type": "skills"}
                )

            # Add projects
            for project in resume_data["projects"]:
                self.add_document(
                    text=f"Project: {project['name']} - {project['description']}",
                    metadata={"type": "project"}
                )
                self.add_document(
                    text=f"Technologies: {', '.join(project['technologies'])}",
                    metadata={"type": "project_tech"}
                )

            # Add certifications
            for cert in resume_data["certifications"]:
                self.add_document(
                    text=f"{cert['name']} ({cert['issuer']}, {cert['year']})",
                    metadata={"type": "certification"}
                )

            # Add languages
            for lang in resume_data["languages"]:
                self.add_document(
                    text=f"{lang['language']}: {lang['proficiency']}",
                    metadata={"type": "language"}
                )

            logger.info("Successfully loaded resume data into knowledge base")
            self._save_store()
            
        except Exception as e:
            logger.error(f"Error loading resume data: {str(e)}")
            raise

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
        Returns a list of dicts containing text and metadata.
        """
        if not self.documents:
            logger.warning("Knowledge base is empty - loading resume data")
            self.load_resume_data()

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