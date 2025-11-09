from typing import List, Dict, Any
from pathlib import Path
import json
import logging
from docx import Document

logger = logging.getLogger(__name__)

class ResumeVectorStore:
    def __init__(self, data_path: str = "data/knowledge_base.json"):
        """Initialize the vector store with a path to the knowledge base file."""
        self.data_path = Path(data_path)
        self.resume_docx_path = Path("DEEPAK.docx")
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
        """Load the resume data from the Word document."""
        try:
            if not self.resume_docx_path.exists():
                logger.error(f"Resume document not found: {self.resume_docx_path}")
                return

            doc = Document(str(self.resume_docx_path))

            # Clear existing documents
            self.documents = []

            # Parse the document content
            content_lines = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    content_lines.append(text)

            # Extract personal information
            self._extract_personal_info(content_lines)

            # Extract profile summary
            self._extract_profile_summary(content_lines)

            # Extract professional experience
            self._extract_professional_experience(content_lines)

            # Extract academic details
            self._extract_academic_details(content_lines)

            # Extract projects
            self._extract_projects(content_lines)

            # Extract technical skills
            self._extract_technical_skills(content_lines)

            # Save the processed data
            await self._save_store()

            logger.info("Successfully loaded resume data from Word document")

        except Exception as e:
            logger.error(f"Error loading resume data from Word document: {e}")
            self.documents = []

    def _extract_personal_info(self, content_lines: List[str]):
        """Extract personal information from the document."""
        for line in content_lines[:10]:  # Check first few lines
            if 'Deepak' in line and 'Kumar' in line:
                self.add_document(
                    text=f"Name: {line.strip()}",
                    metadata={"type": "personal", "category": "name"}
                )
            elif 'Mobile:' in line or 'Phone:' in line:
                self.add_document(
                    text=f"Contact: {line.strip()}",
                    metadata={"type": "personal", "category": "phone"}
                )
            elif 'E-Mail:' in line or 'Email:' in line:
                self.add_document(
                    text=f"Email: {line.strip()}",
                    metadata={"type": "personal", "category": "email"}
                )
            elif 'Team Lead' in line or any(title in line for title in ['Software Engineer', 'Senior Software Engineer']):
                self.add_document(
                    text=f"Current Position: {line.strip()}",
                    metadata={"type": "personal", "category": "title"}
                )

    def _extract_profile_summary(self, content_lines: List[str]):
        """Extract profile summary from the document."""
        in_summary = False
        summary_lines = []

        for line in content_lines:
            if 'PROFILE SUMMARY' in line:
                in_summary = True
                continue
            elif in_summary and line.startswith(('PROFESSIONAL EXPERIENCE', 'TECHINAL PURVIEW', 'ACADEMIC DETAILS')):
                break
            elif in_summary:
                summary_lines.append(line)

        if summary_lines:
            summary_text = ' '.join(summary_lines)
            self.add_document(
                text=f"Profile Summary: {summary_text}",
                metadata={"type": "summary"}
            )

    def _extract_professional_experience(self, content_lines: List[str]):
        """Extract professional experience from the document."""
        in_experience = False

        for line in content_lines:
            if 'PROFESSIONAL EXPERIENCE' in line:
                in_experience = True
                continue
            elif in_experience and line.startswith(('ACADEMIC DETAILS', 'PROJECTS')):
                break
            elif in_experience and ('Working in' in line or 'from' in line and 'to' in line):
                self.add_document(
                    text=f"Experience: {line.strip()}",
                    metadata={"type": "experience"}
                )

    def _extract_academic_details(self, content_lines: List[str]):
        """Extract academic details from the document."""
        in_academic = False

        for line in content_lines:
            if 'ACADEMIC DETAILS' in line:
                in_academic = True
                continue
            elif in_academic and line.startswith(('PROJECTS', 'PERSONAL DETAILS')):
                break
            elif in_academic and ('MCA' in line or 'BSc' in line or '10+' in line or '10th' in line):
                self.add_document(
                    text=f"Education: {line.strip()}",
                    metadata={"type": "education"}
                )

    def _extract_projects(self, content_lines: List[str]):
        """Extract project information from the document."""
        in_projects = False
        current_project = []

        for line in content_lines:
            if 'PROJECTS' in line:
                in_projects = True
                continue
            elif in_projects and line.startswith('PERSONAL DETAILS'):
                break
            elif in_projects:
                if 'Fidelity International' in line or 'Genpact Headstrong' in line or 'NexGen Consultancy' in line:
                    # Save previous project if exists
                    if current_project:
                        project_text = ' '.join(current_project)
                        self.add_document(
                            text=f"Project: {project_text}",
                            metadata={"type": "project"}
                        )
                    current_project = [line]
                elif current_project:
                    current_project.append(line)

        # Save last project
        if current_project:
            project_text = ' '.join(current_project)
            self.add_document(
                text=f"Project: {project_text}",
                metadata={"type": "project"}
            )

    def _extract_technical_skills(self, content_lines: List[str]):
        """Extract technical skills from the document."""
        # Look for technologies mentioned in projects
        technologies = set()

        for line in content_lines:
            if 'Technologies Used:' in line:
                continue
            elif 'Database:' in line or 'ORM/Framework:' in line or 'Language:' in line or 'Tools:' in line:
                tech_info = line.split(':', 1)
                if len(tech_info) > 1:
                    tech_list = tech_info[1].strip()
                    technologies.add(tech_list)

        if technologies:
            tech_text = ', '.join(sorted(technologies))
            self.add_document(
                text=f"Technical Skills: {tech_text}",
                metadata={"type": "skills"}
            )

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