from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROPERTIES_PATH = PROJECT_ROOT / "config" / "agent.properties"


def _unescape(value: str) -> str:
    return value.replace("\\n", "\n").strip()


def _parse_properties(content: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        properties[key.strip()] = _unescape(value.strip())
    return properties


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclass(frozen=True)
class AgentConfig:
    website_name: str
    website_url: str
    initial_greeting: str
    instructions: str
    no_results_message: str
    knowledge_not_ready_message: str
    knowledge_data_path: Path
    pdf_folder: Path
    max_pages: int
    chunk_size: int
    runtime_scraping_enabled: bool
    embedding_model: str
    properties_path: Path

    def format(self, template: str) -> str:
        return template.format(
            website_name=self.website_name,
            website_url=self.website_url,
        )


def load_agent_config(
    properties_path: Path | None = None,
    overrides: dict[str, str] | None = None,
) -> AgentConfig:
    path = properties_path or Path(
        os.getenv("AGENT_CONFIG_PATH", DEFAULT_PROPERTIES_PATH)
    )
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    raw: dict[str, str] = {}
    if path.exists():
        raw = _parse_properties(path.read_text(encoding="utf-8"))
    if overrides:
        raw.update(overrides)

    website_name = raw.get("website.name", "Your Company")
    website_url = raw.get("website.url", "https://example.com/")
    template_values = {
        "website_name": website_name,
        "website_url": website_url,
    }

    def fmt(key: str, default: str) -> str:
        return raw.get(key, default).format(**template_values)

    return AgentConfig(
        website_name=website_name,
        website_url=website_url,
        initial_greeting=fmt(
            "agent.initial_greeting",
            "Hello! I can help you learn about {website_name}. What would you like to know?",
        ),
        instructions=fmt(
            "agent.instructions",
            "You are a helpful voice AI assistant for {website_name}.",
        ),
        no_results_message=fmt(
            "agent.no_results_message",
            "I don't have that information for {website_name}.",
        ),
        knowledge_not_ready_message=raw.get(
            "agent.knowledge_not_ready_message",
            "I'm sorry, but my knowledge base is not ready yet. Please try again in a moment.",
        ),
        knowledge_data_path=_resolve_path(
            raw.get("knowledge.data_path", "data/knowledge_base.json")
        ),
        pdf_folder=_resolve_path(raw.get("knowledge.pdf_folder", "data")),
        max_pages=int(raw.get("knowledge.max_pages", "100")),
        chunk_size=int(raw.get("knowledge.chunk_size", "1000")),
        runtime_scraping_enabled=raw.get(
            "knowledge.runtime_scraping_enabled", "false"
        ).lower()
        in {"1", "true", "yes", "on"},
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            raw.get("knowledge.embedding_model", "text-embedding-3-small"),
        ),
        properties_path=path,
    )
