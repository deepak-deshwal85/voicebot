from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIENTS_DIR = PROJECT_ROOT / "config" / "clients"
DEFAULT_CLIENT_ID = "client-1"
DEFAULT_PROPERTIES_PATH = CLIENTS_DIR / DEFAULT_CLIENT_ID / "agent.properties"


def _unescape(value: str) -> str:
    return value.replace("\\n", "\n").strip()


def parse_properties(content: str) -> dict[str, str]:
    return _parse_properties(content)


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


def read_client_properties(client_id: str) -> dict[str, str]:
    path = resolve_client_config_path(client_id)
    return _parse_properties(path.read_text(encoding="utf-8"))


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def list_client_ids() -> list[str]:
    if not CLIENTS_DIR.exists():
        return []
    return sorted(
        path.name
        for path in CLIENTS_DIR.iterdir()
        if path.is_dir() and (path / "agent.properties").exists()
    )


def resolve_client_config_path(client_id: str) -> Path:
    path = CLIENTS_DIR / client_id / "agent.properties"
    if not path.exists():
        available = ", ".join(list_client_ids()) or "none"
        raise ValueError(
            f"Unknown client '{client_id}'. Available clients: {available}"
        )
    return path


def resolve_config_path(
    client_id: str | None = None,
    properties_path: Path | None = None,
) -> Path:
    if properties_path is not None:
        return (
            properties_path
            if properties_path.is_absolute()
            else PROJECT_ROOT / properties_path
        )

    if client_id:
        return resolve_client_config_path(client_id)

    env_client = os.getenv("CLIENT_ID")
    if env_client:
        return resolve_client_config_path(env_client)

    env_path = os.getenv("AGENT_CONFIG_PATH")
    if env_path:
        path = Path(env_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    return DEFAULT_PROPERTIES_PATH


@dataclass(frozen=True)
class WorkerSettings:
    agent_name: str
    default_client_id: str


def load_worker_settings() -> WorkerSettings:
    return WorkerSettings(
        agent_name=os.getenv("AGENT_NAME", "telephone-agent"),
        default_client_id=os.getenv("DEFAULT_CLIENT_ID", DEFAULT_CLIENT_ID),
    )


@dataclass(frozen=True)
class AgentConfig:
    client_id: str
    agent_name: str
    telephony_phone_number: str | None
    website_name: str
    website_url: str
    initial_greeting: str
    instructions: str
    no_results_message: str
    knowledge_not_ready_message: str
    knowledge_website_path: Path
    knowledge_pdfs_path: Path
    legacy_knowledge_path: Path
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
    client_id: str | None = None,
    properties_path: Path | None = None,
    overrides: dict[str, str] | None = None,
) -> AgentConfig:
    path = resolve_config_path(client_id=client_id, properties_path=properties_path)

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

    resolved_client_id = raw.get(
        "client.id", client_id or os.getenv("CLIENT_ID", DEFAULT_CLIENT_ID)
    )
    agent_name = raw.get("agent.name", f"{resolved_client_id}-voice-agent")

    legacy_path = _resolve_path(
        raw.get(
            "knowledge.legacy_path",
            f"data/clients/{resolved_client_id}/knowledge_base.json",
        )
    )

    telephony_phone = raw.get("telephony.phone_number", "").strip() or None

    return AgentConfig(
        client_id=resolved_client_id,
        agent_name=agent_name,
        telephony_phone_number=telephony_phone,
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
        knowledge_website_path=_resolve_path(
            raw.get(
                "knowledge.website_path",
                f"data/clients/{resolved_client_id}/knowledge_website.json",
            )
        ),
        knowledge_pdfs_path=_resolve_path(
            raw.get(
                "knowledge.pdfs_path",
                f"data/clients/{resolved_client_id}/knowledge_pdfs.json",
            )
        ),
        legacy_knowledge_path=legacy_path,
        pdf_folder=_resolve_path(
            raw.get(
                "knowledge.pdf_folder",
                f"data/clients/{resolved_client_id}",
            )
        ),
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


def load_store_metadata(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": "invalid_json"}

    if isinstance(payload, dict):
        return payload
    return {"documents": payload}
