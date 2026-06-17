from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
TENANT_MAP_PATH = CONFIG_DIR / "tenant-map.json"
DEFAULT_CLIENT_ID = "client-1"


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


def list_client_ids() -> list[str]:
    if not CONFIG_DIR.exists():
        return []
    return sorted(path.stem for path in CONFIG_DIR.glob("*.properties"))


def client_properties_path(client_id: str) -> Path:
    path = CONFIG_DIR / f"{client_id}.properties"
    if not path.exists():
        available = ", ".join(list_client_ids()) or "none"
        raise ValueError(f"Unknown client '{client_id}'. Available: {available}")
    return path


def client_resume_knowledge_path(client_id: str) -> Path:
    return CONFIG_DIR / f"{client_id}-resume.json"


def load_tenant_map() -> dict[str, str]:
    """Return phone digits -> client_id mapping from config/tenant-map.json."""
    if not TENANT_MAP_PATH.exists():
        return {}

    payload = json.loads(TENANT_MAP_PATH.read_text(encoding="utf-8"))
    index: dict[str, str] = {}
    for phone, client_id in payload.items():
        digits = "".join(ch for ch in str(phone) if ch.isdigit())
        if digits:
            index[digits] = str(client_id)
    return index


def load_tenant_map_by_client() -> dict[str, str]:
    """Return client_id -> phone number from config/tenant-map.json."""
    if not TENANT_MAP_PATH.exists():
        return {}

    payload = json.loads(TENANT_MAP_PATH.read_text(encoding="utf-8"))
    return {str(client_id): str(phone) for phone, client_id in payload.items()}


@dataclass(frozen=True)
class WorkerSettings:
    agent_name: str
    default_client_id: str


def load_worker_settings() -> WorkerSettings:
    return WorkerSettings(
        agent_name=os.getenv("AGENT_NAME", "telephone-agent"),
        default_client_id=os.getenv("DEFAULT_CLIENT_ID", DEFAULT_CLIENT_ID),
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class KnowledgePreloadSettings:
    enabled: bool


def load_knowledge_preload_settings() -> KnowledgePreloadSettings:
    return KnowledgePreloadSettings(
        enabled=_env_bool("PRELOAD_RESUME_KNOWLEDGE", True),
    )


@dataclass(frozen=True)
class AgentConfig:
    client_id: str
    display_name: str
    initial_greeting: str
    instructions: str
    no_results_message: str
    knowledge_not_ready_message: str
    invalid_search_query_message: str
    resume_knowledge_path: Path
    properties_path: Path
    resume_folder: Path
    chunk_size: int
    embedding_model: str

    def format(self, template: str) -> str:
        return template.format(display_name=self.display_name)


def load_agent_config(
    client_id: str | None = None,
    properties_path: Path | None = None,
) -> AgentConfig:
    if properties_path is not None:
        path = (
            properties_path
            if properties_path.is_absolute()
            else PROJECT_ROOT / properties_path
        )
        resolved_client_id = client_id or path.stem
    else:
        resolved_client_id = client_id or os.getenv(
            "DEFAULT_CLIENT_ID", DEFAULT_CLIENT_ID
        )
        path = client_properties_path(resolved_client_id)

    raw = _parse_properties(path.read_text(encoding="utf-8"))
    client_id = raw.get("client.id", resolved_client_id)

    display_name = raw.get("agent.display_name", "Resume Assistant")
    template_values = {"display_name": display_name}

    def fmt(key: str, default: str) -> str:
        return raw.get(key, default).format(**template_values)

    return AgentConfig(
        client_id=client_id,
        display_name=display_name,
        initial_greeting=fmt(
            "agent.initial_greeting",
            "Hello! I can answer questions about the resume on file. What would you like to know?",
        ),
        instructions=fmt(
            "agent.instructions",
            "You are a helpful voice assistant for {display_name}.",
        ),
        no_results_message=fmt(
            "agent.no_results_message",
            "I don't have that information in the resume.",
        ),
        knowledge_not_ready_message=raw.get(
            "agent.knowledge_not_ready_message",
            "I'm sorry, but my knowledge base is not ready yet. Please try again in a moment.",
        ),
        invalid_search_query_message=raw.get(
            "agent.invalid_search_query_message",
            "Please ask a complete question with a few words so I can search accurately.",
        ),
        resume_knowledge_path=client_resume_knowledge_path(client_id),
        properties_path=path,
        resume_folder=_resolve_path(
            raw.get("knowledge.resume_folder", f"knowledge-sources/{client_id}")
        ),
        chunk_size=int(raw.get("knowledge.chunk_size", "1000")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            raw.get("knowledge.embedding_model", "text-embedding-3-small"),
        ),
    )
