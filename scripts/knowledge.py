#!/usr/bin/env python3
"""Build and validate per-client resume knowledge bases under config/."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

from dotenv import load_dotenv  # noqa: E402
from knowledge_builder import KnowledgeBuilder  # noqa: E402

from utils.config import list_client_ids, load_agent_config  # noqa: E402
from utils.knowledge_store import KnowledgeStore  # noqa: E402


def _load_env() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.local", override=True)


def _clients_arg(clients: list[str] | None) -> list[str]:
    return clients or list_client_ids()


async def cmd_build(client_id: str) -> None:
    config = load_agent_config(client_id=client_id)
    output = await KnowledgeBuilder(config).build()
    print(f"Built resume knowledge: {output}")


async def cmd_validate(client_id: str) -> bool:
    config = load_agent_config(client_id=client_id)
    issues: list[str] = []

    if not config.resume_knowledge_path.exists():
        issues.append(f"Missing resume knowledge file for {client_id}")

    store = KnowledgeStore(config)
    await store.initialize()
    await store.ensure_loaded()

    chunk_count = len(store.documents)
    if chunk_count == 0:
        issues.append("Resume knowledge base is empty")

    has_embeddings = any(doc.get("embedding") for doc in store.documents)
    if chunk_count and not has_embeddings:
        issues.append("No embeddings found; rebuild with OPENAI_API_KEY set")

    if chunk_count:
        missing_sections = sum(
            1 for doc in store.documents if not doc.get("metadata", {}).get("section")
        )
        missing_search_text = sum(1 for doc in store.documents if not doc.get("search_text"))
        if missing_sections:
            issues.append(
                f"{missing_sections} chunk(s) missing section metadata; rebuild knowledge base"
            )
        if missing_search_text:
            issues.append(
                f"{missing_search_text} chunk(s) missing search_text; rebuild knowledge base"
            )

    print(f"Client: {client_id}")
    print(f"Resume knowledge: {config.resume_knowledge_path} ({chunk_count} chunks)")

    if issues:
        print("Issues:")
        for issue in issues:
            print(f"- {issue}")
        return False

    print("Validation passed.")
    return True


async def cmd_search(client_id: str, query: str, top_k: int) -> None:
    config = load_agent_config(client_id=client_id)
    store = KnowledgeStore(config)
    await store.initialize()
    results = await store.search(query, top_k=top_k)
    if not results:
        print("No results.")
        return
    print(results)


def main() -> int:
    _load_env()

    parser = argparse.ArgumentParser(
        description="Manage config/{client}-resume.json knowledge bases"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-clients", help="List configured clients")

    build = sub.add_parser(
        "build", help="Build resume knowledge into config/{client}-resume.json"
    )
    build.add_argument("--client", action="append", dest="clients")

    validate = sub.add_parser("validate", help="Validate resume knowledge file")
    validate.add_argument("--client", action="append", dest="clients")

    search = sub.add_parser("search", help="Test resume retrieval")
    search.add_argument("--client", required=True)
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    if args.command == "list-clients":
        for client_id in list_client_ids():
            config = load_agent_config(client_id=client_id)
            print(
                f"{client_id}: {config.display_name} -> "
                f"{config.resume_knowledge_path.name}"
            )
        return 0

    if args.command == "search":
        asyncio.run(cmd_search(args.client, args.query, args.top_k))
        return 0

    exit_code = 0
    for client_id in _clients_arg(args.clients):
        print(f"=== {client_id} ===")
        if args.command == "build":
            asyncio.run(cmd_build(client_id))
        elif args.command == "validate" and not asyncio.run(cmd_validate(client_id)):
            exit_code = 1
        print()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
