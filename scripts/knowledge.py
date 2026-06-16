#!/usr/bin/env python3
"""Build and validate per-client knowledge bases under config/."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from utils.config import list_client_ids, load_agent_config  # noqa: E402
from utils.knowledge_builder import KnowledgeBuilder  # noqa: E402
from utils.knowledge_store import KnowledgeStore  # noqa: E402


def _load_env() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.local", override=True)


def _clients_arg(clients: list[str] | None) -> list[str]:
    return clients or list_client_ids()


async def cmd_build(client_id: str, max_pages: int | None) -> None:
    config = load_agent_config(client_id=client_id)
    output = await KnowledgeBuilder(config).build(max_pages=max_pages)
    print(f"Built {output}")


async def cmd_validate(client_id: str) -> bool:
    config = load_agent_config(client_id=client_id)
    issues: list[str] = []

    if not config.knowledge_path.exists():
        issues.append(f"Missing knowledge file: {config.knowledge_path}")

    store = KnowledgeStore(config)
    await store.initialize()

    if not store.documents:
        issues.append("Knowledge base is empty")
    elif not any(doc.get("embedding") for doc in store.documents):
        issues.append("No embeddings found; rebuild with OPENAI_API_KEY set")

    print(f"Client: {client_id}")
    print(f"Knowledge: {config.knowledge_path}")
    print(f"Chunks: {len(store.documents)}")

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
    for index, result in enumerate(results, start=1):
        score = result.get("score")
        suffix = f" score={score:.3f}" if isinstance(score, float) else ""
        print(f"{index}. [{result.get('source', 'unknown')}]{suffix}")
        print(result.get("text", "").strip())
        print()


def main() -> int:
    _load_env()

    parser = argparse.ArgumentParser(description="Manage config/{client}.json knowledge bases")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-clients", help="List configured clients")

    build = sub.add_parser("build", help="Build website+PDF knowledge into config/{client}.json")
    build.add_argument("--client", action="append", dest="clients")
    build.add_argument("--max-pages", type=int)

    validate = sub.add_parser("validate", help="Validate config/{client}.json")
    validate.add_argument("--client", action="append", dest="clients")

    search = sub.add_parser("search", help="Test retrieval")
    search.add_argument("--client", required=True)
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    if args.command == "list-clients":
        for client_id in list_client_ids():
            config = load_agent_config(client_id=client_id)
            print(f"{client_id}: {config.website_name} -> {config.knowledge_path.name}")
        return 0

    if args.command == "search":
        asyncio.run(cmd_search(args.client, args.query, args.top_k))
        return 0

    exit_code = 0
    for client_id in _clients_arg(args.clients):
        print(f"=== {client_id} ===")
        if args.command == "build":
            asyncio.run(cmd_build(client_id, args.max_pages))
        elif args.command == "validate" and not asyncio.run(cmd_validate(client_id)):
            exit_code = 1
        print()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
