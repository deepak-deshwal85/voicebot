#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import list_client_ids, load_agent_config  # noqa: E402
from utils.knowledge_cli import (  # noqa: E402
    get_status,
    load_environment,
    migrate_legacy,
    print_search_results,
    print_status,
    refresh_knowledge,
    run_for_clients,
    search_knowledge,
    validate_knowledge,
)


def _add_client_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--client",
        action="append",
        dest="clients",
        help="Client id (repeatable). Defaults to CLIENT_ID or client-1",
    )


def _resolve_clients(clients: list[str] | None) -> list[str]:
    if clients:
        return clients
    config = load_agent_config()
    return [config.client_id]


async def _cmd_status(config) -> None:
    print_status(await get_status(config))


async def _cmd_validate(config) -> bool:
    issues = await validate_knowledge(config)
    print_status(await get_status(config))
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"- {issue}")
        return False
    print("Validation passed.")
    return True


async def _cmd_search(config, query: str, top_k: int) -> None:
    results = await search_knowledge(config, query, top_k=top_k)
    print_search_results(results)


async def _cmd_refresh(
    config,
    source: str,
    max_pages: int,
) -> None:
    include_website = source in {"website", "all"}
    include_pdfs = source in {"pdfs", "all"}
    status = await refresh_knowledge(
        config,
        include_website=include_website,
        include_pdfs=include_pdfs,
        max_pages=max_pages,
    )
    print_status(status)


async def _cmd_migrate(config) -> None:
    print_status(await migrate_legacy(config))


def main() -> int:
    load_environment(ROOT_DIR)
    config = load_agent_config()

    parser = argparse.ArgumentParser(
        description="Manage per-client knowledge bases outside the voice agent"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-clients", help="List configured clients")
    list_parser.set_defaults(func=lambda _args: _handle_list_clients())

    status_parser = subparsers.add_parser("status", help="Show knowledge base status")
    _add_client_arg(status_parser)
    status_parser.set_defaults(func=lambda args: _run_clients(args, _cmd_status))

    validate_parser = subparsers.add_parser(
        "validate", help="Validate knowledge stores"
    )
    _add_client_arg(validate_parser)
    validate_parser.set_defaults(
        func=lambda args: _run_clients(args, _cmd_validate, expect_bool=True)
    )

    search_parser = subparsers.add_parser("search", help="Test knowledge retrieval")
    _add_client_arg(search_parser)
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=3)
    search_parser.set_defaults(func=_handle_search)

    refresh_parser = subparsers.add_parser("refresh", help="Rebuild knowledge stores")
    _add_client_arg(refresh_parser)
    refresh_parser.add_argument(
        "source",
        choices=["website", "pdfs", "all"],
        nargs="?",
        default="all",
        help="Which source to refresh (default: all)",
    )
    refresh_parser.add_argument(
        "--max-pages",
        type=int,
        default=config.max_pages,
        help="Maximum website pages to crawl",
    )
    refresh_parser.set_defaults(func=_handle_refresh)

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate legacy combined knowledge file if present",
    )
    _add_client_arg(migrate_parser)
    migrate_parser.set_defaults(func=lambda args: _run_clients(args, _cmd_migrate))

    args = parser.parse_args()
    return args.func(args)


def _handle_list_clients() -> int:
    clients = list_client_ids()
    if not clients:
        print("No clients found under config/clients/")
        return 1
    for client_id in clients:
        config = load_agent_config(client_id=client_id)
        print(f"{client_id}: {config.website_name} -> {config.properties_path}")
    return 0


def _run_clients(args, handler, expect_bool: bool = False) -> int:
    clients = _resolve_clients(args.clients)
    exit_code = 0

    for client_id in clients:
        print(f"=== {client_id} ===")
        config = load_agent_config(client_id=client_id)
        result = asyncio.run(handler(config))
        if expect_bool and result is False:
            exit_code = 1
        print()

    return exit_code


def _handle_search(args) -> int:
    clients = _resolve_clients(args.clients)
    exit_code = 0
    for client_id in clients:
        print(f"=== {client_id} ===")
        config = load_agent_config(client_id=client_id)
        asyncio.run(_cmd_search(config, args.query, args.top_k))
    return exit_code


def _handle_refresh(args) -> int:
    clients = _resolve_clients(args.clients)

    async def runner(config):
        await _cmd_refresh(config, args.source, args.max_pages)

    return run_for_clients(clients, runner)


if __name__ == "__main__":
    raise SystemExit(main())
