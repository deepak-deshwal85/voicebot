from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from utils.config import (
    AgentConfig,
    load_agent_config,
    load_store_metadata,
)
from utils.knowledge_store import KnowledgeStore


@dataclass
class KnowledgeStatus:
    client_id: str
    website_name: str
    website_url: str
    website_chunks: int
    pdf_chunks: int
    embedded_chunks: int
    pdf_files: list[str]
    website_store_path: Path
    pdfs_store_path: Path
    website_built_at: str | None
    pdfs_built_at: str | None


def load_environment(root_dir: Path) -> None:
    from dotenv import load_dotenv

    load_dotenv(root_dir / ".env")
    load_dotenv(root_dir / ".env.local", override=True)


async def get_status(config: AgentConfig) -> KnowledgeStatus:
    store = KnowledgeStore(config)
    await store.initialize()

    website_meta = load_store_metadata(config.knowledge_website_path)
    pdfs_meta = load_store_metadata(config.knowledge_pdfs_path)
    pdf_files = sorted(path.name for path in config.pdf_folder.glob("*.pdf"))

    return KnowledgeStatus(
        client_id=config.client_id,
        website_name=config.website_name,
        website_url=config.website_url,
        website_chunks=len(store.website_documents),
        pdf_chunks=len(store.pdf_documents),
        embedded_chunks=sum(1 for doc in store.documents if doc.get("embedding")),
        pdf_files=pdf_files,
        website_store_path=config.knowledge_website_path,
        pdfs_store_path=config.knowledge_pdfs_path,
        website_built_at=website_meta.get("built_at"),
        pdfs_built_at=pdfs_meta.get("built_at"),
    )


async def refresh_knowledge(
    config: AgentConfig,
    *,
    include_website: bool,
    include_pdfs: bool,
    max_pages: int,
) -> KnowledgeStatus:
    store = KnowledgeStore(config)
    await store.rebuild(
        max_pages=max_pages,
        include_website=include_website,
        include_pdfs=include_pdfs,
    )
    return await get_status(config)


async def search_knowledge(
    config: AgentConfig,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    store = KnowledgeStore(config)
    await store.initialize()
    return await store.search(query, top_k=top_k)


async def validate_knowledge(config: AgentConfig) -> list[str]:
    issues: list[str] = []

    if not config.pdf_folder.exists():
        issues.append(f"PDF folder missing: {config.pdf_folder}")

    if not config.knowledge_website_path.exists():
        issues.append(f"Website store missing: {config.knowledge_website_path}")

    if not config.knowledge_pdfs_path.exists():
        issues.append(f"PDF store missing: {config.knowledge_pdfs_path}")

    status = await get_status(config)
    if status.website_chunks == 0:
        issues.append("Website knowledge store is empty")
    if status.pdf_chunks == 0:
        issues.append("PDF knowledge store is empty")
    if status.embedded_chunks == 0:
        issues.append("No embeddings found; run refresh with OPENAI_API_KEY set")

    return issues


async def migrate_legacy(config: AgentConfig) -> KnowledgeStatus:
    store = KnowledgeStore(config)
    await store.initialize()
    return await get_status(config)


def print_status(status: KnowledgeStatus) -> None:
    print(f"Client: {status.client_id}")
    print(f"Website: {status.website_name} ({status.website_url})")
    print(f"Website store: {status.website_store_path}")
    print(f"PDF store: {status.pdfs_store_path}")
    print(f"PDF folder files: {', '.join(status.pdf_files) or 'none'}")
    print(f"Website chunks: {status.website_chunks}")
    print(f"PDF chunks: {status.pdf_chunks}")
    print(f"Embedded chunks: {status.embedded_chunks}")
    if status.website_built_at:
        print(f"Website built at: {status.website_built_at}")
    if status.pdfs_built_at:
        print(f"PDFs built at: {status.pdfs_built_at}")


def print_search_results(results: list[dict]) -> None:
    if not results:
        print("No results found.")
        return

    for index, result in enumerate(results, start=1):
        score = result.get("score")
        score_text = f" score={score:.3f}" if isinstance(score, float) else ""
        print(f"{index}. [{result.get('source', 'unknown')}]{score_text}")
        print(result.get("text", "").strip())
        print()


def run_for_clients(
    client_ids: list[str],
    runner,
) -> int:
    exit_code = 0
    for client_id in client_ids:
        print(f"=== {client_id} ===")
        try:
            config = load_agent_config(client_id=client_id)
            result = runner(config)
            if asyncio.iscoroutine(result):
                asyncio.run(result)
            elif result is False:
                exit_code = 1
        except Exception as exc:
            exit_code = 1
            print(f"Error for {client_id}: {exc}")
        print()
    return exit_code
