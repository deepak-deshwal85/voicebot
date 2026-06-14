import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import load_agent_config  # noqa: E402
from utils.knowledge_store import KnowledgeStore  # noqa: E402


async def refresh(
    max_pages: int,
    force_refresh: bool,
    include_website: bool,
    include_pdfs: bool,
) -> None:
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv(ROOT_DIR / ".env.local", override=True)
    config = load_agent_config()

    store = KnowledgeStore(config)
    await store.rebuild(
        max_pages=max_pages,
        force_refresh=force_refresh,
        include_website=include_website,
        include_pdfs=include_pdfs,
    )

    website_docs = sum(
        1
        for doc in store.documents
        if doc.get("metadata", {}).get("source") == "website"
    )
    pdf_docs = sum(
        1 for doc in store.documents if doc.get("metadata", {}).get("source") == "pdf"
    )
    embedded_docs = sum(1 for doc in store.documents if doc.get("embedding"))

    print(f"Knowledge base refreshed at: {config.knowledge_data_path}")
    print(f"Website: {config.website_name} ({config.website_url})")
    print(f"Total documents: {len(store.documents)}")
    print(f"Website chunks: {website_docs}")
    print(f"PDF chunks: {pdf_docs}")
    print(f"Embedded documents: {embedded_docs}")


def main() -> None:
    config = load_agent_config()
    parser = argparse.ArgumentParser(
        description="Rebuild the knowledge base from website content and PDF files"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=config.max_pages,
        help="Maximum website pages to crawl",
    )
    parser.add_argument(
        "--no-force-refresh",
        action="store_true",
        help="Append to the existing knowledge base instead of rebuilding it",
    )
    parser.add_argument(
        "--website-only",
        action="store_true",
        help="Only ingest website content",
    )
    parser.add_argument(
        "--pdfs-only",
        action="store_true",
        help="Only ingest PDF documents",
    )

    args = parser.parse_args()
    include_website = not args.pdfs_only
    include_pdfs = not args.website_only

    asyncio.run(
        refresh(
            max_pages=args.max_pages,
            force_refresh=not args.no_force_refresh,
            include_website=include_website,
            include_pdfs=include_pdfs,
        )
    )


if __name__ == "__main__":
    main()
