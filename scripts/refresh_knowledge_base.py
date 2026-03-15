import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.knowledge_store import KnowledgeStore


async def refresh(max_pages: int, force_refresh: bool) -> None:
    load_dotenv(ROOT_DIR / ".env.local", override=True)

    store = KnowledgeStore()
    await store.initialize(
        preload_website=True,
        max_pages=max_pages,
        force_refresh=force_refresh,
    )

    print(f"Knowledge base refreshed. Total documents: {len(store.documents)}")
    print(f"Source website: {os.getenv('KNOWLEDGE_WEBSITE_URL', store.website_url)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="On-demand refresh of website knowledge base"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=int(os.getenv("KNOWLEDGE_PRELOAD_MAX_PAGES", "100")),
        help="Maximum pages to crawl (default: KNOWLEDGE_PRELOAD_MAX_PAGES or 100)",
    )
    parser.add_argument(
        "--no-force-refresh",
        action="store_true",
        help="Do incremental update without clearing existing knowledge",
    )

    args = parser.parse_args()
    asyncio.run(
        refresh(max_pages=args.max_pages, force_refresh=not args.no_force_refresh)
    )


if __name__ == "__main__":
    main()
