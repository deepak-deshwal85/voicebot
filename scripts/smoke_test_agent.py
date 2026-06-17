"""Manual smoke test for local agent + knowledge store initialization."""

import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from agent import Assistant  # noqa: E402
from utils.config import load_agent_config  # noqa: E402


async def main() -> None:
    print("Testing agent initialization...")
    config = load_agent_config(client_id="client-1")
    agent = Assistant(config=config)

    store = await agent._get_store()
    if store is None:
        print("Knowledge store initialization failed")
        return

    print("Knowledge store initialized successfully")
    print(f"Documents loaded: {len(store.documents)}")

    results = await store.search_website("investment")
    print(f"Search test: website results for 'investment': {bool(results)}")


if __name__ == "__main__":
    asyncio.run(main())
