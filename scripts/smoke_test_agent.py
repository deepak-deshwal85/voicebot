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

    success = await agent._ensure_vector_store()
    if success and agent.vector_store:
        print("Knowledge store initialized successfully")
        print(f"Documents loaded: {len(agent.vector_store.documents)}")

        results = await agent.vector_store.search("investment")
        print(f"Search test: Found {len(results)} results for 'investment'")
    else:
        print("Knowledge store initialization failed")


if __name__ == "__main__":
    asyncio.run(main())
