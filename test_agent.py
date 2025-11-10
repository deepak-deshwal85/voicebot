import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent import Assistant

async def test_agent():
    print("Testing agent initialization...")
    agent = Assistant()

    # Test vector store initialization
    success = await agent._ensure_vector_store()
    if success:
        print("Vector store initialized successfully")
        print(f"Website documents loaded: {len(agent.vector_store.documents) if agent.vector_store else 0}")

        # Test a search
        results = await agent.vector_store.search_with_fallback("investment")
        print(f"Search test: Found {len(results)} results for 'investment'")
    else:
        print("Vector store initialization failed")

if __name__ == "__main__":
    asyncio.run(test_agent())