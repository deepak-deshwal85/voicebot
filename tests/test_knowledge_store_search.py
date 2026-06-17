import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from utils.config import load_agent_config
from utils.knowledge_store import KnowledgeStore


@pytest.mark.asyncio
async def test_keyword_search_runs_before_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    client_id = "client-1"
    (config_dir / f"{client_id}.properties").write_text(
        "client.id=client-1\nwebsite.name=Test\nwebsite.url=https://example.com/\n",
        encoding="utf-8",
    )
    (config_dir / f"{client_id}-website.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "text": "ETF and mutual fund differences explained here.",
                        "metadata": {"source": "website", "url": "https://example.com/etf"},
                        "embedding": [0.1, 0.2, 0.3],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.config.CONFIG_DIR", config_dir)

    config = load_agent_config(client_id=client_id)
    store = KnowledgeStore(config)
    await store.website.ensure_loaded()

    embed_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(store.website.embeddings, "embed_query", embed_mock)

    results = await store.website.search("ETF mutual fund differences")
    assert results
    assert "ETF" in results[0]["text"]
    embed_mock.assert_not_called()
