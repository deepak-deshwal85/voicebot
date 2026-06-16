import json
from pathlib import Path

import pytest

from utils.config import load_agent_config
from utils.knowledge_store import KnowledgeStore


@pytest.mark.asyncio
async def test_search_combined_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    client_id = "client-1"
    (config_dir / f"{client_id}.properties").write_text(
        "client.id=client-1\nwebsite.name=Test\nwebsite.url=https://example.com/\n",
        encoding="utf-8",
    )
    (config_dir / f"{client_id}.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "text": "Our pension transfer service helps customers.",
                        "metadata": {"source": "website"},
                    },
                    {
                        "text": "The employee handbook covers annual leave.",
                        "metadata": {"source": "pdf"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.config.CONFIG_DIR", config_dir)

    config = load_agent_config(client_id=client_id)
    store = KnowledgeStore(config)
    await store.initialize()

    assert len(store.documents) == 2
    results = await store.search("pension transfer", top_k=1)
    assert results[0]["source"] == "website"


@pytest.mark.asyncio
async def test_missing_store_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    client_id = "client-1"
    (config_dir / f"{client_id}.properties").write_text(
        "client.id=client-1\nwebsite.name=Test\nwebsite.url=https://example.com/\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.config.CONFIG_DIR", config_dir)

    config = load_agent_config(client_id=client_id)
    store = KnowledgeStore(config)
    await store.initialize()
    assert store.documents == []
