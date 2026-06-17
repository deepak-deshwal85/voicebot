import json
from pathlib import Path

import pytest

from utils.config import load_agent_config
from utils.knowledge_store import KnowledgeStore


@pytest.mark.asyncio
async def test_search_website_store(
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
                        "text": "Our pension transfer service helps customers.",
                        "metadata": {"source": "website"},
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
    await store.website.ensure_loaded()

    assert len(store.website.documents) == 1
    results = await store.search_website("pension transfer")
    assert "pension transfer" in results.lower()


@pytest.mark.asyncio
async def test_embedding_search_falls_back_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    client_id = "client-1"
    (config_dir / f"{client_id}.properties").write_text(
        "client.id=client-1\nwebsite.name=Test\nwebsite.url=https://example.com/\n",
        encoding="utf-8",
    )
    (config_dir / f"{client_id}-pdf.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "text": "Python, FastAPI, and cloud deployment experience.",
                        "metadata": {"source": "pdf", "filename": "resume.pdf"},
                        "embedding": [0.1, 0.2, 0.3],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.config.CONFIG_DIR", config_dir)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = load_agent_config(client_id=client_id)
    store = KnowledgeStore(config)
    await store.initialize()
    await store.pdf.ensure_loaded()

    results = await store.search_pdf("cloud deployment")
    assert results
    assert "cloud" in results.lower()


@pytest.mark.asyncio
async def test_preload_pdf_loads_without_website(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    client_id = "client-1"
    (config_dir / f"{client_id}.properties").write_text(
        "client.id=client-1\nwebsite.name=Test\nwebsite.url=https://example.com/\n",
        encoding="utf-8",
    )
    (config_dir / f"{client_id}-pdf.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "text": "Resume skills include Python and AWS.",
                        "metadata": {"source": "pdf", "filename": "resume.pdf"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.config.CONFIG_DIR", config_dir)

    config = load_agent_config(client_id=client_id)
    store = KnowledgeStore(config)
    loaded = await store.preload_pdf()

    assert loaded is True
    assert len(store.pdf.documents) == 1
    assert store.website.documents == []


@pytest.mark.asyncio
async def test_preload_website_loads_without_pdf(
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
                        "text": "Our pension transfer service helps customers.",
                        "metadata": {"source": "website"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.config.CONFIG_DIR", config_dir)

    config = load_agent_config(client_id=client_id)
    store = KnowledgeStore(config)
    loaded = await store.preload_website()

    assert loaded is True
    assert len(store.website.documents) == 1
    assert store.pdf.documents == []


@pytest.mark.asyncio
async def test_missing_store_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
