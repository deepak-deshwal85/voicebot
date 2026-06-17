import json
from pathlib import Path

import pytest

from utils.config import load_agent_config
from utils.knowledge_store import KnowledgeStore
from utils.resume_sections import SECTION_SEARCH_TERMS, build_search_text


@pytest.mark.asyncio
async def test_search_resume_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    client_id = "client-1"
    (config_dir / f"{client_id}.properties").write_text(
        "client.id=client-1\nagent.display_name=Test\n",
        encoding="utf-8",
    )
    (config_dir / f"{client_id}-resume.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "text": "Python, FastAPI, and cloud deployment experience.",
                        "search_text": (
                            "Section: Skills\nTopics: skills, technical\n"
                            "Python, FastAPI, and cloud deployment experience."
                        ),
                        "metadata": {
                            "source": "resume",
                            "section": "skills",
                            "keywords": sorted(SECTION_SEARCH_TERMS["skills"]),
                            "filename": "resume.pdf",
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.config.CONFIG_DIR", config_dir)

    config = load_agent_config(client_id=client_id)
    store = KnowledgeStore(config)
    await store.ensure_loaded()

    results = await store.search("cloud deployment")
    assert "cloud" in results.lower()


@pytest.mark.asyncio
async def test_keyword_search_runs_before_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    client_id = "client-1"
    (config_dir / f"{client_id}.properties").write_text(
        "client.id=client-1\nagent.display_name=Test\n",
        encoding="utf-8",
    )
    (config_dir / f"{client_id}-resume.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "text": "Skills include Python and AWS certification.",
                        "search_text": (
                            "Section: Skills\nTopics: skills, technical\n"
                            "Skills include Python and AWS certification."
                        ),
                        "metadata": {
                            "source": "resume",
                            "section": "skills",
                            "keywords": sorted(SECTION_SEARCH_TERMS["skills"]),
                            "filename": "resume.pdf",
                        },
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
    await store.ensure_loaded()

    from unittest.mock import AsyncMock

    embed_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(store.embeddings, "embed_query", embed_mock)

    results = await store._search_documents("Python skills")
    assert results
    embed_mock.assert_not_called()


@pytest.mark.asyncio
async def test_qualification_query_prefers_education_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    client_id = "client-1"
    (config_dir / f"{client_id}.properties").write_text(
        "client.id=client-1\nagent.display_name=Test\n",
        encoding="utf-8",
    )
    (config_dir / f"{client_id}-resume.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "text": "The maximum throughput was improved.",
                        "search_text": (
                            "Section: Work Experience\nTopics: experience, employment\n"
                            "The maximum throughput was improved."
                        ),
                        "metadata": {
                            "source": "resume",
                            "section": "experience",
                            "keywords": sorted(SECTION_SEARCH_TERMS["experience"]),
                            "filename": "resume.pdf",
                        },
                        "embedding": [1.0, 0.0, 0.0],
                    },
                    {
                        "text": "MCA postgraduate qualification in computer applications.",
                        "search_text": build_search_text(
                            "education",
                            "MCA postgraduate qualification in computer applications.",
                        ),
                        "metadata": {
                            "source": "resume",
                            "section": "education",
                            "keywords": sorted(SECTION_SEARCH_TERMS["education"]),
                            "filename": "resume.pdf",
                        },
                        "embedding": [0.0, 1.0, 0.0],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.config.CONFIG_DIR", config_dir)

    config = load_agent_config(client_id=client_id)
    store = KnowledgeStore(config)
    await store.ensure_loaded()

    keyword_results = store._search_keywords("maximum qualification", top_k=3)
    assert keyword_results
    assert keyword_results[0]["section"] == "education"

    from unittest.mock import AsyncMock

    embed_mock = AsyncMock(return_value=[0.0, 1.0, 0.0])
    monkeypatch.setattr(store.embeddings, "embed_query", embed_mock)

    results = await store._search_documents("maximum qualification", top_k=3)
    assert results
    assert results[0]["section"] == "education"
    embed_mock.assert_not_called()


@pytest.mark.asyncio
async def test_weak_keyword_search_falls_through_to_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    client_id = "client-1"
    (config_dir / f"{client_id}.properties").write_text(
        "client.id=client-1\nagent.display_name=Test\n",
        encoding="utf-8",
    )
    (config_dir / f"{client_id}-resume.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "text": "Unrelated operational throughput notes.",
                        "search_text": (
                            "Section: Work Experience\nTopics: experience\n"
                            "Unrelated operational throughput notes."
                        ),
                        "metadata": {
                            "source": "resume",
                            "section": "experience",
                            "keywords": ["experience"],
                            "filename": "resume.pdf",
                        },
                        "embedding": [1.0, 0.0, 0.0],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.config.CONFIG_DIR", config_dir)

    config = load_agent_config(client_id=client_id)
    store = KnowledgeStore(config)
    await store.ensure_loaded()

    keyword_results = store._search_keywords("obscure unrelated topic", top_k=3)
    assert keyword_results == []

    from unittest.mock import AsyncMock

    embed_mock = AsyncMock(return_value=[1.0, 0.0, 0.0])
    monkeypatch.setattr(store.embeddings, "embed_query", embed_mock)

    results = await store._search_documents("obscure unrelated topic", top_k=3)
    assert results
    embed_mock.assert_called_once()


@pytest.mark.asyncio
async def test_missing_store_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    client_id = "client-1"
    (config_dir / f"{client_id}.properties").write_text(
        "client.id=client-1\nagent.display_name=Test\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("utils.config.CONFIG_DIR", config_dir)

    config = load_agent_config(client_id=client_id)
    store = KnowledgeStore(config)
    await store.initialize()
    assert store.documents == []
