import json
from pathlib import Path

import pytest

from utils.config import load_agent_config
from utils.knowledge_store import KnowledgeStore


def _make_config(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    properties = tmp_path / "agent.properties"
    properties.write_text(
        "\n".join(
            [
                "website.name=Test Site",
                "website.url=https://example.com/",
                f"knowledge.website_path={tmp_path / 'knowledge_website.json'}",
                f"knowledge.pdfs_path={tmp_path / 'knowledge_pdfs.json'}",
                f"knowledge.pdf_folder={data_dir}",
                "knowledge.runtime_scraping_enabled=false",
            ]
        ),
        encoding="utf-8",
    )
    return load_agent_config(properties_path=properties)


def _write_store(path: Path, source: str, documents: list[dict]) -> None:
    path.write_text(
        json.dumps({"version": 2, "source": source, "documents": documents}),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_search_merges_website_and_pdf_stores(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_store(
        config.knowledge_website_path,
        "website",
        [
            {
                "text": "Our pension transfer service helps customers move old pensions.",
                "metadata": {"source": "website", "type": "website_content"},
            }
        ],
    )
    _write_store(
        config.knowledge_pdfs_path,
        "pdf",
        [
            {
                "text": "The employee handbook covers annual leave and benefits.",
                "metadata": {"source": "pdf", "type": "pdf_content"},
            }
        ],
    )

    store = KnowledgeStore(config)
    await store.initialize()

    website_results = await store.search("pension transfer", top_k=1)
    pdf_results = await store.search("annual leave handbook", top_k=1)

    assert len(store.documents) == 2
    assert website_results[0]["source"] == "website"
    assert pdf_results[0]["source"] == "pdf"


@pytest.mark.asyncio
async def test_website_refresh_keeps_pdf_store(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path)
    _write_store(
        config.knowledge_pdfs_path,
        "pdf",
        [
            {
                "text": "Saved PDF content about refunds.",
                "metadata": {"source": "pdf", "type": "pdf_content"},
            }
        ],
    )

    store = KnowledgeStore(config)

    async def fake_scrape(self, max_pages: int = 10) -> None:
        self.website_documents = [
            {
                "text": "Fresh website content about pensions.",
                "metadata": {"source": "website", "type": "website_content"},
            }
        ]

    async def noop_embeddings(self, documents):
        return None

    monkeypatch.setattr(KnowledgeStore, "scrape_website", fake_scrape)
    monkeypatch.setattr(KnowledgeStore, "_compute_embeddings", noop_embeddings)

    await store.rebuild(include_website=True, include_pdfs=False)

    reloaded = KnowledgeStore(config)
    await reloaded.initialize()

    assert len(reloaded.website_documents) == 1
    assert len(reloaded.pdf_documents) == 1
    assert "refunds" in reloaded.pdf_documents[0]["text"]


@pytest.mark.asyncio
async def test_pdf_refresh_keeps_website_store(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path)
    _write_store(
        config.knowledge_website_path,
        "website",
        [
            {
                "text": "Saved website content about pensions.",
                "metadata": {"source": "website", "type": "website_content"},
            }
        ],
    )

    store = KnowledgeStore(config)

    async def fake_extract(pdf_path: Path) -> str:
        return "Fresh PDF content about annual leave."

    async def noop_embeddings(self, documents):
        return None

    monkeypatch.setattr(store, "_extract_pdf_text", fake_extract)
    monkeypatch.setattr(KnowledgeStore, "_compute_embeddings", noop_embeddings)
    (config.pdf_folder / "sample.pdf").write_bytes(b"%PDF-1.4")

    await store.rebuild(include_website=False, include_pdfs=True)

    reloaded = KnowledgeStore(config)
    await reloaded.initialize()

    assert len(reloaded.website_documents) == 1
    assert len(reloaded.pdf_documents) == 1
    assert "annual leave" in reloaded.pdf_documents[0]["text"]


@pytest.mark.asyncio
async def test_migrate_legacy_combined_store(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    legacy_path = tmp_path / "knowledge_base.json"
    legacy_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "text": "Legacy website chunk.",
                        "metadata": {"source": "website"},
                    },
                    {
                        "text": "Legacy pdf chunk.",
                        "metadata": {"source": "pdf"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    properties = tmp_path / "agent.properties"
    content = properties.read_text(encoding="utf-8")
    properties.write_text(
        content + f"\nknowledge.legacy_path={legacy_path}",
        encoding="utf-8",
    )
    config = load_agent_config(properties_path=properties)

    store = KnowledgeStore(config)
    await store.initialize()

    assert config.knowledge_website_path.exists()
    assert config.knowledge_pdfs_path.exists()
    assert len(store.website_documents) == 1
    assert len(store.pdf_documents) == 1


@pytest.mark.asyncio
async def test_runtime_scraping_disabled_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    config = _make_config(tmp_path)
    store = KnowledgeStore(config)

    async def fail_scrape(*args, **kwargs):
        raise AssertionError("Runtime scraping should not be called")

    monkeypatch.setattr(store, "search_website", fail_scrape)

    results = await store.search_with_fallback("unknown topic", top_k=1)
    assert results == []
