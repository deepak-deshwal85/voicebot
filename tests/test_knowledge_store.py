import json
from pathlib import Path

import pytest

from utils.config import AgentConfig, load_agent_config
from utils.knowledge_store import KnowledgeStore


def _make_config(tmp_path: Path) -> AgentConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    properties = tmp_path / "agent.properties"
    properties.write_text(
        "\n".join(
            [
                "website.name=Test Site",
                "website.url=https://example.com/",
                f"knowledge.data_path={tmp_path / 'knowledge_base.json'}",
                f"knowledge.pdf_folder={data_dir}",
                "knowledge.runtime_scraping_enabled=false",
            ]
        ),
        encoding="utf-8",
    )
    return load_agent_config(properties_path=properties)


@pytest.mark.asyncio
async def test_keyword_search_from_prebuilt_store(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    store_path = config.knowledge_data_path
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps(
            {
                "version": 2,
                "documents": [
                    {
                        "text": "Our pension transfer service helps customers move old pensions.",
                        "metadata": {"source": "website", "type": "website_content"},
                    },
                    {
                        "text": "The employee handbook covers annual leave and benefits.",
                        "metadata": {"source": "pdf", "type": "pdf_content"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    store = KnowledgeStore(config)
    await store.initialize()

    website_results = await store.search("pension transfer", top_k=1)
    pdf_results = await store.search("annual leave handbook", top_k=1)

    assert website_results
    assert "pension transfer" in website_results[0]["text"].lower()
    assert pdf_results
    assert "annual leave" in pdf_results[0]["text"].lower()


@pytest.mark.asyncio
async def test_ingest_pdfs_adds_document_chunks(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path)
    pdf_dir = config.pdf_folder
    (pdf_dir / "sample.pdf").write_bytes(b"%PDF-1.4")

    store = KnowledgeStore(config)

    async def fake_extract(pdf_path: Path) -> str:
        return "This PDF explains the company refund policy in detail."

    monkeypatch.setattr(store, "_extract_pdf_text", fake_extract)
    await store.ingest_pdfs()

    assert store.documents
    assert store.documents[0]["metadata"]["source"] == "pdf"
    assert "refund policy" in store.documents[0]["text"].lower()


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
