from pathlib import Path

import pytest

from utils.config import load_agent_config, load_knowledge_preload_settings


def test_load_agent_config_defaults(tmp_path: Path) -> None:
    properties = tmp_path / "agent.properties"
    properties.write_text(
        "\n".join(
            [
                "website.name=Acme Corp",
                "website.url=https://acme.example/",
                "agent.initial_greeting=Welcome to {website_name}.",
                "agent.instructions=You assist users of {website_name}.",
                "agent.no_results_message=No info for {website_name}.",
            ]
        ),
        encoding="utf-8",
    )

    config = load_agent_config(properties_path=properties)

    assert config.website_name == "Acme Corp"
    assert config.website_url == "https://acme.example/"
    assert config.initial_greeting == "Welcome to Acme Corp."
    assert "Acme Corp" in config.instructions
    assert config.website_knowledge_path.name.endswith("-website.json")
    assert config.pdf_knowledge_path.name.endswith("-pdf.json")


def test_multiline_instructions_are_unescaped(tmp_path: Path) -> None:
    properties = tmp_path / "agent.properties"
    properties.write_text(
        "website.name=Demo\n"
        "website.url=https://demo.example/\n"
        "agent.instructions=Line one.\\nLine two for {website_name}.",
        encoding="utf-8",
    )

    config = load_agent_config(properties_path=properties)
    assert "Line one." in config.instructions
    assert "Line two for Demo." in config.instructions


def test_knowledge_preload_settings_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRELOAD_PDF_KNOWLEDGE", raising=False)
    monkeypatch.delenv("PRELOAD_WEBSITE_KNOWLEDGE", raising=False)

    settings = load_knowledge_preload_settings()
    assert settings.pdf is True
    assert settings.website is True


def test_knowledge_preload_settings_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRELOAD_PDF_KNOWLEDGE", "false")
    monkeypatch.setenv("PRELOAD_WEBSITE_KNOWLEDGE", "true")

    settings = load_knowledge_preload_settings()
    assert settings.pdf is False
    assert settings.website is True
