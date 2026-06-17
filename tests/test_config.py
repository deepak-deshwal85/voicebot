from pathlib import Path

import pytest

from utils.config import load_agent_config, load_knowledge_preload_settings


def test_load_agent_config_defaults(tmp_path: Path) -> None:
    properties = tmp_path / "agent.properties"
    properties.write_text(
        "\n".join(
            [
                "agent.display_name=Acme Resume",
                "agent.initial_greeting=Welcome to {display_name}.",
                "agent.instructions=You assist users of {display_name}.",
                "agent.no_results_message=No info for {display_name}.",
            ]
        ),
        encoding="utf-8",
    )

    config = load_agent_config(properties_path=properties)

    assert config.display_name == "Acme Resume"
    assert config.initial_greeting == "Welcome to Acme Resume."
    assert "Acme Resume" in config.instructions
    assert config.resume_knowledge_path.name.endswith("-resume.json")


def test_multiline_instructions_are_unescaped(tmp_path: Path) -> None:
    properties = tmp_path / "agent.properties"
    properties.write_text(
        "agent.display_name=Demo\n"
        "agent.instructions=Line one.\\nLine two for {display_name}.",
        encoding="utf-8",
    )

    config = load_agent_config(properties_path=properties)
    assert "Line one." in config.instructions
    assert "Line two for Demo." in config.instructions


def test_knowledge_preload_settings_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRELOAD_RESUME_KNOWLEDGE", raising=False)

    settings = load_knowledge_preload_settings()
    assert settings.enabled is True


def test_knowledge_preload_settings_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRELOAD_RESUME_KNOWLEDGE", "false")

    settings = load_knowledge_preload_settings()
    assert settings.enabled is False
