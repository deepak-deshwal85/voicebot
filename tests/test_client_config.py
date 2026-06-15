import pytest

from utils.config import list_client_ids, load_agent_config, resolve_client_config_path


def test_list_client_ids() -> None:
    clients = list_client_ids()
    assert "client-1" in clients
    assert "client-2" in clients


def test_load_client_config_paths() -> None:
    for client_id in ("client-1", "client-2"):
        config = load_agent_config(client_id=client_id)
        assert config.client_id == client_id
        assert config.telephony_phone_number
        assert config.knowledge_website_path.as_posix().endswith(
            f"data/clients/{client_id}/knowledge_website.json"
        )
        assert config.knowledge_pdfs_path.as_posix().endswith(
            f"data/clients/{client_id}/knowledge_pdfs.json"
        )
        assert config.pdf_folder.as_posix().endswith(f"data/clients/{client_id}")


def test_resolve_client_config_path_rejects_unknown_client() -> None:
    with pytest.raises(ValueError, match="Unknown client"):
        resolve_client_config_path("missing-client")
