import pytest

from utils.config import (
    client_properties_path,
    client_resume_knowledge_path,
    list_client_ids,
    load_agent_config,
    load_tenant_map,
)


def test_list_client_ids() -> None:
    clients = list_client_ids()
    assert "client-1" in clients
    assert "client-2" in clients


def test_client_paths() -> None:
    for client_id in ("client-1", "client-2"):
        config = load_agent_config(client_id=client_id)
        assert config.client_id == client_id
        assert config.properties_path == client_properties_path(client_id)
        assert config.resume_knowledge_path == client_resume_knowledge_path(client_id)


def test_tenant_map() -> None:
    mapping = load_tenant_map()
    assert mapping["911171366880"] == "client-1"
    assert mapping["911171366881"] == "client-2"


def test_unknown_client_raises() -> None:
    with pytest.raises(ValueError, match="Unknown client"):
        load_agent_config(client_id="missing-client")
