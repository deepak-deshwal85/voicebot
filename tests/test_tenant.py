"""Test tenant resolution from SIP trunk phone numbers."""

import pytest

from utils.config import load_agent_config
from utils.tenant import (
    build_phone_to_client_index,
    normalize_phone_digits,
    resolve_client_id_for_phone,
)


def test_normalize_phone_digits() -> None:
    assert normalize_phone_digits("+91 117 1366880") == "911171366880"
    assert normalize_phone_digits("911171366880") == "911171366880"


def test_build_phone_to_client_index() -> None:
    index = build_phone_to_client_index()
    assert index["911171366880"] == "client-1"
    assert index["911171366881"] == "client-2"


def test_resolve_client_id_for_phone() -> None:
    assert resolve_client_id_for_phone("+911171366880") == "client-1"
    assert resolve_client_id_for_phone("911171366881") == "client-2"


def test_resolve_client_id_unknown_phone_raises() -> None:
    with pytest.raises(ValueError, match="No client configured"):
        resolve_client_id_for_phone("+10000000000")


def test_resolve_client_id_without_phone_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEFAULT_CLIENT_ID", "client-2")
    assert resolve_client_id_for_phone(None) == "client-2"


def test_client_configs_include_telephony_phone() -> None:
    for client_id in ("client-1", "client-2"):
        config = load_agent_config(client_id=client_id)
        assert config.telephony_phone_number
        assert config.telephony_phone_number.startswith("+")
