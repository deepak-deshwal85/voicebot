"""Test tenant resolution from phone mapping."""

import pytest

from utils.config import load_tenant_map
from utils.tenant import (
    coerce_phone_value,
    is_telephony_room,
    normalize_phone_digits,
    resolve_client_id_for_phone,
)


def test_normalize_phone_digits() -> None:
    assert normalize_phone_digits("+91 117 1366880") == "911171366880"
    assert normalize_phone_digits(911171366880) == "911171366880"


def test_coerce_phone_value() -> None:
    assert coerce_phone_value("+911171366880") == "+911171366880"
    assert coerce_phone_value(911171366880) == "911171366880"
    assert coerce_phone_value(None) is None


def test_is_telephony_room() -> None:
    assert is_telephony_room("call-abc123")
    assert not is_telephony_room("mock_room")


def test_tenant_map() -> None:
    assert load_tenant_map()["911171366880"] == "client-1"


def test_resolve_client_id_for_phone() -> None:
    assert resolve_client_id_for_phone("+911171366880") == "client-1"
    assert resolve_client_id_for_phone("911171366881") == "client-2"


def test_resolve_client_id_unknown_phone_raises() -> None:
    with pytest.raises(ValueError, match="No client mapped"):
        resolve_client_id_for_phone("+10000000000")


def test_resolve_client_id_without_phone_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEFAULT_CLIENT_ID", "client-2")
    assert resolve_client_id_for_phone(None) == "client-2"
