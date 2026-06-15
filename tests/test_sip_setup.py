import pytest

from utils.sip_setup import load_tenant_trunk_specs


def test_load_tenant_trunk_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOBIZ_SIP_USERNAME", "shared-user")
    monkeypatch.setenv("VOBIZ_SIP_PASSWORD", "shared-pass")

    specs = load_tenant_trunk_specs()
    client_ids = {spec.client_id for spec in specs}

    assert "client-1" in client_ids
    assert "client-2" in client_ids

    by_client = {spec.client_id: spec for spec in specs}
    assert by_client["client-1"].phone_number == "+911171366880"
    assert by_client["client-1"].trunk_name == "client-1-inbound"
    assert by_client["client-1"].sip_username == "shared-user"
    assert by_client["client-2"].phone_number == "+911171366881"


def test_load_tenant_trunk_specs_requires_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VOBIZ_SIP_USERNAME", raising=False)
    monkeypatch.delenv("VOBIZ_SIP_PASSWORD", raising=False)

    with pytest.raises(ValueError, match="missing SIP credentials"):
        load_tenant_trunk_specs()
