from __future__ import annotations

import os
import re
from dataclasses import dataclass

from utils.config import list_client_ids, load_agent_config, read_client_properties


@dataclass(frozen=True)
class TenantTrunkSpec:
    client_id: str
    phone_number: str
    trunk_name: str
    sip_username: str
    sip_password: str


def livekit_sip_host(livekit_url: str) -> str:
    subdomain = re.sub(r"^wss://", "", livekit_url.strip())
    subdomain = re.sub(r"\.livekit\.cloud.*$", "", subdomain)
    return f"{subdomain}.sip.livekit.cloud"


def load_tenant_trunk_specs() -> list[TenantTrunkSpec]:
    """Build LiveKit inbound trunk specs from all client configs."""
    default_username = os.getenv("VOBIZ_SIP_USERNAME", "")
    default_password = os.getenv("VOBIZ_SIP_PASSWORD", "")

    specs: list[TenantTrunkSpec] = []
    for client_id in list_client_ids():
        config = load_agent_config(client_id=client_id)
        if not config.telephony_phone_number:
            continue

        raw = read_client_properties(client_id)
        username = raw.get("telephony.sip.username", "").strip() or default_username
        password = raw.get("telephony.sip.password", "").strip() or default_password
        trunk_name = (
            raw.get("telephony.sip.trunk_name", "").strip() or f"{client_id}-inbound"
        )

        if not username or not password:
            raise ValueError(
                f"{client_id}: missing SIP credentials. Set telephony.sip.username/password "
                f"in agent.properties or VOBIZ_SIP_USERNAME/VOBIZ_SIP_PASSWORD in .env.local"
            )

        specs.append(
            TenantTrunkSpec(
                client_id=client_id,
                phone_number=config.telephony_phone_number,
                trunk_name=trunk_name,
                sip_username=username,
                sip_password=password,
            )
        )

    if not specs:
        raise ValueError(
            "No tenants with telephony.phone_number configured under config/clients/"
        )

    names = [spec.trunk_name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate telephony.sip.trunk_name values: {names}")

    return specs
