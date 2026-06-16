from __future__ import annotations

import os
import re
from dataclasses import dataclass

from utils.config import list_client_ids, load_tenant_map_by_client


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
    default_username = os.getenv("VOBIZ_SIP_USERNAME", "")
    default_password = os.getenv("VOBIZ_SIP_PASSWORD", "")

    phone_by_client = load_tenant_map_by_client()
    specs: list[TenantTrunkSpec] = []

    for client_id in list_client_ids():
        phone = phone_by_client.get(client_id)
        if not phone:
            continue

        if not default_username or not default_password:
            raise ValueError(
                f"{client_id}: set VOBIZ_SIP_USERNAME/VOBIZ_SIP_PASSWORD in .env.local"
            )

        specs.append(
            TenantTrunkSpec(
                client_id=client_id,
                phone_number=phone,
                trunk_name=f"{client_id}-inbound",
                sip_username=default_username,
                sip_password=default_password,
            )
        )

    if not specs:
        raise ValueError("No entries in config/tenant-map.json for configured clients")

    return specs
