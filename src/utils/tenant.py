from __future__ import annotations

import asyncio
import logging
import os
import re

from livekit import rtc

from utils.config import DEFAULT_CLIENT_ID, list_client_ids, load_agent_config

logger = logging.getLogger("tenant")

SIP_TRUNK_PHONE_ATTR = "sip.trunkPhoneNumber"
TELEPHONY_ROOM_PREFIXES = ("call-", "outbound-")


def coerce_phone_value(value: object) -> str | None:
    """Normalize SIP trunk phone values from env vars or participant attributes."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)):
        return str(int(value))
    text = str(value).strip()
    return text or None


def is_telephony_room(room_name: str) -> bool:
    return room_name.startswith(TELEPHONY_ROOM_PREFIXES)


def normalize_phone_digits(phone: str | object) -> str:
    """Return digits-only form used for tenant lookup."""
    text = coerce_phone_value(phone)
    if not text:
        return ""
    return re.sub(r"\D", "", text)


def build_phone_to_client_index() -> dict[str, str]:
    """Map normalized phone digits to client id from client configs."""
    index: dict[str, str] = {}
    for client_id in list_client_ids():
        config = load_agent_config(client_id=client_id)
        if not config.telephony_phone_number:
            continue
        digits = normalize_phone_digits(config.telephony_phone_number)
        if not digits:
            continue
        if digits in index and index[digits] != client_id:
            raise ValueError(
                f"Duplicate telephony.phone_number for clients "
                f"'{index[digits]}' and '{client_id}'"
            )
        index[digits] = client_id
    return index


def resolve_client_id_for_phone(phone: str | object | None) -> str:
    """Resolve tenant client id from a SIP trunk phone number."""
    normalized_phone = coerce_phone_value(phone)
    if not normalized_phone:
        default_client = os.getenv("DEFAULT_CLIENT_ID", DEFAULT_CLIENT_ID)
        logger.info(
            "No SIP trunk phone number; using default client %s",
            default_client,
        )
        return default_client

    digits = normalize_phone_digits(normalized_phone)
    if not digits:
        raise ValueError(f"Invalid phone number: {phone!r}")

    client_id = build_phone_to_client_index().get(digits)
    if not client_id:
        configured = ", ".join(
            f"{cid}={load_agent_config(client_id=cid).telephony_phone_number}"
            for cid in list_client_ids()
            if load_agent_config(client_id=cid).telephony_phone_number
        )
        raise ValueError(
            f"No client configured for trunk phone '{normalized_phone}'. "
            f"Configured mappings: {configured or 'none'}"
        )

    logger.info("Resolved trunk phone %s -> client %s", normalized_phone, client_id)
    return client_id


def extract_sip_trunk_phone(participant: rtc.RemoteParticipant) -> str | None:
    return coerce_phone_value(participant.attributes.get(SIP_TRUNK_PHONE_ATTR))


def find_sip_trunk_phone(room: rtc.Room) -> str | None:
    for participant in room.remote_participants.values():
        if phone := extract_sip_trunk_phone(participant):
            return phone
    return None


async def wait_for_sip_trunk_phone(
    room: rtc.Room,
    timeout: float = 15.0,
    poll_interval: float = 0.2,
) -> str | None:
    """Wait until a remote participant exposes sip.trunkPhoneNumber."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    while loop.time() < deadline:
        if phone := find_sip_trunk_phone(room):
            return phone
        await asyncio.sleep(poll_interval)

    return None
