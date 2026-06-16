from __future__ import annotations

import asyncio
import logging
import os

from livekit import rtc

from utils.config import DEFAULT_CLIENT_ID, load_tenant_map

logger = logging.getLogger("tenant")

SIP_TRUNK_PHONE_ATTR = "sip.trunkPhoneNumber"
TELEPHONY_ROOM_PREFIXES = ("call-", "outbound-")


def coerce_phone_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)):
        return str(int(value))
    text = str(value).strip()
    return text or None


def normalize_phone_digits(phone: str | object) -> str:
    text = coerce_phone_value(phone)
    if not text:
        return ""
    return "".join(ch for ch in text if ch.isdigit())


def is_telephony_room(room_name: str) -> bool:
    return room_name.startswith(TELEPHONY_ROOM_PREFIXES)


def resolve_client_id_for_phone(phone: str | object | None) -> str:
    normalized_phone = coerce_phone_value(phone)
    if not normalized_phone:
        default_client = os.getenv("DEFAULT_CLIENT_ID", DEFAULT_CLIENT_ID)
        logger.info("No SIP trunk phone; using default client %s", default_client)
        return default_client

    digits = normalize_phone_digits(normalized_phone)
    if not digits:
        raise ValueError(f"Invalid phone number: {phone!r}")

    client_id = load_tenant_map().get(digits)
    if not client_id:
        raise ValueError(
            f"No client mapped for phone '{normalized_phone}' in config/tenant-map.json"
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
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    while loop.time() < deadline:
        if phone := find_sip_trunk_phone(room):
            return phone
        await asyncio.sleep(poll_interval)

    return None
