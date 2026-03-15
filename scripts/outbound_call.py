"""
Outbound SIP call script.

Usage:
    uv run python scripts/outbound_call.py +919868402577
    uv run python scripts/outbound_call.py +919868402577 --room my-room --wait

Environment variables are loaded from .env.local automatically.
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from livekit import api

load_dotenv(".env.local", override=True)


async def get_outbound_trunk_id(lk_api: api.LiveKitAPI) -> str:
    """Return the first outbound SIP trunk ID found."""
    trunks = await lk_api.sip.list_sip_outbound_trunk(api.ListSIPTrunkRequest())
    if not trunks.items:
        raise RuntimeError(
            "No outbound SIP trunk found. "
            "Create one first using scripts/reset-vobiz-sip.ps1"
        )
    return trunks.items[0].sip_trunk_id


async def place_outbound_call(
    call_to: str,
    room: str,
    identity: str,
    participant_name: str,
    from_number: str,
    trunk_id: str | None,
    wait: bool,
    agent_name: str,
) -> None:
    livekit_url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]

    async with api.LiveKitAPI(url=livekit_url, api_key=api_key, api_secret=api_secret) as lk_api:
        resolved_trunk_id = trunk_id or await get_outbound_trunk_id(lk_api)

        print(f"Placing outbound call via trunk: {resolved_trunk_id}")
        print(f"From: {from_number}  ->  To: {call_to}")
        print(f"Room: {room}")

        # Dispatch the agent to the room BEFORE placing the call so it is
        # ready to handle audio as soon as the callee picks up.
        dispatch = await lk_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=agent_name,
                room=room,
                metadata="outbound",
            )
        )
        print(f"Agent dispatched: {dispatch.agent_name} -> room '{room}'")

        request = api.CreateSIPParticipantRequest(
            sip_trunk_id=resolved_trunk_id,
            sip_call_to=call_to,
            room_name=room,
            participant_identity=identity,
            participant_name=participant_name,
            dtmf="",
            play_ringtone=True,
            wait_until_answered=wait,
        )

        response = await lk_api.sip.create_sip_participant(request)

        print("\n--- Call created ---")
        print(f"SIP Call ID  : {response.sip_call_id}")
        print(f"Participant  : {response.participant_identity}")
        if wait:
            print("Call answered and active.")
        else:
            print("Call is being dialled (non-blocking).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Place an outbound SIP call via LiveKit")
    parser.add_argument("call_to", help="Phone number to call, e.g. +919868402577")
    parser.add_argument(
        "--room",
        default="outbound-call-room",
        help="LiveKit room name (default: outbound-call-room)",
    )
    parser.add_argument(
        "--identity",
        default="outbound-caller",
        help="Participant identity in the room (default: outbound-caller)",
    )
    parser.add_argument(
        "--name",
        default="VoiceBot Outbound",
        dest="participant_name",
        help="Participant display name (default: VoiceBot Outbound)",
    )
    parser.add_argument(
        "--from",
        default=os.environ.get("VOBIZ_PHONE_NUMBER", "+911171366880"),
        dest="from_number",
        help="Caller ID / DID number (default: VOBIZ_PHONE_NUMBER env var)",
    )
    parser.add_argument(
        "--trunk",
        default=None,
        dest="trunk_id",
        help="Outbound SIP trunk ID (auto-detected if omitted)",
    )
    parser.add_argument(
        "--agent",
        default=os.environ.get("VOBIZ_AGENT_NAME", "telephone-agent"),
        dest="agent_name",
        help="Agent name to dispatch to the room (default: telephone-agent)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block until the call is answered",
    )
    args = parser.parse_args()

    asyncio.run(
        place_outbound_call(
            call_to=args.call_to,
            room=args.room,
            identity=args.identity,
            participant_name=args.participant_name,
            from_number=args.from_number,
            trunk_id=args.trunk_id,
            wait=args.wait,
            agent_name=args.agent_name,
        )
    )


if __name__ == "__main__":
    main()
