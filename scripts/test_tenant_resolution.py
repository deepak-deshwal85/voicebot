#!/usr/bin/env python3
"""Print or test tenant resolution from SIP trunk phone numbers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.config import list_client_ids, load_agent_config  # noqa: E402
from utils.tenant import (  # noqa: E402
    build_phone_to_client_index,
    resolve_client_id_for_phone,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test multi-tenant phone routing")
    parser.add_argument(
        "--phone",
        help="SIP trunk phone number to resolve (e.g. +911171366880)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured phone -> client mappings",
    )
    args = parser.parse_args()

    if args.list or not args.phone:
        print("Configured tenant mappings:")
        index = build_phone_to_client_index()
        for client_id in list_client_ids():
            config = load_agent_config(client_id=client_id)
            phone = config.telephony_phone_number or "(not set)"
            print(f"  {client_id}: {phone}")
        print(f"\nLookup index (digits -> client): {index}")

    if args.phone:
        client_id = resolve_client_id_for_phone(args.phone)
        config = load_agent_config(client_id=client_id)
        print(f"\nPhone {args.phone!r} -> client {client_id}")
        print(f"  website: {config.website_name}")
        print(f"  kb: {config.knowledge_website_path.name}")


if __name__ == "__main__":
    main()
