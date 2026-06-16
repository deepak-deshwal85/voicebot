#!/usr/bin/env python3
"""Print phone -> client mappings from config/tenant-map.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.config import load_agent_config, load_tenant_map  # noqa: E402
from utils.tenant import resolve_client_id_for_phone  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Test tenant phone routing")
    parser.add_argument("--phone", help="Phone number to resolve")
    parser.add_argument("--list", action="store_true", help="List mappings")
    args = parser.parse_args()

    if args.list or not args.phone:
        print("Mappings from config/tenant-map.json:")
        for digits, client_id in load_tenant_map().items():
            config = load_agent_config(client_id=client_id)
            print(f"  {client_id}: +{digits} -> {config.knowledge_path.name}")

    if args.phone:
        client_id = resolve_client_id_for_phone(args.phone)
        config = load_agent_config(client_id=client_id)
        print(f"\nPhone {args.phone!r} -> {client_id} ({config.knowledge_path})")


if __name__ == "__main__":
    main()
