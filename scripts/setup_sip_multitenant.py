#!/usr/bin/env python3
"""Create LiveKit inbound SIP trunks for every configured tenant."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

from sip_setup import (  # noqa: E402
    TenantTrunkSpec,
    livekit_sip_host,
    load_tenant_trunk_specs,
)

DISPATCH_RULE_NAME = "multi-tenant-dispatch"
OUTBOUND_TRUNK_NAME = "vobiz-outbound"


def run_lk(
    args: list[str], *, dry_run: bool = False
) -> subprocess.CompletedProcess[str] | None:
    cmd = ["lk", *args]
    print(f"$ {' '.join(cmd)}")
    if dry_run:
        return None
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def lk_json(args: list[str]) -> dict | list:
    result = subprocess.run(
        ["lk", *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout or "[]")


def list_inbound_trunks() -> list[dict]:
    payload = lk_json(["sip", "inbound", "list", "--json"])
    if isinstance(payload, dict):
        return payload.get("items", [])
    return payload


def list_dispatch_rules() -> list[dict]:
    payload = lk_json(["sip", "dispatch", "list", "--json"])
    if isinstance(payload, dict):
        return payload.get("items", [])
    return payload


def list_outbound_trunks() -> list[dict]:
    payload = lk_json(["sip", "outbound", "list", "--json"])
    if isinstance(payload, dict):
        return payload.get("items", [])
    return payload


def delete_all_sip_resources(*, dry_run: bool) -> None:
    for rule in list_dispatch_rules():
        rule_id = rule.get("sip_dispatch_rule_id")
        if rule_id:
            run_lk(["sip", "dispatch", "delete", rule_id], dry_run=dry_run)

    for trunk in list_inbound_trunks():
        trunk_id = trunk.get("sip_trunk_id")
        if trunk_id:
            run_lk(["sip", "inbound", "delete", trunk_id], dry_run=dry_run)

    for trunk in list_outbound_trunks():
        trunk_id = trunk.get("sip_trunk_id")
        if trunk_id:
            run_lk(["sip", "outbound", "delete", trunk_id], dry_run=dry_run)


def find_trunk_id_by_name(trunks: list[dict], name: str) -> str | None:
    for trunk in trunks:
        if trunk.get("name") == name:
            return trunk.get("sip_trunk_id")
    return None


def ensure_inbound_trunk(spec: TenantTrunkSpec, *, dry_run: bool) -> str | None:
    existing = find_trunk_id_by_name(list_inbound_trunks(), spec.trunk_name)
    if existing:
        print(f"  inbound trunk exists: {spec.trunk_name} ({existing})")
        return existing

    run_lk(
        [
            "sip",
            "inbound",
            "create",
            "--name",
            spec.trunk_name,
            "--numbers",
            spec.phone_number,
            "--auth-user",
            spec.sip_username,
            "--auth-pass",
            spec.sip_password,
        ],
        dry_run=dry_run,
    )
    if dry_run:
        return f"ST_DRY_RUN_{spec.client_id}"

    trunk_id = find_trunk_id_by_name(list_inbound_trunks(), spec.trunk_name)
    if not trunk_id:
        raise RuntimeError(f"Created trunk {spec.trunk_name} but could not find its ID")
    print(f"  created inbound trunk: {spec.trunk_name} ({trunk_id})")
    return trunk_id


def ensure_outbound_trunk(
    specs: list[TenantTrunkSpec],
    *,
    sip_domain: str,
    dry_run: bool,
) -> None:
    existing = find_trunk_id_by_name(list_outbound_trunks(), OUTBOUND_TRUNK_NAME)
    if existing:
        print(f"  outbound trunk exists: {OUTBOUND_TRUNK_NAME} ({existing})")
        return

    first = specs[0]
    cmd = [
        "sip",
        "outbound",
        "create",
        "--name",
        OUTBOUND_TRUNK_NAME,
        "--address",
        sip_domain,
        "--auth-user",
        first.sip_username,
        "--auth-pass",
        first.sip_password,
    ]
    for spec in specs:
        cmd.extend(["--numbers", spec.phone_number])

    run_lk(cmd, dry_run=dry_run)
    print(f"  created outbound trunk: {OUTBOUND_TRUNK_NAME}")


def replace_dispatch_rule(
    trunk_ids: list[str],
    *,
    agent_name: str,
    dry_run: bool,
) -> None:
    payload = {
        "name": DISPATCH_RULE_NAME,
        "trunk_ids": trunk_ids,
        "rule": {"dispatchRuleIndividual": {"roomPrefix": "call-"}},
        "agents": [agent_name],
    }

    if dry_run:
        print(f"  would create dispatch rule: {json.dumps(payload, indent=2)}")
        return

    for rule in list_dispatch_rules():
        if rule.get("name") == DISPATCH_RULE_NAME:
            rule_id = rule.get("sip_dispatch_rule_id")
            if rule_id:
                run_lk(["sip", "dispatch", "delete", rule_id], dry_run=False)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2)
        path = handle.name

    try:
        run_lk(["sip", "dispatch", "create", path], dry_run=False)
        print(f"  created dispatch rule: {DISPATCH_RULE_NAME} -> {agent_name}")
    finally:
        Path(path).unlink(missing_ok=True)


def print_vobiz_summary(specs: list[TenantTrunkSpec], livekit_url: str) -> None:
    sip_host = livekit_sip_host(livekit_url)
    print("\n" + "=" * 60)
    print(" Vobiz portal — route each DID to LiveKit")
    print("=" * 60)
    for spec in specs:
        print(f"\n  Tenant: {spec.client_id}")
        print(f"    DID       : {spec.phone_number}")
        print(f"    SIP host  : {sip_host}")
        print(f"    SIP URI   : sip:{spec.phone_number}@{sip_host}")
        print(f"    Auth user : {spec.sip_username}")
    print("\n  Test each number and confirm logs show the matching client id.")
    print("=" * 60)


def run_setup(
    specs: list[TenantTrunkSpec],
    *,
    agent_name: str,
    livekit_url: str,
    sip_domain: str,
    fresh: bool,
    dry_run: bool,
    skip_outbound: bool,
) -> None:
    if dry_run:
        if fresh:
            print("\n>>> Would delete all SIP trunks and dispatch rules")
        print("\n>>> Would create inbound trunks:")
        for spec in specs:
            print(f"\n  {spec.client_id}:")
            run_lk(
                [
                    "sip",
                    "inbound",
                    "create",
                    "--name",
                    spec.trunk_name,
                    "--numbers",
                    spec.phone_number,
                    "--auth-user",
                    spec.sip_username,
                    "--auth-pass",
                    "********",
                ],
                dry_run=True,
            )
        if not skip_outbound and sip_domain:
            print("\n>>> Would create shared outbound trunk:")
            cmd = [
                "sip",
                "outbound",
                "create",
                "--name",
                OUTBOUND_TRUNK_NAME,
                "--address",
                sip_domain,
                "--auth-user",
                specs[0].sip_username,
                "--auth-pass",
                "********",
            ]
            for spec in specs:
                cmd.extend(["--numbers", spec.phone_number])
            run_lk(cmd, dry_run=True)
        trunk_ids = [f"ST_DRY_RUN_{spec.client_id}" for spec in specs]
        print("\n>>> Would create dispatch rule:")
        replace_dispatch_rule(trunk_ids, agent_name=agent_name, dry_run=True)
        return

    if fresh:
        print("\n>>> Removing existing SIP trunks and dispatch rules...")
        delete_all_sip_resources(dry_run=False)

    print("\n>>> Creating inbound trunks...")
    trunk_ids: list[str] = []
    for spec in specs:
        print(f"\n  {spec.client_id}:")
        trunk_id = ensure_inbound_trunk(spec, dry_run=False)
        if trunk_id:
            trunk_ids.append(trunk_id)

    if not skip_outbound:
        if not sip_domain:
            print("\n>>> Skipping outbound trunk (VOBIZ_SIP_DOMAIN not set)")
        else:
            print("\n>>> Creating shared outbound trunk...")
            ensure_outbound_trunk(specs, sip_domain=sip_domain, dry_run=False)

    print("\n>>> Creating dispatch rule...")
    replace_dispatch_rule(trunk_ids, agent_name=agent_name, dry_run=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create LiveKit inbound SIP trunks for all configured tenants"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing SIP trunks and dispatch rules before creating",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print lk commands without executing them",
    )
    parser.add_argument(
        "--skip-outbound",
        action="store_true",
        help="Do not create the shared outbound trunk",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.local", override=True)

    livekit_url = os.getenv("LIVEKIT_URL", "")
    if not livekit_url:
        raise SystemExit("LIVEKIT_URL is not set in .env.local")

    agent_name = os.getenv("AGENT_NAME", "telephone-agent")
    sip_domain = os.getenv("VOBIZ_SIP_DOMAIN", "")
    specs = load_tenant_trunk_specs()

    print("=" * 60)
    print(" Multi-tenant LiveKit SIP setup")
    print(f"  Agent name  : {agent_name}")
    print(f"  LiveKit URL : {livekit_url}")
    print(f"  SIP ingress : {livekit_sip_host(livekit_url)}")
    print(f"  Tenants     : {len(specs)}")
    for spec in specs:
        print(f"    - {spec.client_id}: {spec.phone_number} ({spec.trunk_name})")
    print("=" * 60)

    run_setup(
        specs,
        agent_name=agent_name,
        livekit_url=livekit_url,
        sip_domain=sip_domain,
        fresh=args.fresh,
        dry_run=args.dry_run,
        skip_outbound=args.skip_outbound,
    )

    print("\n>>> Done. Verify with:")
    print("  lk sip inbound list")
    print("  lk sip dispatch list")

    if not args.dry_run:
        print_vobiz_summary(specs, livekit_url)


if __name__ == "__main__":
    main()
