#!/usr/bin/env bash
# =============================================================================
# setup-sip.sh  —  Create LiveKit SIP trunks for all configured tenants
#
# Usage:
#   bash scripts/setup-sip.sh              # create trunks (skip existing)
#   bash scripts/setup-sip.sh --fresh      # delete all trunks/rules, recreate
#   bash scripts/setup-sip.sh --dry-run    # print planned lk commands
#
# Reads phone mappings from config/tenant-map.json.
# Optional per-client overrides:
#   telephony.sip.trunk_name=client-1-inbound
#   telephony.sip.username=...
#   telephony.sip.password=...
#
# Shared credentials fall back to VOBIZ_SIP_USERNAME / VOBIZ_SIP_PASSWORD in .env.local
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec uv run python scripts/setup_sip_multitenant.py "$@"
