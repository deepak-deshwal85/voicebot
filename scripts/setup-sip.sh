#!/usr/bin/env bash
# =============================================================================
# setup-sip.sh  —  Create SIP trunk configuration in LiveKit for Vobiz
#
# Usage:
#   bash scripts/setup-sip.sh          # create trunks (skips if already exist)
#   bash scripts/setup-sip.sh --fresh  # delete existing trunks then recreate
#
# Prerequisites:
#   - lk CLI installed  (https://docs.livekit.io/home/cli/)
#   - .env.local with LIVEKIT_* and VOBIZ_* vars set
#   - Run from project root:  cd /c/AI/voicebot/voicebot
# =============================================================================
set -euo pipefail

# ── Load environment ──────────────────────────────────────────────────────────
if [[ ! -f .env.local ]]; then
  echo "ERROR: .env.local not found. Run from the project root." >&2
  exit 1
fi
export $(grep -v '^#' .env.local | grep -v '^$' | sed 's/"//g' | xargs)

# Derive the LiveKit SIP ingress hostname from the WebSocket URL
# e.g.  wss://voicebotlatest-aah6v9ju.livekit.cloud  →  voicebotlatest-aah6v9ju.sip.livekit.cloud
SUBDOMAIN=$(echo "$LIVEKIT_URL" | sed 's|wss://||' | sed 's|\.livekit\.cloud.*||')
LIVEKIT_SIP_HOST="${SUBDOMAIN}.sip.livekit.cloud"

FRESH=${1:-""}

echo "============================================================"
echo " LiveKit SIP Setup"
echo "  Project URL : $LIVEKIT_URL"
echo "  SIP ingress : $LIVEKIT_SIP_HOST"
echo "  DID number  : $VOBIZ_PHONE_NUMBER"
echo "  Vobiz domain: $VOBIZ_SIP_DOMAIN"
echo "============================================================"
echo ""

# ── Optional: delete existing trunks and dispatch rules ──────────────────────
if [[ "$FRESH" == "--fresh" ]]; then
  echo ">>> --fresh: removing existing trunks and dispatch rules..."

  lk sip dispatch list --json 2>/dev/null \
    | grep -oP '"sip_dispatch_rule_id"\s*:\s*"[^"]+"' \
    | grep -oP 'SDR_\w+' \
    | while read -r id; do
        echo "  Deleting dispatch rule $id"
        lk sip dispatch delete "$id"
      done

  lk sip inbound list --json 2>/dev/null \
    | grep -oP '"sip_trunk_id"\s*:\s*"[^"]+"' \
    | grep -oP 'ST_\w+' \
    | while read -r id; do
        echo "  Deleting inbound trunk $id"
        lk sip inbound delete "$id"
      done

  lk sip outbound list --json 2>/dev/null \
    | grep -oP '"sip_trunk_id"\s*:\s*"[^"]+"' \
    | grep -oP 'ST_\w+' \
    | while read -r id; do
        echo "  Deleting outbound trunk $id"
        lk sip outbound delete "$id"
      done

  echo "Done cleaning up."
  echo ""
fi

# ── Step 1: Create inbound SIP trunk ─────────────────────────────────────────
# Receives calls FROM Vobiz → LiveKit.
# AllowedAddresses is left empty so LiveKit accepts calls from any Vobiz IP.
# Auth (username/password) lets Vobiz authenticate itself to LiveKit.
echo ">>> Step 1: Creating inbound SIP trunk..."
lk sip inbound create \
  --name   "${VOBIZ_SIP_TRUNK_NAME}" \
  --numbers "${VOBIZ_PHONE_NUMBER}" \
  --auth-user "${VOBIZ_SIP_USERNAME}" \
  --auth-pass "${VOBIZ_SIP_PASSWORD}"

echo ""

# ── Step 2: Create outbound SIP trunk ────────────────────────────────────────
# Used when the agent places outbound calls THROUGH Vobiz.
echo ">>> Step 2: Creating outbound SIP trunk..."
lk sip outbound create \
  --name      "vobiz-outbound" \
  --address   "${VOBIZ_SIP_DOMAIN}" \
  --numbers   "${VOBIZ_PHONE_NUMBER}" \
  --auth-user "${VOBIZ_SIP_USERNAME}" \
  --auth-pass "${VOBIZ_SIP_PASSWORD}"

echo ""

# ── Step 3: Get the new inbound trunk ID ─────────────────────────────────────
echo ">>> Step 3: Fetching inbound trunk ID..."
INBOUND_TRUNK_ID=$(lk sip inbound list --json 2>/dev/null \
  | python3 -c "
import json, sys
trunks = json.load(sys.stdin)
items = trunks.get('items', trunks) if isinstance(trunks, dict) else trunks
for t in items:
    if t.get('name') == '${VOBIZ_SIP_TRUNK_NAME}':
        print(t['sip_trunk_id'])
        break
" 2>/dev/null || echo "")

if [[ -z "$INBOUND_TRUNK_ID" ]]; then
  echo "Could not auto-detect inbound trunk ID."
  echo "Run:  lk sip inbound list"
  echo "Then re-run:  bash scripts/setup-sip.sh  (with INBOUND_TRUNK_ID set)"
  INBOUND_TRUNK_ID="ST_REPLACE_WITH_ACTUAL_ID"
fi
echo "  Inbound trunk ID: $INBOUND_TRUNK_ID"
echo ""

# ── Step 4: Create dispatch rule ─────────────────────────────────────────────
# Routes every inbound call on the trunk to a new room and dispatches
# this worker (agent_name = "telephone-agent").
# The --agents field is only available via JSON request (not a CLI flag).
echo ">>> Step 4: Creating dispatch rule..."
DISPATCH_JSON=$(mktemp /tmp/dispatch-XXXXXX.json)
cat > "$DISPATCH_JSON" << EOF
{
  "name": "vobiz-inbound-dispatch",
  "trunk_ids": ["${INBOUND_TRUNK_ID}"],
  "rule": {
    "dispatchRuleIndividual": {
      "roomPrefix": "call-"
    }
  },
  "agents": ["telephone-agent"]
}
EOF
lk sip dispatch create "$DISPATCH_JSON"
rm -f "$DISPATCH_JSON"

echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo " LiveKit setup complete. Verify:"
echo ""
echo "  lk sip inbound list"
echo "  lk sip outbound list"
echo "  lk sip dispatch list"
echo "============================================================"
echo ""
echo "============================================================"
echo " VOBIZ PORTAL — manual steps required"
echo " https://portal.vobiz.ai  (or your Vobiz admin URL)"
echo ""
echo "  1. Go to: Trunks → SIP Trunks → Add SIP Trunk"
echo "     Name        : livekit-inbound"
echo "     Host / URI  : ${LIVEKIT_SIP_HOST}"
echo "     Transport   : UDP (or TCP if UDP is blocked)"
echo "     Auth user   : ${VOBIZ_SIP_USERNAME}"
echo "     Auth pass   : ${VOBIZ_SIP_PASSWORD}"
echo ""
echo "  2. Go to: Numbers → ${VOBIZ_PHONE_NUMBER} → Edit"
echo "     Destination : SIP Trunk → livekit-inbound"
echo "     SIP URI     : sip:${VOBIZ_PHONE_NUMBER}@${LIVEKIT_SIP_HOST}"
echo ""
echo "  3. For OUTBOUND (calls from agent to user):"
echo "     Vobiz already routes calls from ${VOBIZ_SIP_USERNAME} via outbound trunk."
echo "     Ensure LiveKit's egress IPs are whitelisted in Vobiz firewall."
echo "     LiveKit egress IP ranges: https://docs.livekit.io/home/self-hosting/ports-firewall/"
echo ""
echo "  4. Test: call ${VOBIZ_PHONE_NUMBER} — agent should greet you."
echo "============================================================"
