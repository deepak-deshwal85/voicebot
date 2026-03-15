#!/usr/bin/env bash
# =============================================================================
# setup-vobiz.sh  —  Configure Vobiz SIP trunk to route calls to LiveKit
#
# What this does (Vobiz REST API):
#   1. Verify API credentials
#   2. List existing trunks to find the one matching VOBIZ_SIP_DOMAIN
#   3. Update trunk  → set inbound_destination = LiveKit SIP host
#   4. Create SIP credential on the trunk (username/password for LiveKit auth)
#   5. Print a summary of what to verify
#
# Prerequisites:
#   - VOBIZ_AUTH_ID and VOBIZ_AUTH_TOKEN set in .env.local
#     Get them from: https://console.vobiz.ai -> Settings -> API Credentials
#   - jq installed (brew install jq / apt install jq)
#   - Run from project root: cd /c/AI/voicebot/voicebot
# =============================================================================
set -euo pipefail

# ── Load environment ──────────────────────────────────────────────────────────
if [[ ! -f .env.local ]]; then
  echo "ERROR: .env.local not found. Run from the project root." >&2
  exit 1
fi
export $(grep -v '^#' .env.local | grep -v '^$' | sed 's/"//g' | xargs)

VOBIZ_API="https://api.vobiz.ai/api/v1"

# Derive LiveKit SIP host from LIVEKIT_URL
# e.g. wss://voicebotlatest-aah6v9ju.livekit.cloud → voicebotlatest-aah6v9ju.sip.livekit.cloud
SUBDOMAIN=$(echo "$LIVEKIT_URL" | sed 's|wss://||' | sed 's|\.livekit\.cloud.*||')
LIVEKIT_SIP_HOST="${SUBDOMAIN}.sip.livekit.cloud"

# ── Guard: require API credentials ───────────────────────────────────────────
if [[ -z "${VOBIZ_AUTH_ID:-}" || -z "${VOBIZ_AUTH_TOKEN:-}" ]]; then
  echo "============================================================"
  echo " ERROR: VOBIZ_AUTH_ID and VOBIZ_AUTH_TOKEN are not set."
  echo ""
  echo " Get them from:"
  echo "   https://console.vobiz.ai  -> Settings -> API Credentials"
  echo ""
  echo " Then add to .env.local:"
  echo "   VOBIZ_AUTH_ID=MA_XXXXXX"
  echo "   VOBIZ_AUTH_TOKEN=sk_live_XXXXXXXXXXXX"
  echo "============================================================"
  exit 1
fi

# Helper: authenticated curl call, prints response body
vobiz_api() {
  local method=$1
  local path=$2
  local data=${3:-}
  if [[ -n "$data" ]]; then
    curl -s -X "$method" "${VOBIZ_API}${path}" \
      -H "X-Auth-ID: ${VOBIZ_AUTH_ID}" \
      -H "X-Auth-Token: ${VOBIZ_AUTH_TOKEN}" \
      -H "Content-Type: application/json" \
      -d "$data"
  else
    curl -s -X "$method" "${VOBIZ_API}${path}" \
      -H "X-Auth-ID: ${VOBIZ_AUTH_ID}" \
      -H "X-Auth-Token: ${VOBIZ_AUTH_TOKEN}" \
      -H "Content-Type: application/json"
  fi
}

echo "============================================================"
echo " Vobiz SIP Setup"
echo "  Account ID     : $VOBIZ_AUTH_ID"
echo "  Vobiz domain   : $VOBIZ_SIP_DOMAIN"
echo "  DID number     : $VOBIZ_PHONE_NUMBER"
echo "  LiveKit SIP URI: $LIVEKIT_SIP_HOST"
echo "============================================================"
echo ""

# ── Step 1: Verify API credentials ───────────────────────────────────────────
echo ">>> Step 1: Verifying Vobiz API credentials..."
ACCOUNT=$(vobiz_api GET "/auth/me")
ACCOUNT_NAME=$(echo "$ACCOUNT" | jq -r '.name // .email // "unknown"')
ACCOUNT_ID=$(echo "$ACCOUNT"  | jq -r '.auth_id // .id // ""')

if [[ -z "$ACCOUNT_ID" ]]; then
  echo "ERROR: Could not authenticate with Vobiz API."
  echo "Response: $ACCOUNT"
  exit 1
fi
echo "  Authenticated as: $ACCOUNT_NAME  ($ACCOUNT_ID)"
echo ""

# ── Step 2: Find the trunk matching our SIP domain ───────────────────────────
echo ">>> Step 2: Looking for Vobiz trunk matching domain '$VOBIZ_SIP_DOMAIN'..."
TRUNKS=$(vobiz_api GET "/account/${ACCOUNT_ID}/trunks")
TRUNK_ID=$(echo "$TRUNKS" | jq -r \
  --arg domain "$VOBIZ_SIP_DOMAIN" \
  '[.items // . | .[] | select(.trunk_domain == $domain or (.trunk_domain | contains($domain)))] | first | .trunk_id // empty' \
  2>/dev/null || echo "")

if [[ -z "$TRUNK_ID" ]]; then
  # Domain not found — list all and let the user pick
  echo "  No trunk found matching '$VOBIZ_SIP_DOMAIN'."
  echo "  Existing trunks:"
  echo "$TRUNKS" | jq -r '(.items // .) | .[] | "    \(.trunk_id)  \(.name)  domain=\(.trunk_domain // "n/a")'
  echo ""
  echo "  Set VOBIZ_SIP_DOMAIN in .env.local to one of the domains above and re-run."
  exit 1
fi
echo "  Found trunk: $TRUNK_ID"
echo ""

# ── Step 3: Update trunk — set inbound_destination to LiveKit ────────────────
# NOTE: Do NOT include the sip: prefix — Vobiz adds it automatically.
echo ">>> Step 3: Setting inbound_destination → $LIVEKIT_SIP_HOST ..."
UPDATE_RESP=$(vobiz_api PUT "/account/${ACCOUNT_ID}/trunks/${TRUNK_ID}" \
  "{\"inbound_destination\": \"${LIVEKIT_SIP_HOST}\", \"trunk_direction\": \"both\"}")
UPDATED_DEST=$(echo "$UPDATE_RESP" | jq -r '.inbound_destination // "unknown"')
echo "  inbound_destination is now: $UPDATED_DEST"
echo ""

# ── Step 4: Create SIP credential on the trunk ───────────────────────────────
# This is the username/password LiveKit uses when it sends SIP auth to Vobiz.
echo ">>> Step 4: Creating SIP credential (user: $VOBIZ_SIP_USERNAME)..."
CRED_RESP=$(vobiz_api POST "/account/${ACCOUNT_ID}/credentials" \
  "{
    \"username\": \"${VOBIZ_SIP_USERNAME}\",
    \"password\": \"${VOBIZ_SIP_PASSWORD}\",
    \"enabled\": true,
    \"description\": \"LiveKit agent credential\"
  }")
CRED_ID=$(echo "$CRED_RESP" | jq -r '.id // empty')
if [[ -z "$CRED_ID" ]]; then
  # 409 = already exists — that is fine
  CONFLICT=$(echo "$CRED_RESP" | jq -r '.code // empty')
  if [[ "$CONFLICT" == "409" ]]; then
    echo "  Credential '$VOBIZ_SIP_USERNAME' already exists — skipping."
  else
    echo "  WARNING: Unexpected response creating credential:"
    echo "  $CRED_RESP"
  fi
else
  echo "  Created credential: $CRED_ID"

  # Link the credential to the trunk
  echo "  Linking credential to trunk..."
  vobiz_api PUT "/account/${ACCOUNT_ID}/trunks/${TRUNK_ID}" \
    "{\"credential_uuid\": \"${CRED_ID}\"}" > /dev/null
  echo "  Credential linked."
fi
echo ""

# ── Step 5: Summary ───────────────────────────────────────────────────────────
echo "============================================================"
echo " Vobiz configuration complete!"
echo ""
echo " Inbound flow (Phone → Vobiz → LiveKit → Agent):"
echo "   DID $VOBIZ_PHONE_NUMBER  →  $LIVEKIT_SIP_HOST"
echo ""
echo " Outbound flow (Agent → LiveKit → Vobiz → Phone):"
echo "   LiveKit outbound trunk address: $VOBIZ_SIP_DOMAIN"
echo "   SIP auth: $VOBIZ_SIP_USERNAME / [password set]"
echo ""
echo " Verify with:"
echo "   curl -s -X GET '${VOBIZ_API}/account/${ACCOUNT_ID}/trunks/${TRUNK_ID}' \\"
echo "     -H 'X-Auth-ID: ${VOBIZ_AUTH_ID}' \\"
echo "     -H 'X-Auth-Token: ${VOBIZ_AUTH_TOKEN}' | jq '{trunk_domain,inbound_destination,trunk_status}'"
echo ""
echo " Then test: call $VOBIZ_PHONE_NUMBER — agent should greet you."
echo "============================================================"
