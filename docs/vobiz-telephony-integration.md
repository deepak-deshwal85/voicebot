# Vobiz Telephony Integration (SIP -> LiveKit -> Agent)

This project is now telephony-aware and can accept SIP calls routed through LiveKit.

## What was integrated in code

- `src/agent.py` now auto-detects SIP participants.
- Telephony sessions use telephony-optimized input settings when available.
- SIP metadata (caller number, dialed number, trunk id) is captured and logged.
- Agent instructions are adapted for short, phone-friendly responses during SIP calls.

## 1. Configure environment

Start from `.env.example`, then update `.env.local`:

```env
LIVEKIT_URL=wss://<your-livekit-project>.livekit.cloud
LIVEKIT_API_KEY=<your_api_key>
LIVEKIT_API_SECRET=<your_api_secret>

TELEPHONY_PROVIDER=vobiz
TELEPHONY_MODE=auto

VOBIZ_SIP_TRUNK_NAME=vobiz-inbound
VOBIZ_PHONE_NUMBER=<your_vobiz_number>
VOBIZ_SIP_DOMAIN=<vobiz_sip_domain>
VOBIZ_SIP_USERNAME=<vobiz_sip_username>
VOBIZ_SIP_PASSWORD=<vobiz_sip_password>
```

## 2. Create inbound SIP trunk in LiveKit

Use your Vobiz SIP gateway details to create an inbound trunk in LiveKit.

Minimum values to map from Vobiz:
- SIP domain or host
- SIP username
- SIP password
- Phone number (DID)

Use either:
- LiveKit Cloud dashboard: `Telephony -> SIP Trunks -> Create inbound trunk`
- LiveKit CLI: run `lk sip --help`, then create an inbound trunk using your Vobiz SIP credentials and number.

## 3. Create a dispatch rule to this agent

Create a dispatch rule so incoming calls on that trunk are routed to this worker (`agent_name="telephone-agent"` in `src/agent.py`).

Use either:
- LiveKit Cloud dashboard: `Telephony -> Dispatch Rules`
- LiveKit CLI: run `lk sip dispatch --help` and create a rule targeting this agent.

## 4. Run the worker for telephony

```powershell
uv sync
uv run python src/agent.py dev
```

## 5. Validate a call

Call your Vobiz number and check worker logs.

Expected log line format:

```text
SIP call started provider=vobiz caller=<number> dialed=<number> trunk=<trunk-id>
```

If you see this line, telephony integration is active and the call reached the agent.

## Troubleshooting

- Calls not reaching agent:
  - Verify trunk is `active` in LiveKit.
  - Verify dispatch rule points to `telephone-agent`.
  - Verify Vobiz SIP credentials and DID mapping.
- Audio quality issues:
  - Keep `TELEPHONY_MODE=auto` (or set `on` if all sessions are phone calls).
- Auth failures:
  - Re-check Vobiz SIP username/password and allowed source settings on Vobiz.

## Optional next step: outbound calling

If you also want the bot to place outbound calls through Vobiz, add an outbound SIP trunk and use LiveKit outbound call APIs/CLI. The current integration is ready for inbound calls.