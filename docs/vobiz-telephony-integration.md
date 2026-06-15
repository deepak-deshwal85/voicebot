# Vobiz Telephony Integration (SIP -> LiveKit Cloud -> Multi-Tenant Worker)

Telephony routes through **LiveKit Cloud SIP**. A single **agent worker** on your infrastructure registers as `voice-agent` and resolves the tenant from the SIP participant attribute `sip.trunkPhoneNumber`.

See [multi-tenant-telephony.md](multi-tenant-telephony.md) for architecture, local testing, and Oracle Cloud deployment.

## What is integrated in code

- `src/agent.py` resolves tenant from `participant.attributes.get("sip.trunkPhoneNumber")`
- Client config is loaded per session from `config/clients/{client-id}/agent.properties`
- `telephony.phone_number` in each client config must match the DID on the LiveKit inbound trunk

## 1. Configure environment

```env
LIVEKIT_URL=wss://<your-livekit-project>.livekit.cloud
LIVEKIT_API_KEY=<your_api_key>
LIVEKIT_API_SECRET=<your_api_secret>

AGENT_NAME=voice-agent
DEFAULT_CLIENT_ID=client-1
OPENAI_API_KEY=...
```

Per client (`config/clients/client-1/agent.properties`):

```properties
telephony.phone_number=+911171366880
```

## 2. Create inbound SIP trunks in LiveKit Cloud

Create **one inbound trunk per tenant** in LiveKit (Telephony → SIP Trunks). Each trunk's phone number must match that client's `telephony.phone_number`.

```bash
bash scripts/setup-sip.sh
bash scripts/setup-sip.sh --fresh --dry-run
```

Or use `lk sip inbound create` / the LiveKit dashboard manually.

## 3. Create a dispatch rule to the multi-tenant worker

Dispatch rule must target `voice-agent` (same as `AGENT_NAME`):

```bash
lk sip dispatch --help
```

The setup script creates a rule with `"agents": ["voice-agent"]`.

## 4. Run the worker

Local dev:

```bash
uv run python src/agent.py dev
```

Production (Docker on your infra):

```bash
docker build -t voicebot .
docker run --env-file .env.local voicebot
```

## 5. Validate a call

Call a tenant's Vobiz number. Expected log line:

```text
SIP session client=client-1 trunk_phone=+911171366880 room=call-...
```

## Local testing without a phone call

```bash
uv run python scripts/test_tenant_resolution.py --phone +911171366880
TENANT_PHONE_OVERRIDE=+911171366880 uv run python src/agent.py dev
```

## Troubleshooting

- **Call reaches LiveKit but no agent**: dispatch rule must list `voice-agent`; worker must be running
- **Wrong tenant**: verify `telephony.phone_number` matches LiveKit trunk DID
- **Unknown phone error**: add mapping in client config for that DID
