# Multi-Tenant Telephony Architecture

One **agent worker** runs on your infrastructure and connects to **LiveKit Cloud**. Incoming SIP calls are routed by LiveKit; the worker selects the tenant client config from the trunk phone number on the SIP participant.

## Architecture

```text
Caller -> Vobiz/PSTN -> LiveKit Cloud SIP -> dispatch rule -> voice-agent worker (your infra)
                                                                    |
                                                    sip.trunkPhoneNumber
                                                                    |
                                                    config/clients/{client-id}/
```

| Component | Location | Role |
|-----------|----------|------|
| LiveKit server | LiveKit Cloud | Rooms, SIP trunks, dispatch rules |
| Agent worker | Your infra (Oracle Cloud, Docker, VM) | Single `voice-agent` process |
| Client config | Repo `config/clients/` | Prompts, KB paths, `telephony.phone_number` |
| Knowledge base | Repo `data/clients/` | Per-tenant embeddings JSON |

## Tenant routing

On each job, the worker:

1. Connects to the LiveKit room
2. Reads `participant.attributes.get("sip.trunkPhoneNumber")` from the SIP caller
3. Maps that number to a client via `telephony.phone_number` in `config/clients/{client-id}/agent.properties`
4. Loads that client's prompts and knowledge store for the session

Configure one inbound SIP trunk **per tenant** in LiveKit Cloud. Each trunk's phone number must match the client's `telephony.phone_number`.

## Configuration

### Worker (`.env.local`)

```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
OPENAI_API_KEY=...

AGENT_NAME=voice-agent
DEFAULT_CLIENT_ID=client-1
```

### Per client (`config/clients/client-1/agent.properties`)

```properties
client.id=client-1
telephony.phone_number=+911171366880
website.name=Fidelity UK
...
```

Repeat for `client-2` with a different `telephony.phone_number`.

### LiveKit Cloud

1. **Inbound SIP trunk** per DID (Telephony → SIP Trunks)
2. **Dispatch rule** targeting `voice-agent` (must match `AGENT_NAME`)
3. Phone number on each trunk must match `telephony.phone_number` in the matching client config

Use `bash scripts/setup-sip.sh` to create **inbound trunks for all tenants** and one dispatch rule:

```bash
bash scripts/setup-sip.sh              # create missing trunks + dispatch rule
bash scripts/setup-sip.sh --fresh      # delete and recreate everything
bash scripts/setup-sip.sh --dry-run    # preview lk commands
```

The script reads `telephony.phone_number` from every `config/clients/*/agent.properties` file.

## Local testing (before Oracle Cloud deploy)

### 1. Verify phone → client mapping

```bash
uv run python scripts/test_tenant_resolution.py --list
uv run python scripts/test_tenant_resolution.py --phone +911171366880
```

### 2. Console mode (no SIP, default tenant)

Uses `DEFAULT_CLIENT_ID` when no SIP participant is present:

```bash
uv run python src/agent.py console
```

### 3. Dev mode with simulated trunk phone

Test a specific tenant without placing a call:

```bash
export TENANT_PHONE_OVERRIDE=+911171366881
uv run python src/agent.py dev
```

Join from the LiveKit playground; the worker loads `client-2` config.

### 4. Smoke test knowledge store per client

```bash
DEFAULT_CLIENT_ID=client-1 uv run python scripts/smoke_test_agent.py
# Or edit smoke test / use knowledge CLI:
uv run python scripts/knowledge.py search "pension" --client client-2
```

### 5. Full telephony test (recommended before production)

Terminal 1 — start worker locally:

```bash
uv run python src/agent.py dev
```

Ensure LiveKit dispatch rule points to `voice-agent` and your local worker is registered (dev mode connects to the same LiveKit project).

Call the Vobiz number configured for `client-1`. Check logs for:

```text
SIP session client=client-1 trunk_phone=+911171366880 room=call-...
```

Call the second tenant's number and confirm `client=client-2`.

### 6. Run unit tests

```bash
uv run pytest tests/test_tenant.py -v
uv run pytest -v
```

## Deploy to Oracle Cloud (self-hosted worker)

LiveKit stays on LiveKit Cloud; only the **worker** runs on OCI.

### Build image (includes all tenants)

```bash
docker build -t voicebot .
```

### Run on OCI VM / Container Instances

```bash
docker run -d --name voicebot \
  --env-file .env.local \
  -e AGENT_NAME=voice-agent \
  -e DEFAULT_CLIENT_ID=client-1 \
  voicebot
```

The container runs `uv run src/agent.py start`, registers as `voice-agent`, and waits for jobs from LiveKit Cloud.

Requirements on OCI:

- Outbound HTTPS to LiveKit Cloud and model providers
- Sufficient CPU/RAM for STT/LLM/TTS pipeline (2+ vCPU recommended)
- `.env.local` with LiveKit credentials and `OPENAI_API_KEY`
- All client knowledge JSON files baked into the image or mounted as a volume

### Register worker with LiveKit

Your worker must connect to the **same** LiveKit project where SIP trunks and dispatch rules are configured. No separate "agent deploy" is required for self-hosted workers if you run the Docker container yourself — only ensure `AGENT_NAME=voice-agent` matches the dispatch rule.

Alternatively, use `lk agent deploy` from CI to LiveKit Cloud-hosted workers; for OCI self-host, run the container directly.

## Outbound calls

```bash
uv run python scripts/outbound_call.py +919868402577 --agent voice-agent
```

Pass `--room` with a unique name per call. Tenant for outbound is not inferred from trunk phone automatically today; use room metadata or extend dispatch if needed.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| Call reaches LiveKit but no agent | Dispatch rule `agents` includes `voice-agent`; worker is running and connected |
| Wrong tenant KB / prompts | `telephony.phone_number` matches LiveKit trunk DID; check logs for `trunk_phone` |
| `No client configured for trunk phone` | Add or fix `telephony.phone_number` in client config |
| Console/dev uses wrong client | Set `DEFAULT_CLIENT_ID` or `TENANT_PHONE_OVERRIDE` |
