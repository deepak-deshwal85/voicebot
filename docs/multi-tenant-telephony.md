# Multi-Tenant Telephony

One worker (`AGENT_NAME=telephone-agent`) serves all clients. Tenant is chosen **per session** from the SIP trunk phone number.

## Layout

```text
config/
├── tenant-map.json      # +911171366880 -> client-1
├── client-1.properties  # prompts
├── client-1.json        # combined knowledge base
├── client-2.properties
└── client-2.json
```

## Session flow

1. LiveKit dispatches a job to `telephone-agent`
2. Worker reads `participant.attributes["sip.trunkPhoneNumber"]`
3. Looks up client in `config/tenant-map.json`
4. Loads `config/{client}.properties` and `config/{client}.json`

## Build knowledge (outside agent)

```bash
# Put PDFs in knowledge-sources/client-1/
uv run python scripts/knowledge.py build --client client-1
uv run python scripts/knowledge.py validate --client client-1
```

GitHub Actions workflow **Knowledge Refresh** rebuilds and commits updated `config/*.json` files.

## Local testing

```bash
uv run python src/agent.py console                    # uses DEFAULT_CLIENT_ID
TENANT_PHONE_OVERRIDE=+911171366881 uv run python src/agent.py dev
uv run python scripts/test_tenant_resolution.py --list
```
