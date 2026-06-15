<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# LiveKit Agents Starter - Python

A complete starter project for building voice AI apps with [LiveKit Agents for Python](https://github.com/livekit/agents) and [LiveKit Cloud](https://cloud.livekit.io/).

The starter project includes:

- A simple voice AI assistant, ready for extension and customization
- A voice AI pipeline with [models](https://docs.livekit.io/agents/models) from OpenAI, Cartesia, and AssemblyAI served through LiveKit Cloud
  - Easily integrate your preferred [LLM](https://docs.livekit.io/agents/models/llm/), [STT](https://docs.livekit.io/agents/models/stt/), and [TTS](https://docs.livekit.io/agents/models/tts/) instead, or swap to a realtime model like the [OpenAI Realtime API](https://docs.livekit.io/agents/models/realtime/openai)
- Eval suite based on the LiveKit Agents [testing & evaluation framework](https://docs.livekit.io/agents/build/testing/)
- [LiveKit Turn Detector](https://docs.livekit.io/agents/build/turns/turn-detector/) for contextually-aware speaker detection, with multilingual support
- [Background voice cancellation](https://docs.livekit.io/home/cloud/noise-cancellation/)
- Integrated [metrics and logging](https://docs.livekit.io/agents/build/metrics/)
- A Dockerfile ready for [production deployment](https://docs.livekit.io/agents/ops/deployment/)

This starter app is compatible with any [custom web/mobile frontend](https://docs.livekit.io/agents/start/frontend/) or [SIP-based telephony](https://docs.livekit.io/agents/start/telephony/).

## Coding agents and MCP

This project is designed to work with coding agents like [Cursor](https://www.cursor.com/) and [Claude Code](https://www.anthropic.com/claude-code). 

To get the most out of these tools, install the [LiveKit Docs MCP server](https://docs.livekit.io/mcp).

For Cursor, use this link:

[![Install MCP Server](https://cursor.com/deeplink/mcp-install-light.svg)](https://cursor.com/en-US/install-mcp?name=livekit-docs&config=eyJ1cmwiOiJodHRwczovL2RvY3MubGl2ZWtpdC5pby9tY3AifQ%3D%3D)

For Claude Code, run this command:

```
claude mcp add --transport http livekit-docs https://docs.livekit.io/mcp
```

For Codex CLI, use this command to install the server:
```
codex mcp add --url https://docs.livekit.io/mcp livekit-docs
```

For Gemini CLI, use this command to install the server:
```
gemini mcp add --transport http livekit-docs https://docs.livekit.io/mcp
```

The project includes a complete [AGENTS.md](AGENTS.md) file for these assistants. You can modify this file  your needs. To learn more about this file, see [https://agents.md](https://agents.md).

## Dev Setup

Clone the repository and install dependencies:

```console
cd voicebot
uv sync
```

Copy `.env.example` to `.env.local` and fill in the required values. At minimum you need:

| Variable | Purpose |
|----------|---------|
| `LIVEKIT_URL` | LiveKit Cloud WebSocket URL |
| `LIVEKIT_API_KEY` | LiveKit API key |
| `LIVEKIT_API_SECRET` | LiveKit API secret |
| `AGENT_NAME` | Worker name registered with LiveKit (`voice-agent`) |
| `DEFAULT_CLIENT_ID` | Fallback tenant for console/dev when no SIP call |
| `OPENAI_API_KEY` | Embeddings for knowledge base search (optional but recommended) |

Example `.env.local`:

```env
AGENT_NAME=voice-agent
DEFAULT_CLIENT_ID=client-1
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
OPENAI_API_KEY=...
EMBEDDING_MODEL=text-embedding-3-small
```

Simulate a tenant locally without SIP:

```env
TENANT_PHONE_OVERRIDE=+911171366881
```

You can load LiveKit credentials automatically with the [LiveKit CLI](https://docs.livekit.io/home/cli/cli-setup):

```bash
lk cloud auth
lk app env -w -d .env.local
```

Then add `AGENT_NAME`, `DEFAULT_CLIENT_ID`, and `OPENAI_API_KEY` to `.env.local`.

## Multi-tenant architecture

A **single agent worker** (`AGENT_NAME=voice-agent`) runs on your infrastructure and connects to **LiveKit Cloud**. Each tenant (client) has its own config and knowledge base; the worker picks the tenant at runtime from the SIP trunk phone number:

```text
participant.attributes.get("sip.trunkPhoneNumber")  →  client config  →  knowledge store
```

```text
config/clients/
├── client-1/agent.properties   → telephony.phone_number=+911171366880
└── client-2/agent.properties   → telephony.phone_number=+911171366881

data/clients/
├── client-1/                   → PDFs + knowledge JSON stores
└── client-2/
```

Set `telephony.phone_number` in each client config to match the DID on that tenant's **LiveKit inbound SIP trunk**. See [docs/multi-tenant-telephony.md](docs/multi-tenant-telephony.md) for local testing and Oracle Cloud deployment.

Edit prompts and URLs in `config/clients/{client-id}/agent.properties`. Place PDFs in `data/clients/{client-id}/`.

## Knowledge base utility

Knowledge is managed **outside** the voice agent with `scripts/knowledge.py`:

```bash
# List clients
uv run python scripts/knowledge.py list-clients

# Status
uv run python scripts/knowledge.py status --client client-1

# Rebuild website + PDFs for one client
uv run python scripts/knowledge.py refresh all --client client-1

# Rebuild website only (PDF store unchanged)
uv run python scripts/knowledge.py refresh website --client client-2

# Rebuild PDFs only
uv run python scripts/knowledge.py refresh pdfs --client client-1

# Test retrieval
uv run python scripts/knowledge.py search "pension transfer" --client client-1

# Validate before deploy
uv run python scripts/knowledge.py validate --client client-1
```

Task shortcuts:

```bash
task kb-clients
task kb-status CLIENT=client-1
task refresh-knowledge CLIENT=client-1
task kb-validate CLIENT=client-2
```

Rebuild knowledge after changing PDFs or website content, then restart the agent.

## Run the agent

Before your first run, you must download certain models such as [Silero VAD](https://docs.livekit.io/agents/build/turns/vad/) and the [LiveKit turn detector](https://docs.livekit.io/agents/build/turns/turn-detector/):

```console
uv run python src/agent.py download-files
```

Next, run this command to speak to your agent directly in your terminal:

```console
uv run python src/agent.py console
```

To run the agent for use with a frontend or telephony, use the `dev` command:

```console
uv run python src/agent.py dev
```

In production, use the `start` command:

```console
uv run python src/agent.py start
```

The worker registers as `AGENT_NAME` (default `voice-agent`). Tenant config is chosen per session from the SIP trunk phone number.

### Docker (single multi-tenant image)

```bash
docker build -t voicebot .
docker run --env-file .env.local voicebot
```

See [docs/multi-tenant-telephony.md](docs/multi-tenant-telephony.md) for Oracle Cloud deployment.

## Frontend & Telephony

Get started quickly with our pre-built frontend starter apps, or add telephony support:

| Platform | Link | Description |
|----------|----------|-------------|
| **Web** | [`livekit-examples/agent-starter-react`](https://github.com/livekit-examples/agent-starter-react) | Web voice AI assistant with React & Next.js |
| **iOS/macOS** | [`livekit-examples/agent-starter-swift`](https://github.com/livekit-examples/agent-starter-swift) | Native iOS, macOS, and visionOS voice AI assistant |
| **Flutter** | [`livekit-examples/agent-starter-flutter`](https://github.com/livekit-examples/agent-starter-flutter) | Cross-platform voice AI assistant app |
| **React Native** | [`livekit-examples/voice-assistant-react-native`](https://github.com/livekit-examples/voice-assistant-react-native) | Native mobile app with React Native & Expo |
| **Android** | [`livekit-examples/agent-starter-android`](https://github.com/livekit-examples/agent-starter-android) | Native Android app with Kotlin & Jetpack Compose |
| **Web Embed** | [`livekit-examples/agent-starter-embed`](https://github.com/livekit-examples/agent-starter-embed) | Voice AI widget for any website |
| **Telephony** | [📚 Documentation](https://docs.livekit.io/agents/start/telephony/) | Add inbound or outbound calling to your agent |

For advanced customization, see the [complete frontend guide](https://docs.livekit.io/agents/start/frontend/).

For Vobiz SIP integration, see [docs/vobiz-telephony-integration.md](docs/vobiz-telephony-integration.md) and [docs/multi-tenant-telephony.md](docs/multi-tenant-telephony.md).

## Tests and evals

This project includes a complete suite of evals, based on the LiveKit Agents [testing & evaluation framework](https://docs.livekit.io/agents/build/testing/). To run them:

```console
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## CI/CD (GitHub Actions)

Workflows live in `.github/workflows/`.

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **CI** (`ci.yml`) | Push / PR to `main` | Ruff lint, pytest, client config validation |
| **Knowledge Refresh** (`knowledge-refresh.yml`) | Manual | Rebuild website/PDF knowledge for `client-1`, `client-2`, or `all` |
| **Deploy Agent** (`deploy-agent.yml`) | Manual | Deploy multi-tenant worker with `lk agent deploy` |

### GitHub repository secrets

Add these under **Settings → Secrets and variables → Actions**:

| Secret | Required for |
|--------|----------------|
| `LIVEKIT_URL` | CI tests, deploy |
| `LIVEKIT_API_KEY` | CI tests, deploy |
| `LIVEKIT_API_SECRET` | CI tests, deploy |
| `OPENAI_API_KEY` | Knowledge refresh workflow (embeddings) |

### Refresh knowledge in CI

1. Open **Actions → Knowledge Refresh → Run workflow**
2. Choose `client` (`client-1`, `client-2`, or `all`)
3. Choose `source` (`all`, `website`, or `pdfs`)
4. Set `max_pages` for website crawl (default `100`)

Refreshed JSON files are uploaded as workflow artifacts.

### Deploy the agent worker

1. Ensure `livekit.toml` is **committed to git** (must contain your agent `id`).
2. Ensure knowledge stores are built for all tenants (`knowledge.py validate --client client-1`).
3. Add GitHub secrets: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `OPENAI_API_KEY`.
4. Open **Actions → Deploy Agent → Run workflow**.

The deploy workflow runs `lk agent deploy` with:

- `OPENAI_API_KEY`
- `AGENT_NAME=voice-agent`
- `DEFAULT_CLIENT_ID=client-1`
- `EMBEDDING_MODEL`

For **self-hosted workers on Oracle Cloud**, build and run the Docker image directly instead of `lk agent deploy`. See [docs/multi-tenant-telephony.md](docs/multi-tenant-telephony.md).

## Using this template repo for your own project

Once you've started your own project based on this repo, you should:

1. **Check in your `uv.lock`**: This file is currently untracked for the template, but you should commit it to your repository for reproducible builds and proper configuration management. (The same applies to `livekit.toml`, if you run your agents in LiveKit Cloud)

2. **Remove the git tracking test**: Delete the "Check files not tracked in git" step from `.github/workflows/tests.yml` since you'll now want this file to be tracked. These are just there for development purposes in the template repo itself.

3. **Add your own repository secrets**: Add `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and `OPENAI_API_KEY` for CI/CD workflows (see [CI/CD (GitHub Actions)](#cicd-github-actions)).

## Deploying to production

This project includes a working `Dockerfile` for a **multi-tenant** worker (all client configs and knowledge stores in one image). For LiveKit Cloud-hosted workers, use the **Deploy Agent** GitHub workflow. For Oracle Cloud or other self-hosted infra, run the Docker container with `--env-file .env.local`. See [docs/multi-tenant-telephony.md](docs/multi-tenant-telephony.md).

Before deploying, refresh and validate the client's knowledge base:

```bash
uv run python scripts/knowledge.py refresh all --client client-1
uv run python scripts/knowledge.py validate --client client-1
```

## Self-hosted LiveKit

You can also self-host LiveKit instead of using LiveKit Cloud. See the [self-hosting](https://docs.livekit.io/home/self-hosting/) guide for more information. If you choose to self-host, you'll need to also use [model plugins](https://docs.livekit.io/agents/models/#plugins) instead of LiveKit Inference and will need to remove the [LiveKit Cloud noise cancellation](https://docs.livekit.io/home/cloud/noise-cancellation/) plugin.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.