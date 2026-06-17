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
| `AGENT_NAME` | Worker name registered with LiveKit (`telephone-agent`) |
| `DEFAULT_CLIENT_ID` | Fallback tenant for console/dev when no SIP call |
| `OPENAI_API_KEY` | Embeddings for knowledge base search (optional but recommended) |
| `PRELOAD_PDF_KNOWLEDGE` | Preload PDF index at call connect (`true` default) |
| `PRELOAD_WEBSITE_KNOWLEDGE` | Preload website index at call connect (`false` default; ~60MB) |

Example `.env.local`:

```env
AGENT_NAME=telephone-agent
DEFAULT_CLIENT_ID=client-1
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
OPENAI_API_KEY=...
EMBEDDING_MODEL=text-embedding-3-small
PRELOAD_PDF_KNOWLEDGE=true
PRELOAD_WEBSITE_KNOWLEDGE=false
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

## Multi-tenant configuration

All client files live in a single `config/` folder:

```text
config/
├── tenant-map.json           # phone number -> client id
├── client-1.properties       # prompts and build settings
├── client-1-website.json     # website index (runtime, lazy-loaded)
├── client-1-pdf.json         # PDF index (runtime, lazy-loaded)
├── client-2.properties
├── client-2-website.json
└── client-2-pdf.json
```

Phone routing uses `config/tenant-map.json`. Knowledge bases are built **outside** the agent and loaded **on demand** at runtime through search tools:

- Financial services, products, fees → `search_website_docs`
- Personal info, education, projects, skills, resume → `search_document_library`

```bash
uv run python scripts/knowledge.py build --client client-1
uv run python scripts/knowledge.py validate --client client-1
uv run python scripts/knowledge.py search --client client-1 "pension transfer"
uv run python scripts/knowledge.py search --client client-1 "what skills are on the resume?"
```

PDFs for building go in `knowledge-sources/client-1/` (not loaded at runtime).

## Knowledge base utility

Knowledge is managed **outside** the voice agent with `scripts/knowledge.py`. The `build` command writes two files per client:

| File | Purpose |
|------|---------|
| `config/{client}-website.json` | Website-only index used by `search_website_docs` |
| `config/{client}-pdf.json` | PDF-only index used by `search_document_library` |

At runtime the agent loads only the index it needs (for example, PDF questions load the small PDF file, not the full website store).

```bash
# List clients
uv run python scripts/knowledge.py list-clients

# Build website + PDF knowledge bases for one client
uv run python scripts/knowledge.py build --client client-1 --max-pages 100

# Validate website and PDF indexes before deploy
uv run python scripts/knowledge.py validate --client client-1

# Test routed retrieval (prints route: website | pdf | both)
uv run python scripts/knowledge.py search --client client-1 "pension transfer" --top-k 3
uv run python scripts/knowledge.py search --client client-1 "what skills are on the resume?" --top-k 3

# Build for all clients (omit --client)
uv run python scripts/knowledge.py build --max-pages 100
uv run python scripts/knowledge.py validate
```

Task shortcuts:

```bash
task kb-clients
task refresh-knowledge CLIENT=client-1
task kb-validate CLIENT=client-2
```

Rebuild knowledge after changing PDFs or website content, then redeploy the agent worker.

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

The worker registers as `AGENT_NAME` (default `telephone-agent`). Tenant config is chosen per session from the SIP trunk phone number.

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
3. Set `max_pages` for website crawl (default `100`)

The updated `config/client-*-website.json` and `config/client-*-pdf.json` files are committed and pushed back to the repo by the workflow.

### Deploy the agent worker

1. Ensure `livekit.toml` is **committed to git** (must contain your agent `id`).
2. Ensure split knowledge stores exist for all tenants:
   ```bash
   uv run python scripts/knowledge.py validate --client client-1
   uv run python scripts/knowledge.py validate --client client-2
   ```
3. Add GitHub secrets: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `OPENAI_API_KEY`.
4. Open **Actions → Deploy Agent → Run workflow**.

Use `lk agent deploy`, not `lk agent create`, if `livekit.toml` already contains an agent id.

The deploy workflow runs `lk agent deploy` with:

- `OPENAI_API_KEY`
- `AGENT_NAME=telephone-agent`
- `DEFAULT_CLIENT_ID=client-1`
- `EMBEDDING_MODEL`
- `PRELOAD_PDF_KNOWLEDGE` (optional, default `true`)
- `PRELOAD_WEBSITE_KNOWLEDGE` (optional, default `false`; set `true` for faster financial Q&A)

For **local deploys to LiveKit Cloud**, pass the same secrets explicitly or query embeddings will fail at runtime:

```bash
lk agent deploy \
  --secrets "OPENAI_API_KEY=$OPENAI_API_KEY" \
  --secrets "AGENT_NAME=telephone-agent" \
  --secrets "DEFAULT_CLIENT_ID=client-1" \
  --secrets "EMBEDDING_MODEL=text-embedding-3-small" \
  --secrets "PRELOAD_PDF_KNOWLEDGE=true" \
  --secrets "PRELOAD_WEBSITE_KNOWLEDGE=true"
```

After deploy, place a test call and confirm logs show tool searches such as `Website search for:` or `PDF search for:` when you ask a question.

For **self-hosted workers on Oracle Cloud**, build and run the Docker image directly instead of `lk agent deploy`. See [docs/multi-tenant-telephony.md](docs/multi-tenant-telephony.md).

## Using this template repo for your own project

Once you've started your own project based on this repo, you should:

1. **Check in your `uv.lock`**: This file is currently untracked for the template, but you should commit it to your repository for reproducible builds and proper configuration management. (The same applies to `livekit.toml`, if you run your agents in LiveKit Cloud)

2. **Remove the git tracking test**: Delete the "Check files not tracked in git" step from `.github/workflows/tests.yml` since you'll now want this file to be tracked. These are just there for development purposes in the template repo itself.

3. **Add your own repository secrets**: Add `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and `OPENAI_API_KEY` for CI/CD workflows (see [CI/CD (GitHub Actions)](#cicd-github-actions)).

## Deploying to production

This project includes a working `Dockerfile` for a **multi-tenant** worker (all client configs and knowledge stores in one image). For LiveKit Cloud-hosted workers, use the **Deploy Agent** GitHub workflow. For Oracle Cloud or other self-hosted infra, run the Docker container with `--env-file .env.local`. See [docs/multi-tenant-telephony.md](docs/multi-tenant-telephony.md).

Before deploying, refresh and validate the client's knowledge bases:

```bash
uv run python scripts/knowledge.py build --client client-1 --max-pages 100
uv run python scripts/knowledge.py validate --client client-1
uv run python scripts/knowledge.py search --client client-1 "pension transfer"
uv run python scripts/knowledge.py search --client client-1 "what skills are on the resume?"
```

## Self-hosted LiveKit

You can also self-host LiveKit instead of using LiveKit Cloud. See the [self-hosting](https://docs.livekit.io/home/self-hosting/) guide for more information. If you choose to self-host, you'll need to also use [model plugins](https://docs.livekit.io/agents/models/#plugins) instead of LiveKit Inference and will need to remove the [LiveKit Cloud noise cancellation](https://docs.livekit.io/home/cloud/noise-cancellation/) plugin.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.