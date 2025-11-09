# Using the LiveKit Docs MCP server in this workspace

This workspace is pre-configured to register the LiveKit Docs MCP server at:

  https://docs.livekit.io/mcp

What I added
- `.vscode/settings.json` — contains server entries for several MCP-related extensions (so the workspace will be ready after you install an MCP extension).
- `.vscode/tasks.json` — helper tasks to list extensions and remind you to reload the window.
- `mcp.json` — contains the server_url key (already created earlier).

Quick steps to register and connect in VS Code

1. Install an MCP extension (you already installed `nickeolofsson.remember-mcp-vscode`).
2. Close all VS Code windows and re-open VS Code to ensure the process inherits updated workspace settings.
3. Reload the window (Command Palette → "Developer: Reload Window").
4. Open the extension you installed (Extensions view) and check whether it lists `livekit-docs` or `https://docs.livekit.io/mcp`.
   - Many extensions read workspace settings automatically. If the extension does not list the server, use its UI command such as "Add MCP server" and paste the URL `https://docs.livekit.io/mcp`.

Starting / stopping a local MCP server

- The LiveKit Docs MCP server is hosted remotely at the URL above. You don't need to run a local server to use it.
- If you do want to run a local MCP server (for development), add or replace a start command in `.vscode/tasks.json`. Example placeholder task (you must replace the command with your local server start command):

```json
{
  "label": "Start local MCP server (example)",
  "type": "shell",
  "command": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
  "args": ["-m", "your_local_mcp_module", "--config", "mcp.json"]
}
```

How to confirm the server is reachable

- Use the extension UI to run a simple query or check server health (extensions provide different commands).
- From a terminal you can `curl` the server base URL to see if it responds (some MCP servers provide an info endpoint):

```powershell
curl https://docs.livekit.io/mcp
```

Troubleshooting

- Extension doesn't show the server after reload: open the extension's README in the Extensions view to find the exact setting name or command to register a server and add the URL manually.
- Workspace settings were changed but integrated terminal still shows old environment: close all VS Code windows and reopen so the editor process picks up changes.
- If an extension expects a different JSON structure for servers, tell me the extension name and I will update `.vscode/settings.json` accordingly.

If you'd like, I can:
- Update `.vscode/settings.json` to include the exact setting key for the extension you're using (tell me which one you prefer),
- Add a task that starts a local MCP server with the exact command you prefer (provide the command),
- Or demonstrate a quick test query against the MCP server from this environment.
