# autorem

A coding agent built on Claude (`claude-opus-5-5`). It can read and edit files in one workspace folder, and it searches the live web with [Exa](https://exa.ai) to look up current docs and APIs.

## Setup

```bash
pip install -r requirements.txt
```

Credentials:

- `EXA_API_KEY`: get it from https://dashboard.exa.ai/api-keys. In Claude Code, paste it into `.claude/settings.local.json` under `env` (git ignores that file). To run the agent from your own terminal, also set it as a user environment variable: `setx EXA_API_KEY "<key>"`, then open a new terminal.
- `ANTHROPIC_API_KEY`: set it the same way, or sign in with `ant auth login`.
- `DEVIN_API_KEY` (optional): turns on the Devin tools. Create a service user in Devin under **Settings > Devin API** with the Member role (it needs the `UseDevinSessions` permission) and copy its `cog_` key. The key is shown only once. A personal access token works too. Set it the same way as the other keys.
  - `DEVIN_ORG_ID` (optional): the org ID shown on that same page. If you leave it out, autorem looks it up from the key with `GET /v3/self`.
  - `DEVIN_MAX_ACU_LIMIT` (optional): the default ACU cap for each session Claude starts.

## Run

```bash
python -m autorem.agent --workspace path/to/project "upgrade the httpx calls to the current API"
python -m autorem.agent --workspace path/to/project      # interactive
```

## How it works

- `autorem/agent.py` runs the agent loop on the Anthropic Messages API: streaming, adaptive thinking, `high` effort, prompt caching, and server-side refusal fallback.
- Web tools come from the `exa-py` Claude helpers (`exa.anthropic.web_search()` and `get_contents()`). Search uses Exa's recommended request (`type: "auto"` with highlights), and Claude writes a `query` and an `objective` for every search.
- File edits use Claude's built-in text editor tool. `autorem/editor.py` handles it and blocks any path outside `--workspace`.
- When `DEVIN_API_KEY` is set, Claude can hand tasks to [Devin](https://docs.devin.ai/api-reference/overview), Cognition's AI software engineer, through the v3 API. `autorem/devin.py` is the client, and it adds three tools:
  - `devin_create_session` starts a session from a prompt and returns the session ID and URL.
  - `devin_get_session` returns the status, ACUs used, PRs, and Devin's recent messages.
  - `devin_send_message` sends a follow-up message.

  Devin runs on its own and every session uses paid ACUs. So the tools never wait for Devin to finish, and Claude uses them only when you ask it to delegate. Without the key, these tools aren't registered.
- Tests: `python -m tests.test_devin` runs offline. It mocks HTTP, so it makes no real Devin calls.
- `.claude/skills/build-with-exa` is Exa's integration skill. Claude Code uses it as the reference when changing the Exa code.
