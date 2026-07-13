# Quickstart

A tiny proxy that lets **Codex / Codex++** talk to any OpenAI-compatible upstream, keeping automatic tool calls working.

## 1. Install

```bat
pip install -r requirements.txt
```

## 2. Configure

Copy `.env.example` to `.env` and fill it in:

```ini
UPSTREAM_BASE_URL=https://api.your-provider.com/v1   # no trailing /responses
UPSTREAM_API_KEY=sk-xxxx
PROXY_PORT=8123
UPSTREAM_MODEL=gpt-5.6-luna    # real upstream model (leave empty to pass through)
UPSTREAM_WIRE_API=responses    # responses (default) | chat
```

- `UPSTREAM_WIRE_API=responses` — upstream natively supports `/v1/responses` (recommended, keeps auto tool calls).
- `UPSTREAM_WIRE_API=chat` — upstream only has `/v1/chat/completions`.

## 3. Run

```bat
start.bat
```

Check it: open `http://127.0.0.1:8123/health` → `{"ok": true, ...}`

## 4. Point Codex at it

Set Codex's base URL to:

```
http://127.0.0.1:8123/v1
```

**Model name tip:** Codex only attaches its tools for names it recognizes (e.g. `gpt-5-codex`). So set a recognized name in Codex, and use `UPSTREAM_MODEL` to rewrite it to your real upstream model. The model Codex shows is just a label — the one that actually runs is `UPSTREAM_MODEL`.

Done.
