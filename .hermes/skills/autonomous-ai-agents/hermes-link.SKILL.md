---
name: hermes-link
description: Understand and troubleshoot Hermes Link, the local companion service that connects Hermes Agent to the HermesPilot mobile app.
---

# Hermes Link

Hermes Link is a secure local companion service for hermes-agent. It lets the HermesPilot mobile app connect to the user's local Hermes Agent safely and reliably.

Hermes Link is specifically for HermesPilot. It is not OpenClaw, ClawPilot, clawlink, or any OpenClaw companion service. Do not confuse those systems when helping the user.

## Installation & Updates

Hermes Link is a **Node.js** CLI tool, installed from npm:

```bash
# Install (one-time)
npm install -g @hermespilot/link

# Update to latest version
npm update -g @hermespilot/link

# Check installed version
npm ls -g @hermespilot/link
# OR
hermeslink status | head -1
```

### Update Workflow

1. **Check current version** — `hermeslink status` shows version (e.g., `Hermes Link 0.6.7`)
2. **Update** — `npm update -g @hermespilot/link`
3. **Restart the daemon** — `hermeslink restart`
4. **Verify** — `hermeslink status` confirms the new version + daemon healthy + paired status
5. **Check gateway reconnection** — `hermes status --all` should show gateway healthy after link restarts

> Hermes Link is NOT a Python package. Do NOT use `pip install hermes-link`. The binary lives at `~/.npm-global/bin/hermeslink` when installed globally.

## What Hermes Link Does

Hermes Link can:

- pair the HermesPilot mobile app with this computer
- expose a controlled local API for the app
- connect through LAN, public direct routes, or Hermes Relay
- help the app access Hermes profiles, conversations, messages, files, voice input, logs, and status
- start or check the Hermes Gateway when needed

Hermes Link does not replace Hermes Agent. Hermes Agent remains the actual AI agent runtime.

## When To Use This Skill

Use this skill when the user asks about:

- HermesPilot App cannot connect
- Link appears offline
- pairing or QR code problems
- mobile access to local Hermes
- Relay, LAN, or public direct route problems
- Hermes Link logs, daemon status, or diagnostics

## Useful Commands

Start with:

```bash
hermeslink status
hermeslink doctor
```

For logs:

```bash
hermeslink logs --error -n 50
hermeslink logs --warn -n 100
hermeslink logs -f
hermeslink logs --all --level debug -f
```

If the daemon appears stuck, suggest:

```bash
hermeslink restart
```

Explain that restarting Hermes Link may briefly disconnect the mobile app.

## Troubleshooting Flow

1. Check `hermeslink status`.
2. Run `hermeslink doctor`.
3. Inspect recent errors with `hermeslink logs --error -n 50`.
4. If the issue is intermittent, reproduce it while running `hermeslink logs -f`.
5. Check whether the phone and computer are on the same LAN, using Relay, or using a public direct route.
6. If the user uses WSL, Docker, or a VM, verify that Hermes Agent and Hermes Link run in the same environment.

## Safety

Never reveal API keys, access tokens, refresh tokens, private keys, or full .env contents.

Do not recommend exposing port 52379 directly to the public internet without TLS, VPN, Tailscale, WireGuard, or another access-control layer.

Do not modify Hermes profiles, delete user data, edit config files, or kill processes unless the user explicitly asks.
