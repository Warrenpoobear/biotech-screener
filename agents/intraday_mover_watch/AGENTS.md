# AGENTS.md — Intraday Mover Watch Agent

## Upstream dependencies

- **`price_action_watch`** — shares `common.watchlist_config.WATCHLIST_MAX` and the watchlist constructor. This agent is the real-time intraday counterpart; do not fork the watchlist definition here.
- **Herald (Specs 044 / 053)** — `artifacts/herald/classified/*.json` and `artifacts/herald/raw/*.json` are the authoritative same-day official news source. Missing Herald artifacts produce `news_status=UNKNOWN_SOURCE_STATE`, not silent skipping.
- **`grok_biotech_watch`** — `artifacts/grok_biotech_watch/{date}_watch.json` provides supporting/unverified enrichment only. Grok-only evidence is never labeled OFFICIAL.
- **Rankings snapshot** — `data/snapshots/{date}/rankings.csv` drives actionable_rank, tier, catalyst_days for watchlist pruning and row metadata.

## Downstream consumers

- **Operator email** — immediate HIGH alerts and EOD digest via the Herald SMTP recipients. No programmatic consumer.
- **OpenClaw verdicts** — cron wrapper reads `artifacts/intraday_mover_watch/*_poll.json` and emits `OK / WARN / ACTION REQUIRED / NO DATA / FAIL` lines.

## Must not touch

- `production_data/*`
- `data/snapshots/{date}/rankings.csv` (read-only)
- `artifacts/live_shadow/*` (read-only)
- `artifacts/catalyst_delta/*` (read-only)
- `specs/*` (this agent doesn't edit specs; amend Spec 063 via a new spec)
- Any scoring, ranker, selector, or decision engine code
- Event EV engine, event ledger, or CRT records

## Read-only invariants

1. No mutation of `production_data/` or `data/snapshots/`.
2. No scoring feature is derived from intraday quotes.
3. No Grok-only evidence labeled `OFFICIAL`.
4. No yfinance / `market_data_provider.py` path is the primary live source.
5. Every email has a corresponding artifact on disk.
6. Email send is no-op unless API key present AND real-time tier confirmed.

## Self-learning (Rule 12)

Recurring quote/news linkage miss → `.learnings/LEARNINGS.md`.
