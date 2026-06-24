# AGENTS.md — Grok Biotech Watch

## Session startup

1. Read `SOUL.md` — identity and boundaries
2. Read `TOOLS.md` — data sources and query design
3. Check environment: XAI_API_KEY, SMTP credentials
4. Load today's snapshot and watchlist

## Daily sequence

### Polling mode (every 10-15 min during market hours)

1. Build watchlist from current snapshot (shadow + review queue + trade plan + near-term catalysts)
2. Query xAI Grok for each watchlist name (ticker + catalyst terms)
3. Filter: discard low-relevance, dedupe against dedup_state.json
4. Classify severity: HIGH / MEDIUM / LOW
5. Enrich each match with DEM context (tier, rank, catalyst_days, policy status)
6. If any HIGH alerts: send immediate email
7. Write artifacts (JSON + MD)

### Daily digest (5:30 PM ET, after production run)

1. Aggregate all alerts from the day
2. Group by ticker, dedupe, sort by severity
3. Send digest email with all severities
4. Rotate dedup state (drop entries > 24h old)

## Memory protocol

Write session summaries to `agents/grok_biotech_watch/memory/`.
Track: query volume, rate limit hits, alert counts by severity,
false positive rates (manually noted by operator).

## Self-learning (Rule 12)

Recurring search/dedupe issue → `.learnings/LEARNINGS.md`.

## Red lines

- Do not treat search results as confirmed events
- Do not feed search data into scoring, rankings, or event ledger
- Do not modify any `.py` file, ruleset, or production data
- Do not send more than 5 immediate emails per hour (throttle)
