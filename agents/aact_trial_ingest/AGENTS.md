# AGENTS.md — AACT Trial Ingest Agent

## Upstream

- AACT database (daily dump from aact.ctti-clinicaltrials.org)
- Sponsor alias registry (`production_data/sponsor_alias_map.json`)
- Manual override mappings (`production_data/aact_manual_overrides.json`)

## Downstream consumers

- `src/providers/aact_provider.py` — PIT-safe query layer for DEM pipeline
- `tools/catalyst_resolution_tracker.py` — CRT trial-status context
- `common/milestone_optionality.py` — milestone timing support
- `tools/poll_ctgov_daily.py` — baseline snapshot for daily delta detection
- Dashboard — trial status views, sponsor execution context
- DEM research feature builders — timing priors, sponsor execution priors

## Self-learning (Rule 12)

Ingest/linkage failures → `.learnings/LEARNINGS.md` with `Promotion-lane: skill`.

## Peer agents

- `ctgov_poller` — real-time daily API polling (complements, does not replace this agent)
- `company_news_ingest` — PR collection lane (different source entirely)
- `calibration` — consumes trial deltas for CRT context
