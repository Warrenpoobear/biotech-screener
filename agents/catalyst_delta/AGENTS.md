# AGENTS.md — Catalyst Delta Agent

## Session startup

1. Read `SOUL.md` — your identity and boundaries
2. Read `TOOLS.md` — commands and data sources
3. Read `memory/YYYY-MM-DD.md` (today + yesterday) if they exist

## Daily sequence

1. Identify today's date and the latest snapshot date
2. Load prior delta: `artifacts/catalyst_delta/{prior_date}_delta.json`
3. Compare current event artifacts against prior:
   - `data/snapshots/{date}/catalyst_source_mix.json`
   - `data/snapshots/{date}/catalyst_shadow_metrics.json`
   - Cache state in `cache/ctgov/`, `cache/sec_8k/`
4. For each changed event, join to model context:
   - `data/snapshots/{date}/rankings.csv` → tier, rank, catalyst_days, is_hard
   - `artifacts/live_shadow/positions/{date}.json` → in shadow?
   - `artifacts/live_shadow/trade_plan/{date}/trade_plan.csv` → in trade plan?
5. Classify each change with a structured code
6. Write delta to `artifacts/catalyst_delta/{date}_delta.json` and `.md`
7. Report: count of changes by code, top 5 most impactful (A/B tier or <=30d)

## Noise filter (LLM elevation rule — narrative scope only)

> **Scope: this filter governs which deltas the LLM ELEVATES into the daily
> narrative / memory note. It does NOT govern what the deterministic builder
> writes. `tools/build_catalyst_delta.py` continues to write every delta to
> `artifacts/catalyst_delta/{date}_delta.{json,md}` unchanged. Downstream
> consumers (`tools/build_options_watch.py`, audits) read the raw artifact and
> are unaffected.**

Elevate (i.e., name explicitly in the LLM narrative) only deltas that meet ALL
of the following — narrowed 2026-05-06 (P1 #2, audit/reporting noise reduction):

- ticker is **in-universe** (present in today's `rankings.csv`); AND
- `catalyst_days <= 60` (event is within the 60-day actionable window); AND
- **change code is one of:** `NEW_HARD_EVENT`, `HARD_EVENT_DATE_CHANGE`,
  `FDA_EVENT_NEW`, `SEC_EVENT_NEW` (HARD events), OR
  `SOURCE_FAMILY_CHANGE`, `TRIAL_STATUS_CHANGE` (family-changing codes).

Soft-event date changes for non-elevated tickers are **NOT** named individually.

### Rollup for suppressed deltas (do not erase — summarize)

For deltas that fail the elevation rule, the LLM narrative MUST still include a
rollup-summary line, e.g.:

```
Suppressed (not individually elevated): N deltas — by code:
  SOFT_EVENT_DATE_CHANGE: N1, TRIAL_STATUS_CHANGE (out-of-window): N2, ...
```

The rollup ensures information is not silently dropped — the raw counts and
codes remain visible for audit. The full per-ticker detail remains in the
deterministic JSON/MD artifact.

The **prior wider filter** (A/B tier OR catalyst ≤30d OR source-family-changed
OR in shadow/trade-plan) is intentionally retained as historical context here
for reviewers comparing old and new behavior:

```
Prior elevation rule (replaced 2026-05-06):
- A or B tier in current rankings
- Catalyst <=30 days away
- Source family changed (hard→soft or soft→hard)
- In the active shadow portfolio or trade plan
```

## Memory

Write daily notes to `memory/YYYY-MM-DD.md`. Keep it concise:
- Number of changes detected, by code
- Names that crossed the noise filter
- Any source-level anomalies (e.g., SEC feed went dark)

## Self-learning (Rule 12)

Signal/scoring delta patterns → `.learnings/LEARNINGS.md` with `Promotion-lane: spec`.

## Red lines

- Do not edit `.py` files, scoring logic, rulesets, or manifest
- Do not `git push`, `git commit`, or modify tracked files
- Do not change event classifications or source priorities
- When in doubt, report the event change and wait for human decision
