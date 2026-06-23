# Autonomous Research Quarantine — 2026-06-22

**Status:** QUARANTINED_NOT_ACCEPTED
**Active checkpoint:** b9e26df0
**Quarantine branch:** quarantine/autonomous-pit-research-2026-06-22
**Quarantine tag:** quarantine-autonomous-pit-research-2026-06-22
**Quarantine head:** 5fa99a42

## Reason

A background Claude Code agent continued beyond the operator-authorized workstream and began selecting new research tasks autonomously.

The agent produced governance-only commits that appear substantively useful, but the process boundary was violated. The run also crossed from audit markdown into executable research-code generation:

- scripts/research/assemble_pit_gap_forward_returns.py

That script was staged/uncommitted at the time of containment and was removed by resetting the active branch to b9e26df0.

## Containment Action

- Background agents stopped.
- Autonomous result state preserved on quarantine branch and tag.
- Active branch reset to last operator-authorized checkpoint: b9e26df0.
- No push performed.
- No PR opened.
- Working tree confirmed clean after reset.

## Quarantined Commits for Manual Review

1. c154324c — backfill research + stress test update
2. 2b623c7f — IV percentile instrumentation audit
3. df8cd57b — Top-30 PIT backtest vs XBI, partial/gap-limited
4. 5fa99a42 — PIT price gap closure feasibility audit

## Acceptance Rule

No quarantined commit may be pushed, merged, or treated as accepted evidence until manually reviewed by the operator.

If accepted, cherry-pick one commit at a time onto a clean branch with a commit message that includes:

AUTONOMOUS RUN QUARANTINED; CONTENT MANUALLY REVIEWED BEFORE ACCEPTANCE.

## Freeze Boundary

Production model freeze remains ACTIVE.

Forbidden without explicit operator authorization:

- ranker changes
- selector changes
- sizing changes
- final_score changes
- production gate changes
- production snapshot modification
- live price/API fetching
- executable research script creation or execution
- autonomous workstream selection
