# AGENTS.md — Calibration Evidence Accumulator

## Session startup

1. Read `SOUL.md` — identity and boundaries
2. Read `TOOLS.md` — data sources and output paths
3. Check if new postmortem records exist since last run

## Daily sequence

1. Scan `artifacts/postmortem/` for resolved events with post-event returns
2. Cross-reference with pre-event snapshot (rank, tier, sort contributions)
3. Build signal contribution tracker — which model components earned weight
4. Build threshold audit — gates that excluded winners or included losers
5. Build calibration curve — hit rates by rank decile
6. Write evidence JSON + MD + append to ledger

## Memory protocol

Write session summaries to `agents/calibration_evidence/memory/`.
Track: events processed, evidence counts, notable findings.

## Self-learning (Rule 12)

Calibration finding → LRN with `Promotion-lane: spec`.

## Red lines

- Do not recommend weight changes or signal promotion
- Do not modify rulesets, manifest, scoring code, or production data
- Do not make causal claims about individual outcomes
