# External AI & Competitive Landscape

*Reference only — not path-scoped. Load on demand.*
*Last updated: 2026-05-20*

## OpenClaw Status: Maintenance-Only (confirmed May 2026)
- Transitioned to maintenance-only. No new features expected.
- DEM's 27-agent fleet runs on OpenClaw. Maintenance-only does NOT affect current operations.
- Hermes Agent v0.14.0 includes `hermes claw migrate` for seamless migration.
- **Action: none.** Fleet runs on deterministic scripts (Lane A) and `run_agent_direct.py` (Lane B). Migration is Tier 4 governance decision.
- **Planning horizon**: evaluate Hermes as successor in Q4 2026 if OpenClaw patch cadence degrades.

## OpenClaw Security (Texas A&M arXiv:2603.27517)
- 470 advisories, 7 architectural layers, 5 attack types
- Three Moderate/High compose into unauthenticated RCE from LLM tool call to host
- DEM insulation: no agent can modify production weights without full multi-gate promotion path

## Hermes Agent (Competitive Frame)
- Nous Research, MIT license, $70M funded ($1B valuation), 153K GitHub stars
- v0.14.0: xAI Grok integration (1M context), OpenAI-compatible local proxy, Claude operator worker launcher
- Self-evolving skill loop is governance-incompatible with CCFT unless versioned and reviewed
- Targeting OpenClaw's installed base via `hermes claw migrate`

## ODIN Engine (Clinical Scoring Benchmark)
- L2-regularized logistic regression, 51 features, 8 signal categories
- AUC 0.9363 on 2,210 historical FDA events; 96.2% on 53 recent outcomes
- Features NOT in DEM: manufacturing/CMC risk, FDA era effects, options implied probability, sponsor historical approval rate
- Tier 4 evaluation candidates through T5 promotion path

## BiotechEdge (Institutional Signal Benchmark)
- Tracks 20 specialist biotech hedge funds, $46.5B+ total, 1,558+ companies, 2,318+ catalysts
- Fund convergence signal (3+ funds buying same stock) validates coinvest_score_z methodology
- Open-source alternative: pr124/Biotech_Fund_Tracker (38-40 specialist funds)

## FDA Real-Time Clinical Trial Initiative (April 2026)
- Two RTCT proof-of-concept studies: AstraZeneca TRAVERSE (MCL), Amgen STREAM-SCLC
- AI-Enabled Early-Phase Trial Pilot RFI (selections August 2026)
- If trials become continuous rather than phase-gated, binary catalyst model evolves — affects catalyst_decay_w and catalyst_quality. Monitor as Tier 4.

## AI Drug Pipeline (Q1 2026)
- 173+ AI-originated clinical programs (94 Phase I, 56 Phase II, 15 Phase III) — 7x since 2022
- Pre-clinical compression: 4-6 years to 12-24 months. Clinical timelines unchanged.
- 2026 is definitive validation year — Phase III results determine if AI beats ~90% historical failure rate

## Industry AI Adoption
- 92% of $1B+ AUM hedge funds use AI/ML (up from 56% in 2022)
- AI-integrated funds outperform traditional systematic by 3-4pp annually
- NBIM ($2.1T): ~50% of 680 staff code own AI tools using Claude
- Only 21% of orgs deploying AI have formal governance — DEM's governance artifacts place it in the leading minority
