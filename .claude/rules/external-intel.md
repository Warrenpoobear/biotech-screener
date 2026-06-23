---
name: external-intel
description: External competitive landscape, industry context, developer profile
metadata:
  type: reference
  status: active
---

# External Intelligence & Context

---

## OpenClaw Status: FENCED — LEGACY_READ_ONLY_DORMANT (2026-06-22)

- **Fenced 2026-06-22 (PR #372):** `**` wildcards and shell write permissions removed from `~/.openclaw/exec-approvals.json`. Exec allowlist is now read-only: `python3, ls, cat, head, tail, grep, find, wc, date, stat, diff, sort, jq`. No bash, no git, no `/mnt/c/**` write access.
- **Hard retire:** deferred — natural fence on next reboot (no `@reboot`/systemd entry). Gateway is loopback-only; no active OpenClaw cron.
- **Hermes is now primary orchestrator.** OpenClaw = legacy dormant. Lane A (deterministic scripts) and Lane B (`run_agent_direct.py`) are unaffected.
- Hermes Agent v0.14.0+ includes native `hermes claw migrate` tool. Any formal migration remains a Tier 4 governance decision.
- See `docs/governance/OPENCLAW_FENCE_DECISION_2026_06_22.md` for audit and decision record.

---

## OpenClaw Security Posture (unchanged)

- Texas A&M SUCCESS Lab (arXiv:2603.27517, April 2026): 470 advisories organized by 7 architectural layers and 5 attack types
- Three Moderate/High-severity advisories compose into complete unauthenticated RCE from LLM tool call to host process
- Exec allowlist bypass via line continuation, busybox multiplexing, GNU long-option abbreviation
- Malicious skill executed two-stage dropper within LLM context, bypassing exec pipeline entirely
- DEM insulation: no agent has authority to modify production weights without traversing the full multi-gate promotion path

---

## Hermes Agent (Competitive Frame)

- Nous Research, launched February 2026, MIT license, $70M funded ($1B valuation)
- 153K GitHub stars (May 16, 2026 — single dated data point from GitHub)
- v0.14.0 shipped May 16, 2026: xAI Grok integration (1M context via SuperGrok OAuth), OpenAI-compatible local proxy for OAuth providers, Claude operator worker launcher (`hermes claude-operator` for spawning Claude Code tmux workers), 12 P0 + 50 P1 bug closures
- Core differentiator: self-learning Skill Documents (Markdown files created after 5+ tool calls, 40% efficiency gains)
- Self-evolving skill loop is governance-incompatible with CCFT unless every skill mutation is versioned, reviewed, and approved
- Native `hermes claw migrate` tool confirms Nous is actively targeting OpenClaw's installed base
- Recommendation: Monitor as potential successor runtime (Q4 2026 evaluation if OpenClaw patch cadence degrades). Do not adopt now. Any migration must preserve AGENT_ROUTING_POLICY.md tier structure and CCFT controls.

---

## ODIN Engine (External Benchmark for Clinical Scoring)

- L2-regularized logistic regression, 51 engineered features, 8 signal categories
- AUC: 0.9363 on 2,210 historical FDA events (2000-2025); verified 96.2% accuracy on 53 outcomes
- ODIN feature categories not in DEM: manufacturing/CMC risk, FDA era effects, options market implied probability, sponsor historical approval rate by therapeutic area
- These are Tier 4 evaluation candidates through T5 promotion path

---

## BiotechEdge (External Benchmark for Institutional Signals)

- Tracks 20 specialist biotech hedge funds, $46.5B+ total assets, 1,558+ companies, 2,318+ catalysts
- Fund convergence signal (3+ independent funds buying same stock) validates DEM's coinvest_score_z methodology
- Open-source alternative: pr124/Biotech_Fund_Tracker (GitHub) parsing 38-40 specialist funds

---

## FDA Real-Time Clinical Trial Initiative (April 2026)

- Two RTCT proof-of-concept studies launched: AstraZeneca TRAVERSE (MCL), Amgen STREAM-SCLC
- AI-Enabled Early-Phase Trial Pilot Program RFI (comments due May 29, 2026; selections August 2026)
- Projected 20-40% trial duration reduction, $120M annual savings
- If trials become continuous rather than phase-gated, binary catalyst model evolves — affects catalyst_decay_w and catalyst_quality calibration. Monitor as Tier 4 governance question.

---

## AI Drug Pipeline (Q1 2026)

- 173+ AI-originated programs in clinical trials (94 Phase I, 56 Phase II, 15 Phase III) — 7x increase since 2022
- Pre-clinical compression: 4-6 years to 12-24 months. Clinical timelines unchanged.
- Insilico Medicine Rentosertib: first fully AI-designed drug with Phase IIa results (Nature Medicine, June 2025)
- Isomorphic Labs: $2.1B Series B (May 2026). Recursion: fifth Sanofi milestone ($134M cumulative)
- 2026 is definitive validation year — Phase III results determine if AI improves beyond ~90% historical failure rate

---

## Industry AI Adoption

- 92% of hedge funds with $1B+ AUM use AI/ML (up from 56% in 2022); 67% describe as "integral"
- AI-integrated funds outperform traditional systematic strategies by 3-4pp annually
- NBIM ($2.1T): ~50% of 680 staff code own AI tools using Claude; all employees use AI daily
- Only 21% of organizations deploying AI have formal governance frameworks — DEM's merged governance artifacts place it in the leading minority

---

## Developer Profile

This system is maintained by an institutional SFO investment professional (CFA, CAIA, 30+ years) who is Director of Investments at Wake Robin (wakerobin.co), a real estate investment and community development company in Holland, MI.

### Quantitative Biotech Investment
- **Biotech equity screening pipeline** — multi-module scoring (financial health, clinical development, catalyst/event resolution, composite), Decimal arithmetic, PIT-safe, deterministic, 13-step daily production pipeline
- **Institutional signal analysis (13F/13D/13G)** — coinvest_score_z pipeline tracking Fairmount Funds, Deep Track Capital, Logos Global Management via SEC EDGAR. PIT cache infrastructure, cohort quarantine, contamination window governance
- **Statistical signal evaluation** — Spearman IC decomposition, Checklist v2 promotion battery (FM, bootstrap, FDR, LOSO, year stability), forward shadow monitoring, evidence hierarchy, dead-lane registry
- **Catalyst & event-driven analysis** — 7+ event sources (ClinicalTrials.gov/AACT, SEC 8-K, FDA ADCOM, PDUFA, EMA), catalyst_decay_w, binary_quality_score, event_ev_p_hit, resolution tracking
- **Decision engine & portfolio construction** — two-stage selector/ranker, B6 coinvest-only selector, pairwise minimal ranker, EW Top-30. Production: +2.34pp/mo net-of-cost, t=2.57
- **Biotech earnings signal classification** — SM/ACC/MR/CON/ID weekly post-mortem system with cumulative ledger

### Family Office & Institutional Modeling
- **SFO liquidity architecture** — 7-layer deterministic modeling stack (entity, account/position, cash-flow, PE pacing, RE+OpCo, liquidity, allocation/policy). Four-line principle. Quarterly ledger spine.
- **PE pacing models** — Takahashi-Alexander, STAIRS market-coupled adapter, capital call obligation bridge, configurable reconciliation gates
- **Spending policy design** — flat-real, smoothing, Owl/Guyton-Klinger guardrails, configurable spending base denominator
- **Institutional asset allocation** — multi-asset-class rebalancing, AUM management ($14B+), sovereign wealth fund and public pension experience
- **Alternatives & derivatives** — options strategies, index futures, structured products, tail risk hedging

### Technical AI/Automation
- **AI agent fleet architecture** — 27-agent Hermes/OpenClaw fleet (per AGENT_REGISTRY.json schema v1.0, as-of 2026-04-28: 27 active, 1 suppressed, 1 retired, 1 shadow) on Llama 3.3 70B via Together AI. Per-agent SOUL.md, four-layer monitoring, Knowledge Layer (Spec 089). Governed by governance/AGENT_ROUTING_POLICY.md (Tier 0-4, merged PR #286 May 16, 2026). Authority levels: observe_only, observe_and_propose, write_artifacts, mutate_data, mutate_config. Only crt_resolution_watcher holds mutate_data. Three-lane operational routing (Lane A deterministic, Lane B cheap monitoring, Lane C manual engineering). No cron job may depend on a gateway token.
- **Production pipeline engineering** — 13-step daily biotech screener (cron 5:30 PM ET), timeout optimization, race condition resolution, sleep-cliff mitigation, determinism enforcement (byte-identical outputs)
- **Town AI platform** — 18 active routines, 19 custom skills encoding pipeline scoring rules, SFO architecture, governance. Routine design with email recipients, MCP servers, callable sub-routines
- **LLM integration** — Claude/Grok/ChatGPT for research synthesis, prompt engineering, Llama-optimized inference tuning, persona configuration
- **DevOps** — WSL2 Python, cron orchestration, Together AI gateway monitoring, token management, log aggregation

### Financial Data Engineering
- **SEC EDGAR pipeline** — PIT-safe ingestion of 13F-HR, 13D/A, 13G, Form 4, 8-K. Per-CIK PIT cache, canonical institutional summary, CUSIP-first reasoning, staleness gates, SEC_USER_AGENT compliance
- **Clinical trial data** — ClinicalTrials.gov/AACT, EU/EEA registries (EUCTR, CTIS, ISRCTN), cross-registry dedup, trial status monitoring
- **Financial data APIs** — yfinance, Alpaca, Polygon, Massive feeds, PubMed, AACT
- **Data validation** — CCFT principles (canonical, complete, frozen, timestamped), PIT audits, survivorship bias detection, snapshot collapse guards

### Research Governance & Spec Lifecycle
- **Evidence standards** — Checklist v2 (FM, bootstrap, FDR, LOSO, calibration), true PIT backtests, forward shadow monitoring
- **Spec lifecycle** — DRAFT > IN PROGRESS > HELD > RESOLVED > CLOSED, with phased acceptance criteria, blocking dependencies, closure memos. Currently managing specs 071-105.
- **Promotion governance** — promotion battery, ruleset health monitor, architecture freeze protocol, rollback capability. Governed by governance/AGENT_ROUTING_POLICY.md Tier 3/4 requirements.
- **Knowledge management** — Hermes Knowledge Layer (Spec 089), Town-Hermes Bridge (Spec 090), held-spec ledger, first-fire ledger, contradiction ledger, operator briefs
