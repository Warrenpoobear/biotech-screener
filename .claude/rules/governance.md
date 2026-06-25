---
name: governance
description: Governance policy, tier classifications, promotion path, 13F onboarding
metadata:
  type: governance
  status: active
  paths:
    - governance/**
    - production_data/decision_rulesets/**
    - tools/promote_ruleset.py
---

# Governance Rules

---

## Governance Artifacts (PR #286, merged May 16, 2026)

### AGENT_ROUTING_POLICY.md
Tier 0-4 routing policy classifying every part of the codebase by governance sensitivity. Defines allowed tools, review requirements, and merge rules per tier. The policy itself is Tier 4 — changes require a memo, not a direct edit. Quarterly review required.

**Tier Summary:**
- **Tier 0 (Deterministic Production Hot Path):** Scripts, cron, tests, static checks only. No LLM may supervise, decide, mutate, or deploy production state.
- **Tier 1 (Low-Governance Utility):** Codex CLI, OpenClaw low-risk agents. Documentation, CLI ergonomics, non-scoring utilities.
- **Tier 2 (Medium-Governance Engineering):** Codex first draft + Claude Code review when output feeds Tier 3 code. Non-production analytics, validation scripts, ingestion plumbing.
- **Tier 3 (High-Governance Production/Evidence):** Claude Code for implementation or mandatory review. CCFT, selector, ranker, scoring, catalyst, CRT, shadow, walk-forward harness, production hashes. Tests asserting Tier 3 behavior are themselves Tier 3.
- **Tier 4 (Governance/Research Judgment):** Claude Chat/project chat + human approval. Architecture changes, signal admission/retirement, catalyst taxonomy, ablation interpretation, this policy itself.

**Walk-Forward Harness:** Permanent Tier 3 surface. Evidence-breaking migrations are Tier 4 decisions requiring a memo with cutover date, affected outputs, disposition of pre-migration evidence, and PM sign-off.

**Production Hash Rotation Rule:** Any diff changing a production hash requires a corresponding entry in `governance/HASH_ROTATIONS.md` with old hash, new hash, effective date, affected surface, reason, downstream impact, and reviewer.

**Merge Rule:** Highest affected tier governs review requirements. Patch size is not evidence of safety.

### STATUS.md
Enforcement status: AGENT_ROUTING_POLICY.md is live. Pending enforcement layers: agent_registry.yml (PR 2), AGENT_DIRECTORY_MAP.md, CI registry validation, import-graph validation. Until enforcement layers are live, routing classifications applied manually.

### HASH_ROTATIONS.md
Empty rotation log (policy effective date 2026-05-16). Required fields defined. No rotations recorded.

### Compliance Memo
"Why the DEM 27-Agent Fleet Is Insulated from Model-Output-as-Control-Signal Failures" — Final version, repo-verified. Cites Texas A&M SUCCESS Lab taxonomy of 470 OpenClaw advisories (arXiv:2603.27517). Available in Content Library at ai-projects/.

---

## Promotion Governance

- **Manifest**: `production_data/decision_rulesets/manifest.json` — all rulesets tracked with status (active/candidate/retired)
- **Promotion battery**: `scripts/research/run_promotion_battery.py` -> bucketed verdicts + weekly live-sim -> PASS/FAIL
- **Promote script**: `scripts/promote_ruleset.py` — blocks promotion unless battery PASS
- **Health monitor**: `tools/ruleset_health_monitor.py` — post-promotion drift detection
- **Rollback**: `scripts/promote_ruleset.py --rollback --reason "..."` — first-class with auto-LKG discovery
- **Governance policy**: `governance/AGENT_ROUTING_POLICY.md` — Tier 3/4 review required for all promotion-adjacent changes

---

## Insider Diagnostic (Spec 104)

`insider_net_buy_value_90d` is **DIAGNOSTIC ONLY**. It is tracked in `DIAGNOSTIC_FIELDS` and explicitly excluded from `ALPHA_FEATURE_REGISTRY`. It does NOT enter the scoring model, ranker, or selector.

**CRITICAL**: The expectation model has an `insider_net_buy_z` weight that activates silently if the field flows into `market_features`. Spec 104 R4a requires an explicit isolation guard.

**Blank vs. Zero semantics**: NaN/None/blank = not fetched. 0.0 = fetched, no insider buy activity. Never collapse blank and zero.

**Promotion requires ALL of**: 20+ stable snapshots, >= 60% non-null coverage, IC > 0 at p < 0.05, Checklist v2 pass, explicit written approval.

---

## Expectation Layer Coverage Gate (Spec 105)

Production pipeline hard-fails if market-expectation fields are missing or under-covered in `rankings.csv`. Required fields: `short_interest_pct` (0.90), `close_price` (0.99), `market_cap_mm` (0.95), `priced_move_pct` (0.80), `insider_net_buy_value_90d` (0.30, nonblocking/diagnostic). Thresholds sourced from `FEATURE_COVERAGE_REQUIREMENTS` (single source of truth).

---

## Hermes Knowledge Layer (Spec 089) & Town Bridge (Spec 090)

### Knowledge Layer
Repo-native "ops brain" with four layers: Capture (read-only from specs, artifacts, registry, git, cron), Normalize (structured ledgers), Reason (drift/contradiction/missed-run detection), Deliver (operator briefs). Artifacts in `artifacts/ops/knowledge_layer/`.

### Town-Hermes Bridge
Routes Hermes events to Town via structured email to `djschulz@gmail.com`. Town routine triggers on `[Hermes]` subject prefix. Town is read-only relay — NOT a scheduler, repo mutator, or spec approver. Phase A complete (dry-run mode). Phase B (live delivery) not yet started.

---

## Adding a 13F Manager

- **Use `tools/onboard_manager.py`** — never edit `production_data/manager_registry.json` directly.
- One-shot flow: registry append -> backfill across every existing PIT dir (lookback=40 ~ 10y) -> warm current as-of date -> run `tools/test_manager_integration.py` (6/6 gate).
- Example: `python tools/onboard_manager.py --cik 1802528 --name "Fairmount Funds Management" --aum-b 1.3 --style concentrated_clinical_stage --tier elite_core --notes "..."`
- For reruns or partial flows use `--skip-registry`, `--skip-backfill`, `--skip-current`, `--skip-test`.
- Underlying primitive: `tools/warm_13f_cache.py --ciks <CIK> --existing-pit-dirs --elite-only` (merges into each PIT dir's `index.json`, doesn't disturb other managers).

---

## Long-Call Contract Recommendations (Post-Screen)

When producing long-call candidates from the screen output, also recommend the best executable long-call contract for each surviving candidate.

**Goal:** For every name that passes the long-call filter, recommend:
1. One primary contract
2. One backup contract
3. Or explicitly mark `NO_TRADE` if no contract is liquid / priced well enough

Do NOT just say "buy calls." Pick an actual strike + expiry from the chain data available in the repo/output.

### Step 1 — Expiry selection
- Base case: choose the first liquid expiry that is AFTER the catalyst date and still leaves 14-35 calendar days of cushion after the event
- If catalyst_days is 21-45: allow tighter post-event cushion of 7-21 days
- Avoid expiries that occur BEFORE the catalyst
- Avoid very long expiries unless all nearer expiries are illiquid or the event date is uncertain
- Prefer standard monthly expiries over odd weeklies when liquidity is similar

### Step 2 — Strike selection
- Target call delta between 0.30 and 0.50
- Higher-conviction names: prefer 0.40-0.50 delta
- Lower-conviction / higher-IV names: prefer 0.30-0.40 delta
- Avoid ultra-OTM lottery strikes unless premium is tiny and liquidity is still acceptable
- Avoid deep ITM unless spread/liquidity is clearly superior and thesis is very high conviction

### Step 3 — Liquidity filter
Reject contracts if any of these are true:
- open_interest is too low
- volume is too low
- bid/ask spread is too wide
- pricing looks stale

If the repo does not have exact spread fields, use the best liquidity proxies available and state the limitation.

### Step 4 — Entry economics
For each candidate contract, compute or estimate:
- mid premium
- breakeven move to expiry
- event-date implied move
- crush-adjusted move if available
- delta
- DTE

Prefer contracts where:
- directional thesis is confirmed by RR / skew
- implied move is not already extreme
- the contract still has room to profit after likely post-event IV compression
- premium at risk is reasonable relative to conviction

### Step 5 — Rank contracts
Choose the primary contract by this priority:
1. Expiry appropriately covering the catalyst
2. Strongest liquidity
3. Delta in target band
4. Best breakeven vs thesis
5. Cleaner spread / execution quality

Choose one backup contract that is either:
- one strike lower/higher with similar expiry, or
- next best expiry with similar delta profile

### Output format for each candidate
```
ticker:
  catalyst: <event_type> in <N> days
  thesis: <1-2 lines>
  primary_contract:
    expiry:
    DTE:
    strike:
    option_type: CALL
    delta:
    premium_or_mid:
    open_interest:
    volume:
    spread_or_liquidity_proxy:
    breakeven_move_pct:
    why_this_contract:
  backup_contract:
    <same fields>
  no_trade_reason: <if applicable>
```

### Important constraints
- If exact contract-chain data is unavailable from the snapshot alone, look for the nearest chain artifact/cache already produced by the repo for that date
- If the contract recommendation depends on missing chain fields, say so explicitly and give the best constrained recommendation possible
- Do not change DEM scoring or ranking logic
- This is a post-screen execution recommendation only

---

## Options Expression Layer (Spec 062, 2026-04-13)

- **Status**: Shadow-only, merged to main. Zero alpha impact.
- **Module**: `event_ev/expression_layer.py` — classification -> mapping -> gates -> sizing
- **Attribution**: `event_ev/expression_attribution.py` — JSONL logging, CRT resolution, kill switches
- **Wiring**: `run_screen.py` emits `expression_overlay_summary.json` + `expression_recommendations.json` per snapshot
- **Tests**: 123 (83 expression + 40 attribution)
- **Policy**: overlay-only. Does NOT enter selector/ranker/construction. Expression layer must NEVER be imported by `selector_engine.py`, `ranker_engine.py`, or `decision_engine.py`.
- **Review horizon**: 30 days from first emission. No threshold tuning before then.

---

## Data Explorer Agent (2026-04-13)

- **CLI**: `python -m tools.data_explorer {summary,compare,qa,catalog,field,top-n,daily}`
- **Package**: `tools/data_explorer/` (loader, catalog, explorer, comparator, reporter, viz)
- **Tests**: 33
- **Policy**: Read-only analysis. Canonical reporting source — console agent summaries are non-authoritative unless backed by dataset evidence.
- **Output**: `reports/data_explorer/` (timestamped directories with markdown + PNG charts)

---

## Architecture Freeze & Lift (2026-06-20+)

**Scoped production model freeze** (INC-2026-06-20-AUTOPUSH): ranker, selector, sizing, `final_score`, portfolio/snapshot files frozen.

| Safe during freeze | Blocked without lift + Spec |
| --- | --- |
| Expectation verification, Spec 105 QA | Ranker/selector weight changes |
| Event EV shadow, Sci-Cart diagnostics | `final_score` production changes |
| Observability, fleet ops, Hermes read-only | Portfolio construction changes |

**Lift workflow:** `docs/governance/FREEZE_LIFT_FORWARD_EVIDENCE_PACKAGE_2026_06_25.md`

```bash
FREEZE_LIFT_ACK=1 bash tools/run_forward_evidence_package.sh --write
```

Operator sign-off required. Evidence package does not auto-lift freeze.
