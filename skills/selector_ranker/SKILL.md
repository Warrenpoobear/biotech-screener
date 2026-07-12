---
name: selector-ranker
---

# Selector / Ranker / Construction Skill

## Purpose

Reference for the production two-stage selector/ranker architecture and EW Top-30 construction. This is how the screener turns scored tickers into an actionable ranked portfolio.

This skill is organized into two sections:

1. **Framework Reference** \- Stable architecture, engines, dead lanes, and governance \(changes only with code updates\)
2. **Operational State** \- Volatile research status and metrics that require periodic refresh

**Codebase context:** the three production engines total ~4,000 Python lines — a **compact, high-leverage** surface. `run_screen.py` (~13k lines) orchestrates them into daily output. Repo scale does not reduce governance tier; it increases the cost of small-looking edits.

---

## Codegraph Preflight (mandatory before any code edit)

Every code change to selector/ranker surfaces is Tier 3 (see `governance/AGENT_ROUTING_POLICY.md`). Before editing any symbol in this area, run the standard codegraph preflight per `skills/codegraph/SKILL.md`:

1. `codegraph_search("<symbol>")` — locate the target symbol
2. `codegraph_node("<symbol>", source=True)` — inspect its signature and body
3. `codegraph_callers("<symbol>")` — identify all production callers
4. `codegraph_callees("<symbol>")` — map downstream dependencies
5. `codegraph_impact("<symbol>", depth=2)` — confirm blast radius

**Gate:** If `codegraph_impact` reaches `selector_engine.py`, `ranker_engine.py`, `decision_engine.py`, `final_score`, or `rankings.csv` — the change is **BLOCKED** until operator approval.

---

# SECTION 1: FRAMEWORK REFERENCE

---

## Production Stack \(v1.14.0\)

```
Modules 1-5 (scoring)
  -> Decision Engine (L0 gates -> L2 overlays -> L4 tiers -> L3 sizing -> sort key)
  -> Selector Engine (B6: coinvest_score_z 100%)
  -> Ranker Engine (pairwise_minimal: 2 deployed features, top-60 cohort, ordinal-only)
  -> Sort by final_score -> EW Top-30 -> rankings.csv
```

---

## Selector Engine

**File**: `selector_engine.py`

### B6 Selector \(Production\)

- **v1.14.0**: coinvest\_score\_z at 100% weight \(coinvest-only\)
- **Prior \(v1.13.0\)**: coinvest 65% + inst\_delta\_z 35%
- inst\_delta\_z zeroed 2026-05-04 \(ALERT: mean IC = -0.097 over 36 dates, two-frame confirmed\)
- Reinstatement conditions documented in governance log

### Selector Validation

- Checklist v2 \(2026-04-04\): bootstrap +2.42pp/mo, 95% CI \[1.25%, 3.70%\], P\(>0\) = 99.99%
- LOSO: ROBUST across all dimensions
- Neither component survives standalone, but the bundle is real
- Sort anchor: `selector_score`

### What the Selector Learns

Coinvest selects WHICH 30 names enter the portfolio. It captures institutional co-investment conviction from elite biotech managers.

---

## Ranker Engine

**File**: `ranker_v2_pairwise.py`

### Pairwise Minimal Ranker \(Production\)

- 2 deployed features, ordinal-only \(no rank-weighting, no confidence sizing\)
- Live artifact: `production_data/ranker_v2_model.json`
- Deployed weights: `coinvest_score_z` +0.02 \(capped Family C live pilot\), `financial_score` -0.0533
- ECE = 0.129 \(POOR calibration - confirms ordinal-only is correct\)
- Top-60 cohort scope
- inst\_delta\_z excluded from production ranker and zeroed in selector since v1.14.0

### Within-Top-30 Feature Roles

| Feature | Role | Evidence / Weight |
| --- | --- | --- |
| coinvest\_score\_z | Bounded institutional support within top-60 | +0.02 deployed \(+0.0613 trained\) |
| financial\_score | True negative penalty \(stress-upside\) | -0.0533 deployed; NW t-stat -3.41 |
| inst\_delta\_z | Diagnostic/export only | 0% selector weight; absent from ranker |

### financial\_score Sign Direction \(RESOLVED, Spec 093\)

- Weight: -0.0533 in `production_data/ranker_v2_model.json`
- **Confirmed intentional**: stress-upside thesis \(Spec 074, reconfirmed Spec 093 2026-05-13\)
- Classification: INTENTIONAL\_STRESS\_UPSIDE
- Negative weight means financially safe names are penalized \(more catalytic, less safe names preferred\)
- Evidence: correct higher\_is\_better=True encoding; six-diagnostic audit confirmed TRUE PENALTY in both bull \(NW-t=-3.42\) and bear \(-3.38\) regimes
- Raw components: 50% runway + 30% dilution + 20% liquidity \(all directional: higher = better health\)
- Rank-normalized within stage x size cohort \(direction preserved\)
- t-statistic significant \(-3.41\), persists across cohorts and regimes

---

## Construction

### EW Top-30

- Equal-weight, top 30 names by final\_score
- K=30 validated by PIT sweep \(stable K=25-35 plateau, net-of-cost peak\)
- RW-EW delta = -0.09pp, t = -0.95 \(rank-weighting does NOT help\)

### Production Evidence

- True PIT backtest (split-adjusted v2, 2026-06-27): **+3.33pp/mo hedged** (+3.20pp net-of-cost), t_nw=2.74, 66.7% hit rate, 69 monthly periods (Jun 2020 – Feb 2026)
- Regime breakdown (all three regimes positive — prior "bull adverse" claim RETRACTED, was split artifact):

| Regime | n | Mean (pp/mo) | t (NW) | Hit% |
|---|---|---|---|---|
| Bull | 30 | **+3.05** | 2.06 | 66.7% |
| Bear | 26 | **+1.94** | 1.95 | 61.5% |
| Neutral | 13 | **+6.76** | 2.69 | 76.9% |

- **Prior claim retracted**: "Bear/neutral engine, Bull -0.37pp (50% hit)" was from raw (unadjusted) price file. Split-adjusted data shows bull regime +3.05pp.
- YTD (Dec 31 → Jun 26, monthly EW rebalance): **DEM +38.52% vs XBI +27.43%, +11.09pp excess**

---

## Decision Engine

**File**: `decision_engine.py`

### Pipeline Layers

| Layer | Purpose |
| --- | --- |
| L0 | Hard gates \(liquidity, price, data quality\) |
| L2 | Overlays \(event\_type\_score as diagnostic\) |
| L4 | Tier classification |
| L3 | Position sizing |

### EV/Sizing Severity Consumption \(Spec 101, RESOLVED\)

The L3 position sizing layer consumes `ev_severity_score` \(from runway severity v1.1\) to compute:

```
dilution_haircut = 0.35 * ev_severity_score
size_multiplier = max(0.40, 1.0 - 0.60 * ev_severity_score)
```

`ev_severity_score` is now exported to `rankings.csv` and `SNAPSHOT_COLUMNS` \(Spec 101, commits eaa4ea87 + cba4ee0f\). `check_severity_formulas()` QA validation runs every snapshot.

---

## Dead Lanes \(Do Not Reopen Without New Evidence\)

| Lane | Status | Why |
| --- | --- | --- |
| Options surface-shape as ranker | DEAD | 50-month IC negative all horizons |
| Options-as-alpha \(Spec 053\) | CLOSED | 37 signals tested, ALL fail |
| Static execution features \(Spec 054\) | CLOSED | All noise/destructive |
| Clinical composites as ranker \(Spec 055\) | CLOSED | Negative across ALL slices |
| total\_volume\_z | DEAD | IC = -0.10 on PIT data |
| Always-on rank-weighting | NOT PROMOTED | RW-EW = -0.09pp |
| insider\_exec\_buy\_value\_90d | SHADOW ONLY | 1/5 Checklist v2 |
| aact\_execution\_score | SHADOW ONLY | 1/5 Checklist v2 |
| cal\_alpha | REMOVED v1.12.0 | Confirmed no-op |
| Clinical sort signal | OFF | Insufficient IC |
| Fixed sleeve budgets | RETIRED | Primary construction damage \(+153.6pp drag\) |

---

## Promotion Governance

| Component | File |
| --- | --- |
| Manifest | `production_data/decision_rulesets/manifest.json` |
| Promotion Battery | `scripts/research/run_promotion_battery.py` |
| Promote Script | `scripts/promote_ruleset.py` \(blocks unless battery PASS\) |
| Health Monitor | `tools/ruleset_health_monitor.py` \(post-promotion drift\) |
| Rollback | `scripts/promote_ruleset.py --rollback --reason "..."` |

### Drift Detection

- History: JSONL append per evaluation \(idempotent on same-day reruns\)
- Consecutive WARN tracking by active ruleset ID
- Recommend rollback after sustained degradation

---

## Source Files

| Component | File |
| --- | --- |
| Decision Engine | `decision_engine.py` |
| Selector Engine | `selector_engine.py` |
| Ranker v2 Pairwise | `ranker_v2_pairwise.py` |
| Ranker Legacy | `ranker_engine.py` |
| Main Orchestrator | `run_screen.py` |
| Ruleset Manifest | `production_data/decision_rulesets/manifest.json` |
| Promotion Battery | `scripts/research/run_promotion_battery.py` |
| Checklist v2 | `scripts/research/checklist_v2_rerun.py` |

---

# SECTION 2: OPERATIONAL STATE

> **SNAPSHOT DATA** \- The values below are point-in-time and go stale. Verify against current pipeline output before citing.

---

## Active Ruleset

*Last reviewed: 2026-05-13*

- **ID**: `8887576e` \(v1.14.0\)
- **File**: `production_data/decision_rulesets/v1.14.0_coinvest_only_selector.json`
- **Architecture freeze**: In effect until post-h20d checkpoint \(\~2026-05-26\). No new enforcement logic or scoring changes until then.
- **ranker\_active\_contract.py**: Exists on unmerged branch \(`hygiene/ranker-active-contract-2026-04-30`\), deferred to post-freeze. Manual spot-checks via snapshot\_integrity verifier in the interim.

## Ranker Alternatives Research \(T1-T8, updated Specs 093-100\)

*Last reviewed: 2026-05-17 (governance freeze update). Spec implementation status updated.*

### Key Findings \(2026-05-13 audit cycle\)

**Spec 093 \(financial\_score sign direction\)**: RESOLVED as INTENTIONAL\_STRESS\_UPSIDE. Not an artifact. Closes the critical blocker for all ablation baseline interpretation.

**Spec 094 \(selector-only comparator\)**: Classification RANKER\_UNPROVEN. Jaccard overlap between selector-only and production top-30 is 42.7% \(significant churn\). Ranker-added names show lower coinvest\_z \(0.55 vs 0.95\) and lower financial\_score \(24.5 vs 47.4\) - stress bias confirmed. Forward-return coverage too sparse for significance \(6/69 postmortems, 8.7%\). Rerun target: 2026-05-27.

**Spec 095 \(evaluation scope\)**: CURRENT\_TOOLS\_CONFLATED. IC backtest measures composite\_score, NOT production final\_score. Ranker IC is UNMEASURED. See ic-evaluation skill for full details.

**Spec 100 \(ranker IC tooling correction\)**: ✓ IMPLEMENTED (commit 2faa88e6, 2026-05-17). IC backtest now measures `final_score` (production ranker) by default. Prior composite_score IC claims invalidated; final_score IC baseline established. Interpretation deferred until post-freeze Checklist v2 evaluation.

> **Numbering note \(2026-05-14\):** "Spec 100" remains the ranker IC tooling correction. The expectation layer coverage verification spec was renumbered to Spec 105 \(commit cb242311\) to resolve the collision. No ambiguity remains.

### 10 Candidate Alternatives

| Alternative | Status | Notes |
| --- | --- | --- |
| Alt 1 | Coinvest double-count | rho = +0.882 with final\_score |
| Alt 3 | HIGH\_POTENTIAL\_BUT\_BLOCKED | Spec 071 Lane 2, \~Q3 2026 |
| Alt 4 | HIGH\_POTENTIAL\_BUT\_BLOCKED | Spec 071 Lane 2, \~Q3 2026 |
| Alt 6 | HIGH\_POTENTIAL\_BUT\_BLOCKED | Spec 077 prospective accumulation |
| Alt 7 | NO\_GO | EES v3 closed |
| Alt 8 | NO\_GO | Clinical closed lane |
| Alt 9 | NO\_GO | Underpowered |
| Alt 10 | OBSERVE | No-ranker comparator, INCONCLUSIVE at n=11 snapshots |

**Alt 10 detail** \(2026-05-13\): Selector wins 4/6 in clean window \(diff = +0.020\), 0/5 in regime window \(diff = -0.025\). Pooled: selector wins 4/11 \(ranker-override slightly better, dominated by regime confounding\). Powered verdict requires Gate 4 + Gate 7 \(\~2026-07-15\).

**Promotion eligibility horizon**: April 2027 \(one-year stability gate\)

### Monitoring Specs \(2026-05-13\)

| Spec | Purpose | Gate | Next Review |
| --- | --- | --- | --- |
| 096 | Gate/ranker separation doctrine | Defines promotion paths | Ongoing |
| 097 | Event-EV prospective monitoring | Brier <= 0.08, n >= 30 | Monthly |
| 098 | Catalyst timing prospective monitor | Correlation > 0.15 | Monthly |
| 099 | Clinical orthogonality audit | Pre-promotion gate | Before any clinical signal promotion |

---

## Governance Freeze & Quarantine Status (2026-07-10)

- Production ruleset frozen at `8887576e` (v1.14.0). Model status: **FROZEN (BLOCKED_LEVEL_0)** — canonical authority `docs/model_documentation.md`. The binding window is the **DEM candidate freeze + NO_MODEL_CHANGE forward-validation window** (`docs/FORWARD_VALIDATION_PROTOCOL.md` §1; mandate SM-20260629-001, 0/20 eligible live windows). No selector/ranker/sizing change is authorized; any such change resets the out-of-sample clock and requires explicit operator authorization.
- **Candidate identity:** `model_hash=827c35a9ed3ee6e1` (`ast-v1` behavioral fingerprint over `ranker_engine.py`/`selector_engine.py`/`decision_engine.py`); legacy raw-bytes `a9983a67c6954813`. Registered 2026-06-26 (unchanged); re-hashed 2026-07-10.
- **Deployed ≠ trained:** `production_data/ranker_v2_model.json` is a 2-feature `pairwise_logistic` with `coinvest_score_z` capped to +0.02 (trained +0.0613) and `financial_score` −0.0533; see the artifact's `provenance` block. Any prose claiming a 5- or 6-feature production ranker is wrong.
- **13F Q1 2026: HOLD / observation-only.** 55-manager Jaccard last measured 0.463 (FAIL vs 0.70); the 0.875 figure was Q4/49-manager. Authority: `artifacts/readiness/H20D_REEVAL_VERDICT_2026_07_04.md`; clearance requires `tools/check_13f_cohort_quarantine.py` against a post-Q1-promotion snapshot.

## Session-end learning

If anything surprised you, log a learning per the `self-improving` skill (Pattern-Key `SKILL_SELECTOR_RANKER_{description}`); recurrence ≥ 3 in 7 days → propose a patch via `tools/pattern_to_skillpatch.py`.
