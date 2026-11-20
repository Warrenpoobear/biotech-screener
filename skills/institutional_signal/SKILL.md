---
name: institutional-signal
---

# Institutional Signal Skill

## Purpose

Reference for the 13F institutional signal pipeline - from SEC EDGAR filing ingestion through coinvest\_score\_z production signal and cohort quarantine governance. This is the dominant selector signal \(100% weight in v1.14.0\).

This skill is organized into two sections:

1. **Framework Reference** \- Stable architecture, rules, and processes \(changes only with code updates\)
2. **Operational State** \- Volatile status snapshots that require periodic refresh

---

## Codegraph Preflight (mandatory before any code edit)

The institutional signal (13F / coinvest_score_z) is the sole selector signal at v1.14.0 — Tier 3. Before editing any symbol, run the standard preflight per `skills/codegraph/SKILL.md`:

1. `codegraph_search("<symbol>")` — locate the target
2. `codegraph_node("<symbol>", source=True)` — inspect signature and body
3. `codegraph_callers("<symbol>")` — identify all production callers
4. `codegraph_callees("<symbol>")` — map downstream dependencies
5. `codegraph_impact("<symbol>", depth=2)` — confirm blast radius

**Gate:** If impact reaches `selector_engine`, `ranker_engine`, `decision_engine`, `final_score`, `coinvest_score_z`, or `rankings.csv` — change is **BLOCKED** until operator approval.

---

# SECTION 1: FRAMEWORK REFERENCE

---

## Architecture Overview

```
SEC EDGAR 13F-HR filings
  -> warm_13f_cache.py (per-CIK PIT cache dirs)
  -> build_institutional_summary.py (canonical summary)
  -> coinvest_score_z (selector signal, 100% weight v1.14.0)
  -> inst_delta_z (diagnostic/export only in v1.14.0)
```

## Manager Registry

**File**: `production_data/manager_registry.json`
**Never edit directly** \- use `tools/onboard_manager.py`

### Tiers

| Tier | Description | Signal Weight |
| --- | --- | --- |
| elite\_core | Highest-conviction biotech-focused managers | Full weight |
| conditional | Broader institutional managers with biotech exposure | Reduced weight |
| retired | Former managers kept for reproducibility \(added v4.0, 2026-11-01\) | Excluded from coinvest |

Current roster note, 2026-11-01: registry v4.0 has 30 `elite_core` managers, specified by the operator \(was 49\). It also has 6 `conditional` and 26 `retired` managers. Firm CIKs that belong together \(Foresite IV/V/VI, Broadfin old and new CIK\) share a `firm_group` and count as one holder. 26 of the 30 elite managers were chosen on trailing 3-year return, so backtests over that window are selection-biased and must not be cited. Memo: `docs/governance/2026-11-01-coinvest-13f-integrity-clock-reset.md`.

### Onboarding Flow

```bash
python tools/onboard_manager.py \
  --cik 1802528 \
  --name "Fairmount Funds Management" \
  --aum-b 1.3 \
  --style concentrated_clinical_stage \
  --tier elite_core \
  --notes "..."
```

One-shot flow: registry append -> backfill across every existing PIT dir \(lookback=40, approx 10y\) -> warm current as-of date -> run `tools/test_manager_integration.py` \(6/6 gate\).

Partial reruns: `--skip-registry`, `--skip-backfill`, `--skip-current`, `--skip-test`.

---

## coinvest\_score\_z

The production selector signal. Measures institutional co-investment conviction across elite biotech managers.

### Key Properties

- Drives 100% of selector weight \(v1.14.0, coinvest-only after inst\_delta\_z demotion\)
- Correlation with final\_score: rho = +0.882 \(double-count concern, documented in T1 ranker anatomy\)
- Checklist v2: 3/5 standalone; prior bundle with inst\_delta is historical evidence only
- Collapse guard: SD floor = 0.10 \(below this, snapshot integrity check FAILs\)

### Data Flow

1. PIT cache: `data/caches/sec_13f/PIT/{YYYY-MM-DD}/` per manager
2. Canonical summary: `production_data/institutional_summary.json`
3. Delta computation: `institutional_summary_delta.json` \(pre vs post refresh\)
4. Score: z-scored across eligible universe per snapshot

Production input note, 2026-11-01: production `coinvest_score_z` comes from the pre-built `production_data/coinvest_signals.json`. That file is produced from the PIT cache by `scripts/build_coinvest_features_from_13f.py` 1.1.0. The builder applies three gates:

- a 230-day `period_of_report` staleness gate;
- `firm_group` holder grouping;
- an active-roster gate.

`run_screen.py` recomputes the filing age from `latest_filed_at` on every run. Current vintage: Q2 2026 \(as\_of 2026-10-30\). The rebuild after each 13F cycle is a **manual, governed step**: check the file's `period_of_report` before treating coinvest as current.

---

## inst\_delta\_z

Quarter-over-quarter change in institutional holdings. Measures whether smart money is accumulating or distributing.

### Governance Rules

- Reinstatement requires IC recovery evidence documented in governance log
- Current production status: selector weight 0%, absent from the production 2-feature ranker
- Historical audit showed positive within-cohort discrimination \(NW-t = +3.32\), but this is not active production behavior
- When zeroed, selector runs on coinvest\_score\_z alone and ranker uses only `coinvest_score_z` + `financial_score`
- **Blank vs. zero, 2026-11-12 (PR #635/#636):** a missing or unusable delta stays **blank** with `has_inst_delta="False"`. It is never a neutral 0.0. Causes:
  - an unknown 13F filing period;
  - a same-source prior;
  - a manager-registry transition, handled by `run_screen.py::_apply_inst_registry_transition_guard`. Raw `inst_delta_*` counts are kept.

  The IC dashboard skips blank dates. Do not impute 0 or count blanks as IC observations. Memo: `docs/governance/2026-11-04-missing-fields-clock-reset.md`.

---

## insider\_net\_buy\_value\_90d \(Spec 104, Diagnostic Only\)

Form 4-derived insider buying signal. Tracks net insider purchase value over a trailing 90-day window.

### Status: DIAGNOSTIC ONLY

- Listed in `DIAGNOSTIC_FIELDS`, NOT in `ALPHA_FEATURE_REGISTRY`
- Tracked and exported for observability
- Does NOT enter the scoring model, ranker, or selector
- Does NOT affect ranks, actions, or position sizing

### Blank vs. Zero Semantics \(CRITICAL\)

| Value | Meaning |
| --- | --- |
| NaN / None / blank | Not fetched, no Form 4 coverage for this ticker |
| 0.0 | Fetched successfully, no insider buy activity in 90-day window |

Never collapse blank and zero. Never impute zero for missing or blank for zero.

### Expectation Model Isolation Guard \(Spec 104, R4a\)

The expectation model has an `insider_net_buy_z` weight that activates silently if `insider_net_buy_value_90d` flows into `market_features`. This is the fragile path that "diagnostic-only" depends on. Spec 104 requires an explicit guard: either runtime assertion that the field is NOT in `market_features`, or weight zeroing, or a pre-inference drop guard.

### Promotion Criteria \(future, not current build\)

Requires ALL of: 20+ stable snapshots with >= 60% non-null coverage, blank/zero integrity verified, IC > 0 at p < 0.05, Checklist v2 battery pass, explicit written approval. Until all five are met, insider stays diagnostic.

### Relationship to Crowd-Belief Estimation

If insider data eventually proves useful for crowd-belief estimation \(market expectation modeling\), it would enter through the expectation pipeline, NOT through the institutional signal pipeline. The 13F-based signals \(coinvest\_score\_z, inst\_delta\_z\) measure institutional conviction from quarterly portfolio disclosures. Insider buying measures company-insider conviction from Form 4 filings. They are different data sources with different provenance rules.

---

## 13F Refresh Cycle

SEC 13F filings have a 45-day lag from quarter-end. Filings typically cluster in the final 3 business days before the deadline.

### Pre-Refresh Readiness \(`tools/prep_13f_refresh.py`\)

5 guards, all must PASS:

| Guard | Check |
| --- | --- |
| 1 | Most recent snapshot has valid institutional\_summary\_delta.json |
| 2 | coinvest\_score\_z has healthy variance \(SD > 0.10\) |
| 3 | PIT cache has entries within 3 days of today |
| 4 | SEC EDGAR endpoint is reachable |
| 5 | Dry-run: build\_institutional\_summary\(\) produces valid output \(>=80% coverage\) |

Writes baseline artifact: `artifacts/13f_pre_refresh_baseline_{date}.json`

### Cohort Quarantine \(`tools/check_13f_cohort_quarantine.py`\)

Run after new filings land. Compares pre-refresh vs post-refresh snapshots.

**Sections:**

- A: Manager-level diff \(filing counts, coverage\)
- B: Coverage diff \(tickers\_with\_signal, signal\_coverage\_pct\)
- C: Per-ticker score diff \(coinvest\_score\_z, inst\_delta\_z distributions\)
- D: Top-30 churn \(Jaccard similarity, entries/exits, rank movement\)

**Verdicts:**

| Verdict | Meaning | Action |
| --- | --- | --- |
| CLEAN | Normal refresh, minimal churn | Proceed |
| QUARANTINE | Significant score/rank disruption | Hold for review |
| PRODUCER\_AUDIT\_REQUIRED | Anomalous coverage or manager changes | Deep investigation |

Telegram alerting on QUARANTINE/PRODUCER\_AUDIT\_REQUIRED \(suppressible with `--no-alert`\).

### Contamination Window

After adding new managers, a contamination window opens \(typically 20 trading days\). IC measurements during this window are flagged as contaminated and excluded from clean IC calculations.

---

## Data Provenance Rules

- **Holdings truth source**: `production_data/institutional_summary.json` is canonical
- **CUSIP-first, not issuer-first**: Always reason from CUSIP -> canonical ticker
- **Raw EDGAR XML is debug-only**: Never build narratives from raw filing parses
- **If raw count != summary count**: investigate the summary pipeline first

---

## Key Biotech 13F Filers to Track

Per user preference: Fairmount Funds, Deep Track Capital, Logos Capital.
Also monitor BioPharm IQ Twitter \([https://twitter.com/BioPharmIQ](https://twitter.com/BioPharmIQ)\).

---

## Source Files

| Component | File |
| --- | --- |
| Manager Onboarding | `tools/onboard_manager.py` |
| 13F Cache Warmer | `tools/warm_13f_cache.py` |
| Institutional Summary Builder | `build_institutional_summary.py` |
| 13F Refresh Readiness | `tools/prep_13f_refresh.py` |
| Cohort Quarantine | `tools/check_13f_cohort_quarantine.py` |
| Snapshot Collapse Guards | `tools/verify_snapshot_integrity.py` |
| Manager Registry | `production_data/manager_registry.json` |
| Institutional Summary | `production_data/institutional_summary.json` |

---

# SECTION 2: OPERATIONAL STATE

> **SNAPSHOT DATA** \- The values below are point-in-time and go stale. Verify against current pipeline output before citing.

---

## inst\_delta\_z Current Status

*Last reviewed: 2026-05-04*

- **Zeroed in selector** \(2026-05-04\): ALERT confirmed, mean IC = -0.097 over 36 dates
- **Absent from production ranker**: the live pairwise artifact has only `coinvest_score_z` and `financial_score`
- **Historical audit note**: prior within-top-30 analysis found positive discrimination \(NW-t = +3.32\), retained as context only
- **Reinstatement conditions**: documented in governance log, requires IC recovery evidence

## 13F Filing Cycle Status

*Last reviewed: 2026-05-24*

- **Completed cycle**: Q1 2026 \(period ending March 31, 2026\) -- ALL THREE FILED May 15, 2026
- **Accession numbers**: Fairmount 0001104659-26-062419, Deep Track 0001856083-26-000003, Logos Global 0001172661-26-002196
- **Filing pattern**: All three filed on deadline day \(May 15\), consistent with Q1 2025 pattern \(all three also filed May 15, 2025\)
- **CIKs**: Fairmount 0001802528, Deep Track 0001856083, Logos Global 0001792126
- **Post-filing action sequence**: \(1\) Warm 13F cache, \(2\) Run cohort quarantine, \(3\) Check collapse guards \(coinvest\_score\_z SD\), \(4\) Refresh IC decomposition, \(5\) 5-day observation window before treating as production-grade
- **Q2 2026** \(period ending June 30, 2026\): filed 2026-08-14. Promoted into production coinvest only on 2026-11-01, through the operator-authorized fix with a clock reset \(see memo above\).
- **Next cycle**: Q3 2026 \(period ending September 30, 2026\). Filing deadline \~2026-11-14. Rebuild `coinvest_signals.json` afterwards; this is a manual, governed step.
- **SEC compliance**: SEC\_USER\_AGENT preflight check added \(2026-05-13\) for EDGAR fair-use policy

## Q1 2026 13F-HR Filing Summary \(COMPLETE\)

*Filed: 2026-05-15*

**Fairmount Funds** \(accession 0001104659-26-062419\):

- AUM \~$1.38B. Key new: DAMORA THERAPEUTICS \($225.7M, 16.3% of portfolio -- largest new position, NOT signaled by 13D/13G pre-filing\). Massive APGE trim \(-85.4%\), COGT trim \(-38.9%\). Exits: KINIKSA, NUVALENT. VRDN held \(3.9M shares at 3/31\). Post-Q1: VRDN stake raised to 14.04% via $20M purchase May 11 \(13D/A filed May 13\). ORKA 19.99% held.

**Deep Track Capital** \(accession 0001856083-26-000003\):

- AUM $6,124M \(was $5,609M, +9.2%\). 63 positions \(was 55\). SPDR ETF hedge $1,277M. Top equity: GH $308M, IMVT $286M, TARS $252M, VCYT $250M, GPCR $206M. 16 new positions including ALMS \($149M\), NUVL \($141M\), GMAB \($98M\), DFNT \($57M\). Biggest adds: JANX +1225%, OCUL +144%, COGT +121%. Exits: DVAX \($242M largest\), MNMD, XENE, BHVN, RAPT.

**Logos Global** \(accession 0001172661-26-002196\):

- AUM $2,003M \(was $1,655M, +21.0%\). 66 positions. Top: RVMD $194.5M, ERSA $180.4M, IDYA $115M, TERN $105.4M. Massive CNTA add \(+963%, now $84.4M top-6 holding\). New: UTHR \($47M\), MDGL \($44.5M\), XENE \($26M\). 15 exits including CDTX \($68.5M\).

**Coinvest signals**: VRDN \(FM 3.9M shares + DT 1.4M shares at 3/31; DT accumulated to 5.4M post-Q1\). ORKA \(FM 3.7M + DT 2.0M\). Triple overlap on CRESCENT BIOPHARMA only. DT+Logos 22 overlaps; top by combined value: IMVT $313M, VCYT $267M, GPCR $230M, CNTA $210M \(new -- Logos +963%\).

## Q1 2026 Early Signals (13G/13D, pre-13F-HR) -- VALIDATED

*Captured: 2026-05-11. Validated against 13F-HR filings 2026-05-15.*

**Fairmount**: 13D/13G signals mostly validated. COGT trim confirmed ($243M sale 3/31). ORKA 19.99% confirmed. VRDN accumulation confirmed (3.9M shares at 3/31). SURPRISE: DAMORA THERAPEUTICS ($225.7M, 16.3% of portfolio) was NOT signaled by any pre-filing 13D/13G -- largest new position was invisible until 13F-HR dropped.

**Deep Track**: VRDN accumulation was primarily post-Q1 (only 1.4M shares at 3/31 vs 5.4M by May per 13G). ALMS confirmed ($149M new). New positions not signaled by 13G: NUVL ($141M), GMAB ($98M), DFNT ($57M).

**Logos**: TENX and AVLO confirmed. CNTA massive add (+963%) was NOT signaled pre-filing.

**Lesson**: 13D/13G pre-signals capture ~60-70% of major moves but systematically miss sub-5% positions and non-reporting-threshold changes. The largest surprises (DAMORA for Fairmount, CNTA for Logos) were invisible until 13F-HR.

## 2026 YTD Model-Relevant Acquisitions

*Last reviewed: 2026-07-05. Full terms + sources: `artifacts/acquisitions/biotech_model_acquisitions_ytd_2026.md`. Registry: `production_data/corporate_actions.json`.*

| Ticker | Acquirer | Consideration | Status | Holder / watchlist impact |
| --- | --- | --- | --- | --- |
| ACLX | Gilead | $115.00 + $5.00 CVR | Closed 2026-04-28 | Paradigm |
| APLS | Biogen | $41.00 + $4.00 CVR | Closed 2026-05-14 | Deep Track \(Tier 1\) |
| KALV | Chiesi | $27.00 cash | Closed 2026-06-11 | Universe \(HAE\); no Tier-1 holder |
| CNTA | Eli Lilly | $38.00 + up to $9.00 CVR | Closed 2026-06-24 | Logos \(Tier 1\) / Farallon; **watchlist removed 06-24** \(17→16\) |
| NUVL | GSK | $124.00 cash | Pending — tender exp 2026-07-14 | Deerfield \(#1 position\) + Paradigm |
| APGE | AbbVie | $135.11 cash | Pending — Q3 2026 | Fairmount \(Tier 1\) / RTW / Affinity; Robinhood |

**CNTA watchlist note \(Jun 24\):** Eli Lilly acquisition of Centessa Pharmaceuticals closed June 24. CNTA ADS delisted. Watchlist reduced from 17 to 16 tickers effective Jun 24. Logos $84M position converts to deal consideration at $38/share + up to $9 CVR.

**Pending-deal rule:** `pending_acquisition` entries in `corporate_actions.json` are documentation-only until close — the loader does not gate NUVL/APGE from ranker/selector via `is_dead()` yet. Promote to `acquisition` with close `effective_date` when deals complete.

**Personal-holding overlap YTD:** APGE only \(Robinhood\). ORKA held but not acquired — excluded from log.

### Interpretation — do not overclaim

Six takeouts YTD \(4 closed, 2 pending\) is **positive evidence for universe construction and registry relevance**, **moderate evidence for surveillance value**, and **not sufficient evidence for ranker alpha** without a pre-announcement score/event study.

- The count validates where sophisticated capital clusters before pharma acts — not that the model predicted the deals.
- KALV: universe coverage, not Tier-1 holder signal. APGE: personal holding, not systematic pick proof.
- ACLX/NUVL registry fixes protect IC and event-return validity more than they prove selection skill. See full operator framing in the acquisitions log § Interpretation.

## Session-end learning

After completing this skill's task, if you encountered an unexpected behavior, constraint, API response, or workflow edge case, log it:

```
[LRN-YYYYMMDD-NNN]
Pattern-Key: SKILL_INSTITUTIONAL_SIGNAL_{description}
Area: hermes_ops | data_pipeline | research | portfolio
Promotion-lane: skill | none
Recurrence-Count: 1
Context: <one line — what happened>
Rule: <one line — what to do differently>
Suggested-Action: <patch to this SKILL.md, or none>
```

Recurrence ≥ 3 in 7 days → propose a patch to this `SKILL.md` via `tools/pattern_to_skillpatch.py`. Full protocol: see `self-improving` skill.
