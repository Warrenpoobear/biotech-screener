# Hash Rotations
No rotations recorded as of 2026-05-16 (policy effective date).
See `AGENT_ROUTING_POLICY.md` for required fields.
Required fields:
- old hash
- new hash
- effective date
- affected surface
- reason for rotation
- expected downstream artifact impact
- reviewer

---

## 2026-11-01 — coinvest 13F inputs (operator-authorized clock reset)

Memo: `docs/governance/2026-11-01-coinvest-13f-integrity-clock-reset.md`

| Surface | Old hash (sha256[:16]) | New hash (sha256[:16]) |
|---|---|---|
| `production_data/coinvest_signals.json` | `05296939cf4d03c1` (as_of 2026-05-22, period 2026-03-31) | `f44bc5432fb51567` (as_of 2026-10-30, period 2026-06-30) |
| `production_data/manager_registry.json` | `60efdadc64db9a19` (v3.2) | `9cfea1d5236278c2` (v4.0) |
| Candidate `model_hash` (ast-v1) | `827c35a9ed3ee6e1` | unchanged — engine files not touched |
| Ruleset | `8887576e` | unchanged |

- **Effective date:** first production run on or after 2026-11-02 (`oos_clock_start` in `artifacts/forward_validation/CANDIDATE.json`).
- **Affected surface:** coinvest_score_z (sole selector input) via `coinvest_signals.json`; 13F manager roster; coinvest builder (`scripts/build_coinvest_features_from_13f.py` 1.0.0 → 1.1.0); filing-age load path in `run_screen.py`; mandate eligibility in `tools/run_forward_validation.py`.
- **Reason:** (1) coinvest had not been rebuilt since Q1 2026 and carried a frozen 7-day filing age; (2) no staleness gate on 13F filings (Broadfin moved CIK, old book persisted); (3) Foresite IV/V/VI counted as three holders; (4) operator replaced the elite roster (49 → 30 managers).
- **Expected downstream impact:** top-30 composition changes (see memo for the 2026-10-30 control vs treatment comparison); every rankings.csv hash from 2026-11-02 differs from what the prior inputs would have produced; forward-validation mandate windows restart at 0/20.
- **Reviewer:** operator (Darren Schulz) — authorized in session 2026-11-01; PR review pending.

## 2026-11-01 (D5) — coinvest CUSIP-first ticker resolution

Memo: `docs/governance/2026-11-01-coinvest-cusip-first-resolution.md`

| Surface | Old hash (sha256[:16]) | New hash (sha256[:16]) |
|---|---|---|
| `production_data/coinvest_signals.json` | `f44bc5432fb51567` | `c03652d20cfacc2c` (as_of 2026-10-30, period 2026-06-30) |
| `production_data/cusip_static_map.json` | `fce151f7d4995269` (352 entries) | `3785dd5d33b08a99` (399 entries) |
| Candidate `model_hash` (ast-v1) | `827c35a9ed3ee6e1` | unchanged |
| Ruleset | `8887576e` | unchanged |

- **Effective date:** first production run on or after 2026-11-02. Same out-of-sample clock start as the D1–D4 entry above; no additional reset.
- **Affected surface:** `_resolve_ticker` in the coinvest builder (1.1.0 → 1.2.0), the CUSIP static map, and coinvest_score_z via the rebuilt `coinvest_signals.json`.
- **Reason:** the builder trusted the cache's live-lookup ticker. That ticker was blank for some universe names and wrong for some non-universe issuers (Organon → ORIC and others), and it changed from one warm to the next.
- **Expected downstream impact:** 2026-10-30 Top-30 overlap 27/30 against the D1–D4 file. Every coinvest-derived output from 2026-11-02 differs.
- **Reviewer:** operator (Darren Schulz) — authorized in session 2026-11-01; PR review pending.

## 2026-11-03 — curated PDUFA calendar refresh

Memo: `docs/governance/2026-11-03-pdufa-calendar-refresh.md`

| Surface | Old hash (sha256[:16]) | New hash (sha256[:16]) |
|---|---|---|
| `production_data/pdufa_dates.json` | `30c418f5e664965e` (11 entries, 1 upcoming) | `747216259674ef58` (43 entries) |
| Candidate `model_hash` (ast-v1) | `827c35a9ed3ee6e1` | unchanged |
| Ruleset | `8887576e` | unchanged |

- **Effective date:** the first production run after merge. **Not yet decided**: merging lands inside the window that began at `oos_clock_start = 2026-11-02` (see memo §4).
- **Affected surface:** catalyst / regulatory calendar inputs (`event_ledger.py`, regulatory-calendar gate, event-EV paths).
- **Reason:** the curated calendar had not been edited since 2026-06-22. This adds 32 dates, each verified against its source SEC filing.
- **Expected downstream impact:** on 2026-10-30, Top-30 overlap 29/30 (TVTX out, PVLA in) and headwind exposure 59.9% → 56.6%.
- **Reviewer:** operator (Darren Schulz). PR open; merge pending.

## 2026-11-13 — production order switched to selector-only (operator-authorized; behavior change WITHOUT a hash change)

Memo: `docs/governance/2026-11-13-selector-only-production-switch.md`

| Surface | Old | New |
|---|---|---|
| `run_screen.py --ranker-mode` default | `pairwise_minimal` | `selector_only` |
| Candidate `model_hash` (ast-v1) | `827c35a9ed3ee6e1` | **unchanged** — the hash covers `ranker_engine.py`, `selector_engine.py`, `decision_engine.py`; the ranker dispatch is in `run_screen.py` |
| `production_data/ranker_v2_model.json` | unchanged | unchanged (scored as a shadow only) |
| Ruleset | `8887576e` | unchanged |

- **Effective date:** first production run on the merged code; `oos_clock_start` to be recorded in `artifacts/forward_validation/CANDIDATE.json` after that run.
- **Affected surface:** `final_score`, `actionable_rank`, `ranker_active` and every Top-30 from that run. On 2026-11-12 inputs the Top-30 overlap with the ranker order is 17/30.
- **Reason:** the ranker's `financial_score` weight was fitted on history with market-cap look-ahead; with point-in-time market cap the selector order alone did better (+0.57 pt/period, t = 1.63, 79 periods). Checklist v2 was not run; operator decision.
- **Expected downstream impact:** every `rankings.csv` hash differs from what the ranker would have produced; the forward-validation mandate restarts at 0/20; `ranker_shadow_comparison.json` reports `production_ranker=selector_only`, `shadow_ranker=pairwise_minimal`; the selector-only shadow ledger gains a `ranker` basket.
- **Hash gap:** `model_hash_match` will keep passing across this change. Until the hash covers the dispatch, `CANDIDATE.json.ranker_mode` is the record of which order is the candidate.
- **Reviewer:** operator — authorized in session 2026-11-13.

