# Coinvest 13F integrity fix + roster change — out-of-sample clock reset

**Date:** 2026-11-01
**Classification:** Tier 3 (coinvest/selector input, production hash) with Tier 4 elements (manager roster = signal definition; Foresite grouping is a methodology judgment)
**Authorization:** operator (Darren Schulz), in session 2026-11-01 — "Fix now, reset clock"; "Yes, one combined reset"; roster supplied by operator; ship-with-reset, retire (not delete), keep conditional tier.
**Protocol:** `docs/FORWARD_VALIDATION_PROTOCOL.md` §1 — authorized change during the NO_MODEL_CHANGE window; resets the out-of-sample clock.
**Hash rotation:** `governance/HASH_ROTATIONS.md` (2026-11-01 entry)

---

## 1. Defects found

| # | Defect | Evidence | Effect in production |
|---|---|---|---|
| D1 | `production_data/coinvest_signals.json` never rebuilt after Q1 2026 | file `as_of_date=2026-05-22`, `period_of_report=2026-03-31`; Q2 13Fs filed 2026-08-14 | coinvest_score_z (sole selector input) ran on Q1 holdings from June through October |
| D2 | Filing age frozen at build time | `days_since_latest_filing=7` stored per ticker; run_screen read it verbatim | every run since May treated 5-month-old holdings as "fresh"; the 90-day filing-age decay never applied |
| D3 | No staleness gate on 13F selection | PIT cache rule is "latest filing among last 8", no age limit | a filer that stops (or changes CIK) is carried forever — Broadfin's Q4 2025 book (old CIK 0001601692) persisted at recency weight 0.7 |
| D4 | One firm counted as several holders | coinvest counts per filer CIK | Foresite Capital IV/V/VI counted as three independent holders |

Note: D1 also means the 2026-08-17 → 2026-10-13 regenerated pseudo-PIT snapshots (`data/snapshots_pit_gap_2026q3/`) used Q2 13F holdings while production used Q1. Comparisons across that boundary mix two coinvest vintages.

## 2. Changes

1. **Builder** (`scripts/build_coinvest_features_from_13f.py` 1.0.0 → 1.1.0)
   - `MAX_REPORT_AGE_DAYS = 230`: drop a manager whose `period_of_report` is more than 230 days before as_of (latest legitimate filing ≤ ~137d; one late quarter ≈ +92d).
   - `firm_group`: filer CIKs of one firm form one holder unit; only members at the group's latest period count; books are summed.
   - Roster gate: cached managers not in the active roster are excluded (old caches still list retired managers).
   - Emits `latest_filed_at` per ticker; provenance lists `stale_managers_excluded`, `managers_not_in_roster`.
2. **Load path** (`run_screen.py::_refresh_coinvest_filing_age`): recomputes `days_since_latest_filing` and recency state from `latest_filed_at` at each run.
3. **Registry** (`production_data/manager_registry.json` v3.2 → v4.0)
   - Broadfin Holdings LLC (CIK 0002110839) onboarded as successor filer (13F-NT for Q4 2025, own 13F-HR from Q1 2026; Q1/Q2 filings list a single position).
   - `firm_group` on Foresite IV/V/VI and both Broadfin CIKs.
   - elite_core replaced by the operator roster of 30 managers; 26 prior elite managers moved to a new `retired` section (kept for reproducibility, excluded from coinvest). Conditional tier (6) unchanged.
4. **Forward validation** (`tools/run_forward_validation.py`, `tools/weekly_validation_summary.py`, `CANDIDATE.json`)
   - `oos_clock_start = 2026-11-02`; captures dated earlier are ineligible for SM-20260629-001.
   - Needed because the candidate `model_hash` covers only ranker/selector/decision engine files — an input change would otherwise not reset the clock.
5. **Rebuilt input**: `coinvest_signals.json` as of 2026-10-30 from the Q2 2026 PIT cache (prior Q1), 30 managers, 256/357 tickers with signal (was 279/338 at Q1).

### Roster (elite_core, 30)

Kept (21): Boxer, Caligan, Fairmount, Siren, Vestal Point, Vivo, Opaleye, Kynam, Frazier, Logos, TCG Crossover, Deerfield, Bain Capital Life Sciences, Sofinnova, Integral, Paradigm, Affinity, Deep Track, Baker Bros, Perceptive, Avoro.
Operator kept from prior roster (3): Suvretta, BVF, DAFNA.
New (6): Lynx1 (0001910456), Commodore (0001831942), Venrock Healthcare (filer Venrock Adviser LLC, 0001615982), ADAR1 (0001940272), Octagon (0001839435), MPM BioImpact (0001687078).
Retired (26): RA Capital, Ally Bridge, Redmile, Orbimed, Venbio, RTW, Tang, Casdin, Cormorant, EcoR1, Palo Alto Investors, Sectoral, Great Point, Avidity, Samsara, Krensavage, Acuta, Rock Springs, Sarissa, Foresite IV/V/VI, Broadfin (both CIKs), Soleus, Braidwell.

**Selection-bias caveat.** 26 of the 30 were chosen by trailing 3-year annualised return. Any backtest over that window is selection-biased and must not be cited as evidence. Forward capture from `oos_clock_start` is the only clean evidence. No ablation was run (operator chose ship-with-reset).

## 3. Measured impact — 2026-10-30, control vs treatment

Same code, same inputs; only `coinvest_signals.json` differs. Control reproduces the production 2026-10-30 top 30 exactly (30/30).

| | Control (Q1 file) | Treatment (fixed, 30 managers) |
|---|---|---|
| Top-30 overlap | — | 25/30 |
| Leaves | | ARGX, ASND, NBIX, SRRK, TARS |
| Enters | | ABSI, BLTE, PVLA, TRAX, TRVI |
| Filing age | 7 (frozen) | 77–78 (from 2026-08-14) |
| coinvest_score_z SD | 0.968 | 0.684 (collapse guard > 0.10: PASS) |
| coinvest coverage | 81.9% | 76.8% |
| Phase-2 health | FAIL (catalyst_7d_count_high, headwind_exposure) | FAIL (headwind_exposure) — catalyst ≤7d count 8 → 6 |

The SD drop is the 90-day half-life filing-age decay applying for the first time (≈0.55× at 77 days), not a signal collapse.

## 4. Disposition of pre-reset evidence

- 7 mandate-eligible LIVE windows (mean weekly excess +0.713%, t = 1.64) remain in `captures.jsonl` / `fills.jsonl` as recorded history; they no longer count toward the 20/40/52-window gates.
- Mandate counter restarts at 0/20 from 2026-11-02.
- If the PR merges after the 2026-11-02 production run, move `oos_clock_start` to the first run that uses the merged code.

## 5. Follow-ups (not in this change)

- Run 13F post-filing sequence after the first post-merge snapshot: `tools/check_13f_cohort_quarantine.py --pre-date 2026-10-30 --post-date <first post-merge>`; 5-day observation window.
- Q3 2026 13F deadline ≈ 2026-11-14: rebuild `coinvest_signals.json` (no longer automatic staleness — but still a manual governed step).
- PIT cache indexes now list 56 filers against a 30-manager cohort (`coverage_pct` > 100%); cosmetic, builder ignores off-roster filers.
- `institutional_summary.py` (inst_delta_z, zeroed in selector) does not yet apply firm grouping or the roster gate.
- `total_elite_aum_b` uses registry AUM estimates; new entries use Q2 2026 13F reportable value.
