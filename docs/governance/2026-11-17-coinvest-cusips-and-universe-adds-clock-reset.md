# Coinvest CUSIP mappings + RPRX, ILMN, STVN added to the scored universe — input change and clock reset (2026-11-17)

**Status:** operator-authorized 2026-11-17 ("do both in one batch"). Merging the PR that carries this record is the authorization. It is a **model-input change**, so the out-of-sample clock resets: `oos_clock_start` is recorded from the first promoted production run on the merged code (expected 2026-11-17), and mandate SM-20260629-001 restarts at 0/20. Both changes are batched so the clock resets once.

No weight, signal, selection rule, order, sizing or eligibility rule changes. `model_hash a9a430154568e242` (`ast-v2`) is unchanged; no hash rotation.

## 1. Coinvest: 14 CUSIP mappings

`production_data/cusip_static_map.json` gains 14 common-stock CUSIPs for active universe names whose 13F holdings were never resolved in any build (the builder is CUSIP-first since 1.2.0, so an unmapped CUSIP is dropped):

| Ticker | CUSIP(s) | Roster holders on Q3 2026 |
|---|---|---|
| AVTX | 05338F306 | 12 |
| CBIO | G2545C104, 38000Q201 | 10 |
| FTH | 81728A207 | 9 |
| QUIT | 004468500 | 7 |
| COAG | 423494103 | 7 |
| NKTR | 640268306 | 6 |
| AKTS | 01021M104 | 6 |
| OVID | 690469101 | 3 |
| STTK | 82024L103 | 3 |
| TEVA | M8769Q102 (ordinary shares) | 2 |
| SKYE | 83086J309 | 1 |
| IMMX | 45258H106 | 1 |
| MNPR | 61023L207 | 0 in this build |

Each issuer name on the 13F matches the universe name and each check digit is valid. Not mapped: convertible notes and warrants of mapped names, and one false name match (`02074J501` is Alpha Cognition, not TKNO). `coinvest_signals.json` is rebuilt (same roster, builder 1.2.0, PIT cache 2026-11-16): 273 tickers with a signal (was 261).

## 2. RPRX, ILMN, STVN: coverage exemptions

The three were held at `pending_coverage` because `tools/refresh_eligible_biotech_universe.py` requires clinical-trial and scientific-cartography coverage. None of them runs drug trials of its own (Royalty Pharma: royalty aggregator; Illumina: sequencing tools, has trial records but no drug interventions; Stevanato: drug containment supplier), so that coverage can never arrive.

- New `config/universe_coverage_exemptions.json` lists each name, the exempted fields and the reason.
- The refresh tool reads it (`load_coverage_exemptions`, `--exemptions-path`) and marks exempted fields `exempt`. Only `clinical_trials` and `scientific_cartography` can be exempted; company name and market data are always required. Tests added.
- Deployment: after merge, `tools/refresh_eligible_biotech_universe.py --finalize-collection --apply` on production sets the three to `active`.

## Measured impact (2026-11-16 inputs, `run_screen` control vs treatment)

Control = production data after PR #692. Treatment = this coinvest file + the three names active.

- **Top-30: unchanged (30/30), top-20 20/20.** Eligible rank correlation 0.971; eligible names 154 → 157.
- RPRX #147, STVN #155, ILMN #157: eligible, but large caps with 0–1 roster holders score below average once holder counts are residualised on market cap.
- Newly mapped names: COAG #32, QUIT #35, OVID #75, TEVA #92. AVTX, CBIO, FTH, NKTR, AKTS and STTK are ineligible on the existing `deep_drawdown` rule (−42% to −77% over the trailing window; checked for split artifacts, none found).

So the batch changes ranks below the Top-30 and which names can enter later, not today's portfolio. It is still an input change and is treated as one.

## Records

- Clock start: recorded in `artifacts/forward_validation/CANDIDATE.json` by a follow-up PR after the first promoted run on the merged code.
- Related: `docs/governance/2026-11-01-coinvest-cusip-first-resolution.md` (CUSIP-first builder), `docs/governance/2026-11-16-selector-only-clock-start.md` (superseded start).
