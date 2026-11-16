# Out-of-sample clock start recorded: selector-only, 2026-11-16

**Date:** 2026-11-16
**Classification:** Tier 4 governance record. No code, model, ranker, selector, eligibility, sizing or gate behavior changes.
**Authorization:** operator, in session 2026-11-16: decided on 2026-11-15 to keep the selector-only order and let the live shadow ledger and the rollback rule decide; chose on 2026-11-16 to land the PIT cash fix (PR #682) before the run so it is part of the starting configuration; instructed that `oos_clock_start` be recorded after the first promoted selector-only run.
**Hash rotation:** none. `model_hash a9a430154568e242` (`ast-v2`) and ruleset `8887576e` are unchanged.

---

## 1. Out-of-sample clock start = 2026-11-16

`artifacts/forward_validation/CANDIDATE.json` now reads `oos_clock_start = 2026-11-16`. This completes the `pending_clock_reset` entry opened on 2026-11-13 (memo `2026-11-13-selector-only-production-switch.md`) and supersedes the 2026-11-12 start, which is kept under `prior_oos_clock_starts`.

The 2026-11-13 runs were the first on the selector-only code but failed promotion on the drift gate and took no capture. The start is therefore the first **promoted** selector-only run:

| Item | Value |
|---|---|
| Run | 2026-11-16, cron 16:30 ET, finished 16:53 (snapshot promoted 16:46) ET |
| Run manifest commit | `ae5ceb81` |
| Result | WARN; snapshot promoted; 44 gates; 39 PASS, 5 WARN, no errors |
| Drift gate | WARN, not blocked: dated override `2026-11-13-selector-only-switch` applied against 2026-11-12; name turnover 50% (dated override `config/drift_gate_overrides.json`, valid 2026-11-16 → 2026-11-20, PR #666) |
| `ranker_mode` | `selector_only` in the snapshot; forward-validation hard check `ranker_mode_match` passed (capture `ranker_mode=selector_only`, quality PASS) |
| Forward-validation capture | LIVE, quality PASS, model hash `a9a430154568e242` match, `ranker_mode=selector_only`, eligible for the mandate |
| Prices | through 2026-11-16 (350 tickers appended in the run, 0 failures) |

Changes that are part of the starting configuration because they were merged and deployed before this run (PRs #657 → #686):

- **Model order:** PR #665 — `run_screen.py --ranker-mode selector_only` (`final_score = selector_score`); the pairwise `minimal_v2` ranker is scored as a shadow only.
- **Input fixes:**
  - PR #682 — PIT financial store collects every cash and current-investment XBRL tag; cash uses the latest balance-sheet date with tag priority. Memo `2026-11-16-pit-financials-cash-securities-pre-clock.md`. Predicted on 2026-11-13 inputs: eligible 151 → 163, Top-30 overlap 25/30 (in COGT, VRDN, DNTH, CYTK, SYRE).
  - PR #676 — `de_vol_60d` refreshed every run. Memo `2026-11-15-vol-60d-refresh-pre-clock.md`.
  - PR #690 — Q3 2026 13F refresh of `coinvest_signals.json`, operator-chosen at 15:41 ET on the filing deadline: same roster (v4.0) and builder (1.2.0), so a routine refresh under `2026-11-15-pre-clock-decisions.md`, not a reset. Partial: 16 of 30 managers on Q3, 14 still on Q2 (not yet filed). On 2026-11-13 inputs: Top-30 overlap 25/30 against the Q2 file, eligible rank correlation 0.954. A second refresh follows once the rest have filed.
  - PR #662 — false splits removed from corporate actions; PR #663/#668 — ACHV → QUIT ticker change; PR #657 — CUSIP resolver duplicate keys; PR #673 — regulatory calendar refresh.
- **Governance and gates:** PR #666 (dated drift-gate override), #667 and #669 (pre-clock decisions; Rule 4 at −10/−15 pp; protocol §4 fail and rollback rules), #671 (candidate hash covers the selector config and order mode: `ast-v2`).
- **Capture-only shadows (no model effect):** PR #653 selector-only/ranker shadow ledger; PR #674 low-risk sleeve shadow (first LIVE row this run).
- **Tooling and CI only (no model effect):** #658, #659, #664, #670, #672, #675, #679, #680, #683, #684, #685, #686.

What the run showed against the prediction:

- Top-30 on 2026-11-16: RVMD, COGT, TNGX, DYN, IRON, EWTX, VRDN, TRVI, SVRA, ELVN, PRAX, SYRE, CLDX, ERAS, LBRX, DNTH, TARS, PHVS, VOR, BCRX, IDYA, DSGN, MIRM, SRZN, STOK, ALKS, DFTX, CYTK, MBX, KARD; 154 eligible names
- COGT, VRDN, CYTK, SYRE, DNTH: all five in, as predicted for PR #682: COGT #2, VRDN #7, SYRE #12, DNTH #16, CYTK #28. The PR #690 changes also landed as predicted: ERAS #14, TARS #17, VOR #19, DSGN #22, SRZN #24 in; DMRA, AMLX, IMCR, AURA, SGMT out (ranks 34-63)
- `de_vol_60d` varies by date: yes; 301 of 302 tickers differ from 2026-11-12, so PR #676 is live
- Shadow ledgers: selector-only/ranker first LIVE row with the ranker basket recorded (overlap with production 14/30; production and selector-only are now identical by construction); low-risk sleeve first LIVE row (120 names scored, prices as of 2026-11-16).

Consequences:

- Mandate SM-20260629-001 restarts at **0/20** from 2026-11-16. Captures dated 2026-11-12 → 2026-11-13 are history only. `WEEKLY_SUMMARY.md` regenerated under the new start shows 0 mandate-eligible windows.
- The selector-only rollback rule (protocol §4: paired t ≤ −1.65 against the shadow ranker basket after 20 windows) and the fail rule run from this start.
- A routine quarterly 13F refresh (same roster, builder and code) is not a reset (`2026-11-15-pre-clock-decisions.md`). The partial Q3 refresh (PR #690) is in the starting configuration; completing it after the remaining filings land is also not a reset.
- Any further authorized input or model fix resets the clock again.

## 2. Ledgers committed with this record

- `artifacts/forward_validation/captures.jsonl` (+1: 2026-11-16), `fills.jsonl`, `WEEKLY_SUMMARY.md` (regenerated).
- `artifacts/selector_only_shadow/captures.jsonl` (+1) and `summary.md`.
- `artifacts/low_risk_sleeve_shadow/captures.jsonl` (first row) and summary.
