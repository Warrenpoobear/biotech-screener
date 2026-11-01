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
