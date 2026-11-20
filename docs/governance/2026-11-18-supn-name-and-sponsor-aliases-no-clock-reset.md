# SUPN company name and EU-registry sponsor aliases: no clock reset

**Date:** 2026-11-18
**Classification:** Tier 4 governance record. No code, model, ranker, selector, eligibility, sizing or gate behavior changes.
**Authorization:** operator, 2026-11-18. PR #709 was merged on operator instruction, and the operator ruled in session that it does not reset the out-of-sample clock ("record it as no clock reset").
**Hash rotation:** none. `model_hash a9a430154568e242` (`ast-v2`) and ruleset `8887576e` are unchanged.
**Clock:** `oos_clock_start` stays at 2026-11-17. Mandate SM-20260629-001 continues; `CANDIDATE.json` is not edited.

---

## 1. What changed (PR #709, merge `5491d420`, deployed 2026-11-18 before the 16:30 ET run)

Indivior acquired Supernus Pharmaceuticals on 2026-11-02, renamed itself Supernus, Inc., and began trading as SUPN in place of INDV on 2026-11-03 (#618). PR #709 corrects two things that the ticker change left behind:

- **Company name.** In `production_data/universe.json`, SUPN's `company_name` and `market_data.company_name` still read "Indivior Pharmaceuticals, Inc." and "Indivior PLC". Both now read "Supernus, Inc.", matching the entry's top-level `name`. `rankings.csv` shows the corrected name from the first run on the merged code.
- **EU-registry sponsor aliases.** `SPONSOR_ALIASES` in `wake_robin_data_pipeline/collectors/trials_collector.py` is shared by the EUCTR, CTIS and ISRCTN collectors. It was still keyed `INDV`, which is no longer in the universe, so those collectors searched SUPN by company name alone. The entry is now keyed `SUPN`. It carries the same six sponsor names as `collect_ctgov_data.py`, covering both Supernus and Indivior.

## 2. Why this is not a clock reset

- **The only model input affected is the set of EU-registry trials matched to one name, SUPN.** Prices, market data, 13F/coinvest inputs, eligibility rules and every other ticker are untouched. The CT.gov sponsor list already carried both names (#617, before the 2026-11-12 start).
- **SUPN is far outside the production basket.** On 2026-11-17 it was rank 103 and tier C. Production order is selector-only on `coinvest_score_z`, which does not read trial data.
- **No look-ahead.** Sponsor names select which registry records are collected. The records keep their own posted dates, and the existing PIT date filters still apply.
- **No ranking impact was measured.** If the Top-30 differs from the preceding day for a reason traceable to SUPN's trial records, that is a reason to revisit this ruling.

## 3. Operational note

The production checkout had uncommitted changes to `universe.json` and `universe_audit_log.jsonl` from the 2026-11-17 coverage refresh (ILMN, RPRX and STVN made active). To fast-forward, those files were backed up, the fast-forward was applied, the live copies were restored, and the two SUPN name edits and the `name_correction` audit entry were reapplied on top. After the fast-forward, ILMN, RPRX and STVN remained active and KALA remained `pending_coverage`.

## 4. Not changed

`collect_ctgov_data.py` still has a leftover `INDV` sponsor entry beside the correct `SUPN` one. It has no effect, because INDV is not in the universe, and is left for routine cleanup.
