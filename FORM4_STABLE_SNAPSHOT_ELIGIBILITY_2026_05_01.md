# Form 4 Stable-Snapshot Eligibility — 2026-05-01

**Verdict:** ❌ **NOT ELIGIBLE** — flip slips per Spec 065 §3 #2.
**Eval date:** 2026-05-01 (today, eligible-by-rule).
**Spec:** `specs/changes/spec_065_form4_stable_snapshot_gate.md`.

This is a **data-integrity gate** evaluation. Eligibility flips `production_qa_check.py:117` from `("insider_net_buy_value_90d", 0.30, False)` → `("insider_net_buy_value_90d", 0.30, True)`, converting the QA check from advisory to blocking. **It does NOT add Form 4 to selector / ranker / sizing / alpha** — that lane stays closed per the 2026-04-05 decision.

---

## 1. Five most recent consecutive snapshots evaluated

`2026-04-27` (Mon), `2026-04-28` (Tue), `2026-04-29` (Wed), `2026-04-30` (Thu), `2026-05-01` (Fri).
(2026-04-25 Saturday research run excluded as non-trading-day; would not change the verdict.)

---

## 2. Per-snapshot stability table (Spec 065 §1, 8 criteria)

| # | Criterion | 04-27 | 04-28 | 04-29 | 04-30 | 05-01 | Stable? |
|---|---|---|---|---|---|---|---|
| 1 | Producer ran to completion for date D | ✗ | ⚠ | ✗ | ⚠ | ✗ | **FAIL** |
| 2 | No producer-side schema drift | ? | ? | ? | ? | ? | **INDETERMINATE** |
| 3 | Coverage drift within ±5pp warning / ±15pp hard | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| 4 | Blank-vs-zero distinction preserved | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| 5 | Signed delta reconciliation within tolerance | ? | ? | ? | ? | ? | **INDETERMINATE** |
| 6 | Mapping failures ≤ 5 | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| 7 | No stale-source warning | ✗ | ✗ | ✗ | ✗ | ✗ | **FAIL** |
| 8 | Column at expected position + valid tokens | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |

**No partial credit per Spec 065 §1.** Any single fail in any of the 5 snapshots breaks the streak.

---

## 3. Evidence

### Pass criteria (#3, #4, #6, #8)

| metric | value | status |
|---|---|---|
| Field present rate (% of rows with `insider_net_buy_value_90d != ""`) | 100% across all 5 days, drift = 0pp | PASS — coverage well above 30% threshold |
| Blank count vs Zero count | blank = 0; zero = 90/108/109/109/109 | PASS — all 297 rankings tickers have a raw file (universe ⊆ raw); blank=0 is the natural correct state, not a collapse |
| Universe ⟷ raw mapping anomalies | 0 rankings tickers without raw file; only `_XBI_BENCHMARK_` synthetic placeholder is in raw without rankings (1 ≤ 5) | PASS |
| Column position | index 161 (spec said "~156"); +5 drift over time as new columns added | PASS — within tolerance |
| Token validity | 0 bad tokens (no `None`/`NaN`/`NULL`); all non-empty values parse as float | PASS |

### Fail criteria (#1, #7) — root cause: producer cadence + fetch_state staleness

**`fetch_state.json` shows producer last ran 2026-04-24T18:05 UTC** — 7 days ago.
- File mtime of `fetch_state.json` itself: 2026-04-25 17:58 UTC (so the file was rewritten 04-25 but the `last_fetch` field stayed stale — this is itself a data-integrity bug worth flagging).
- `failed_tickers: 0` (no producer failures recorded).

**Raw file mtime distribution** (when raw payloads were last touched):
- 2026-04-25: **314** files (bulk refresh — likely Saturday batch run)
- 2026-04-28: **2** files (small incremental)
- 2026-04-30: **26** files (small incremental)
- 2026-04-27, 04-29, 05-01: **0** files touched

**`form4_panel.csv` mtime: 2026-04-25 17:58 UTC** — the event-keyed panel that Pass B reads from has not been rebuilt since 04-25. New raw filings on 04-28 and 04-30 have not propagated to the panel.

This combination means:
- Criterion #1 strictly fails: the producer (`tools/fetch_form4_insider.py`) is not running on every snapshot date — only sporadically (04-25 batch + 04-28 + 04-30 incrementals).
- Criterion #7 strictly fails: the panel is 6+ days stale; a stale-source check would flag it.

### Indeterminate criteria (#2, #5) — infrastructure missing

**Criterion #2 (schema fingerprint)**: 30 random raw files show 2 distinct schemas — could be drift OR could be diversity (e.g., empty-array vs filings-present, or P/A vs S/D transaction shapes). Spec called for "hash of producer schema fingerprint" history; no fingerprint history exists. Cannot definitively pass or fail.

**Criterion #5 (signed delta reconciliation)**: Cross-snapshot value-changes are sane in magnitude (28 / 3 / 4 / 4 tickers per consecutive pair — plausible for 90d window roll + small incrementals). Spec called for `reconciliation script` against `data/form4/form4_panel.csv` within ±$1,000 absolute / ±0.5% relative; **the reconciliation script does not exist**. Cannot run the strict check.

---

## 4. Final verdict

> **Form 4 stable-snapshot gate: NOT ELIGIBLE on 2026-05-01.**
>
> 2 hard fails (#1 producer cadence, #7 stale source) + 2 indeterminate (#2 schema fingerprint history missing, #5 reconciliation script not built). 4 passes (#3, #4, #6, #8) are clean but insufficient.
>
> Per Spec 065 §3 #2: any data-integrity criterion failure in the last 5 snapshots → flip slips. **Earliest possible re-evaluation: 5 production days after producer cadence is restored to daily AND fetch_state.json reflects each daily run accurately.**
>
> Eligibility — even if it had passed — would NOT have added `insider_net_buy_value_90d` to selector / ranker / sizing / alpha. That promotion path stays closed per the 2026-04-05 decision and Spec 065 §4 explicit non-goals.

---

## 5. Exact next action

Before re-evaluating Form 4 stable-snapshot eligibility:

1. **Restore daily producer cadence.** Investigate why `tools/fetch_form4_insider.py` is running sporadically (04-25 / 04-28 / 04-30) instead of daily. Likely candidates: cron config, WSL uptime gap, or producer-internal "incremental skip" guard misbehaving.
2. **Fix fetch_state.json `last_fetch` update logic.** The file is being rewritten (mtime advances) but the `last_fetch` timestamp field is not. This is a producer bookkeeping bug — likely a single-line fix in `tools/fetch_form4_insider.py`.
3. **Rebuild `form4_panel.csv` and confirm rebuild cadence.** The panel is 6+ days stale despite incremental raw-file updates on 04-28 and 04-30. The panel-rebuild step is either missing from the cron or skipping on no-new-filings days when it should still re-aggregate.
4. **Build the reconciliation script (Spec 065 §1 #5).** Without it, criterion #5 will remain INDETERMINATE on every future eval. Spec called for `data/form4/form4_panel.csv` reconstruction matching snapshot deltas within ±$1,000 absolute / ±0.5% relative.
5. **Decide whether the schema fingerprint history (Spec 065 §1 #2) needs persisting.** Options: (a) build it as part of the producer's per-run output, (b) accept that any single-run fingerprint check is sufficient (relax the spec).

Earliest realistic re-eval (assuming items 1–3 are fixed in the next 1–2 days): **~2026-05-08** (5 production days post-fix). Items 4 and 5 can land in parallel without blocking the next eval, provided we accept INDETERMINATE on those two criteria for now.

---

## 6. Out of scope (deferred)

Per Spec 065 §4 explicit non-goals — and per the user's instruction at this evaluation:
- ❌ No scoring change.
- ❌ No selector / ranker / sizing / construction change.
- ❌ No insider-as-alpha promotion.
- ❌ No 30d / 60d variants, cluster / exec / unique-buyer flags.
- ❌ No 13F cohort-quarantine prep (next task; eval will follow this artifact).
- ❌ No threshold tuning of `_INSIDER_REQUIRED_COVERAGE` (stays at 0.30).
- ❌ No `common/feature_registry.py` re-add of `insider_net_buy_value_90d`.

---

## 7. Artifacts

- Spec: `specs/changes/spec_065_form4_stable_snapshot_gate.md`
- Producer: `tools/fetch_form4_insider.py`
- Enrichment: `common/insider_enrichment.py`
- Raw store: `data/form4/raw/{TICKER}.json` (342 files)
- Event panel: `data/form4/form4_panel.csv` (last rebuilt 2026-04-25 17:58 UTC — STALE)
- Producer state: `data/form4/fetch_state.json` (`last_fetch=2026-04-24T18:05` — STALE)
- Production QA registration: `tools/production_qa_check.py:117` (`("insider_net_buy_value_90d", 0.30, False)` — currently advisory)
- Memory: `project_insider_form4_pass_b_landed_2026_04_24.md`
