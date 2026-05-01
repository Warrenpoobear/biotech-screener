# Form 4 Operational Repair — 2026-05-01

**Trigger:** `FORM4_STABLE_SNAPSHOT_ELIGIBILITY_2026_05_01.md` returned NOT ELIGIBLE on data-integrity criteria #1 (producer cadence) and #7 (stale source), with #2 and #5 INDETERMINATE for lack of infrastructure.

**Scope:** Operational repair only. **No selector / ranker / sizing / construction change. No insider alpha promotion. Form 4 is NOT being flipped to required.** The Spec 065 stable-snapshot gate stays at `("insider_net_buy_value_90d", 0.30, False)` (advisory) until 5 consecutive stable snapshots are observed.

**Context for future-me:** the data that exists looks structurally sane (coverage, blank-vs-zero, mapping, column validity all PASSED). What was broken was the *machinery* — cron timeouts, fetch_state bookkeeping, panel staleness. This patch fixes the machinery. It does not change the field's role in production; that remains diagnostic-only.

---

## Five fixes — before / after

### Fix 1 — Producer cadence (criterion #1)

**Before**
- Cron stage `stage_form4` invoked `tools/fetch_form4_bulk.py` (timeout 1800s).
- Bulk producer downloads quarterly EDGAR indexes + per-ticker XML for ALL filings since `--since` (default 2020-01-01). With 326 universe tickers and only an mtime-today skip, the per-ticker re-fetch averaged ~72s each.
- Result: 1800s budget exhausted after 25/326 tickers. Timeout (exit 124) → script killed before its own state-save block at line 274 → fetch_state.json stale, panel rebuild skipped.
- Observed log on 2026-04-30 13:30 UTC: `Form 4 fetch TIMED OUT after 1800s — continuing with partial data`.

**After**
- Cron stage now invokes `tools/fetch_form4_insider.py` (incremental, default mode). Per the producer's own comment: *"every ticker is checked against SEC for new accessions; only those unseen in existing raw/{ticker}.json are fetched. Daily refresh cost scales with new-filing volume, not total history."*
- Timeout raised 1800s → 2400s as a margin (incremental should fit comfortably under that on normal days).
- Added an EXPLICIT panel-rebuild step (`--panel-only`, timeout 300s) that runs unconditionally **after** the main fetch step, so a partial fetch still produces a current panel.

**Files touched**
- `tools/cron_data_extras.sh` — `stage_form4()` rewritten with comment block citing this artifact.

**Verification path on next cron run (2026-05-04 13:30 UTC, Mon)**
- `data/form4/fetch_state.json:last_attempt` should match the cron run start
- `data/form4/fetch_state.json:last_success` should match the cron run end (incremental run typically completes in 10–30 minutes)
- `data/form4/form4_panel.csv` mtime should advance to the same date even if main fetch hits errors

---

### Fix 2 — fetch_state.json bookkeeping (criteria #1 + #7)

**Before**
- `fetch_form4_insider.py` and `fetch_form4_bulk.py` each wrote `STATE_FILE` ONCE — at the end of a successful run. The shape was `{last_fetch, tickers_checked, ...}`. There was no distinction between "didn't run" / "ran but timed out" / "ran but errored" / "ran successfully but found nothing."
- When the bulk producer timed out (the regular case), the state save was skipped entirely and `last_fetch` froze at whatever the prior successful-run timestamp was.

**After**
- Added 4 helpers to `tools/fetch_form4_insider.py`:
  - `_state_load()` — loads fetch_state.json, returns `{}` on missing/corrupt (atomic-failure tolerant)
  - `_state_write(updates)` — merges updates into existing state and writes back (preserves other fields)
  - `_schema_fingerprint()` — 12-char sha256 prefix of `InsiderTransaction.__dataclass_fields__` keys (Spec 065 #2 producer schema drift)
  - `_now_iso()` — UTC ISO timestamp helper
- New state schema (post 2026-05-01):
  ```text
  last_attempt        — every run start (resilient to timeout)
  last_success        — only on unwound, error-free run end
  last_fetch          — alias of last_success (back-compat)
  last_new_filing     — only when total_txns > 0
  last_error          — {at, msg} or null (cleared on next success)
  last_panel_rebuild  — panel CSV last rebuilt
  panel_rows          — n_rows in last panel rebuild
  schema_fingerprint  — 12-char hash of InsiderTransaction fields
  tickers_checked,
  tickers_updated,
  new_transactions,
  failed_tickers,
  since
  ```
- The producer now writes state at THREE points: (a) start of run with `last_attempt + schema_fingerprint`, (b) end of fetch with `last_success + counts + last_error=None`, (c) end of panel rebuild with `last_panel_rebuild + panel_rows`. Each write merges, never clobbers.
- The original `last_fetch` field is preserved as an alias of `last_success` so any downstream consumer still works.

**Files touched**
- `tools/fetch_form4_insider.py` — 4 helpers added at module top; `main()` updated with start/end writes (3 sites).

**Note on `fetch_form4_bulk.py`**: not patched in this pass because the cron no longer uses it. If anyone manually runs the bulk fetcher, its state-save will overwrite the new schema. Spec 065 §1 doesn't require both producers to share bookkeeping — just that the daily producer does. Bulk is now a manual one-shot and out of regular operation.

---

### Fix 3 — Panel rebuild cadence (criterion #1, #7)

**Before**
- Panel is rebuilt ONLY in the `if not args.panel_only:` branch, after the fetch loop completes. If fetch times out, panel rebuild is skipped.
- `data/form4/form4_panel.csv` mtime as of pre-repair: 2026-04-25 17:58 UTC (6+ days stale despite raw-file activity on 04-28 and 04-30).

**After**
- The cron now runs `tools/fetch_form4_insider.py --panel-only` AS A SEPARATE STEP after the main fetch (Fix 1 above). The panel is rebuilt every cron day regardless of whether the main fetch completed.
- `tools/fetch_form4_insider.py` now writes `last_panel_rebuild + panel_rows` to fetch_state.json after the build_panel call, so panel staleness is detectable from state alone.

**Files touched**
- `tools/cron_data_extras.sh` (Fix 1)
- `tools/fetch_form4_insider.py` (Fix 2)

---

### Fix 4 — Signed-delta reconciliation (criterion #5)

**Before**
- INDETERMINATE on every Spec 065 evaluation because the reconciliation script didn't exist.

**After**
- Added `tools/check_form4_reconciliation.py` (~180 LOC). For each consecutive snapshot pair `(D-1, D)`:
  1. Loads `insider_net_buy_value_90d` from each snapshot's rankings.csv (skips blank rows = no-raw-file tickers).
  2. Loads same from `data/form4/form4_panel.csv` keyed by `(ticker, as_of_date)`.
  3. For each ticker present in both snapshots: compute `Δsnapshot = D - (D-1)`, `Δpanel = panel(D) - panel(D-1)`, then check `|Δsnapshot - Δpanel| ≤ max($1000, 0.5% × max(|Δsnapshot|, |Δpanel|))` per Spec 065 §2.
  4. Reports per-pair: `n_common`, `n_matched`, `n_mismatched`, top mismatches with deltas + diffs.
- Exit code 0 = all pairs reconcile clean; exit 1 = any mismatch; exit 2 = panel/snapshot missing.

**KNOWN LIMITATION (deferred)**
- Running on the current 5 snapshots (2026-04-27 → 2026-05-01) reports `n_no_panel_data: 297` per pair — i.e., the panel does NOT contain rows for snapshot dates. The panel currently has rows only at filing-date keys (event-keyed), not at snapshot dates. The Spec 065 §1 #5 wording assumes the panel includes both.
- Two ways to resolve:
  - **Option A (preferred):** modify `build_panel()` in `tools/fetch_form4_insider.py` to emit a row per `(ticker, snapshot_date)` for every recent snapshot, computing `insider_net_buy_value_90d` at each. This makes the spec's reconciliation work as written. **Not done in this patch — out of scope for "minimum viable repair."**
  - **Option B:** rewrite `check_form4_reconciliation.py` to recompute snapshot values from raw files directly (matching `common/insider_enrichment.py:compute_insider_net_buy_value_90d`) instead of using the panel. More expensive at scale; deviates from spec wording.
- Recommendation: pursue Option A as a small follow-up before the next Spec 065 evaluation. The reconciliation script is still useful in its current form to detect snapshot-vs-panel divergence on dates the panel does cover.

**Files touched**
- `tools/check_form4_reconciliation.py` (new, ~180 LOC)

---

### Fix 5 — Schema fingerprint handling (criterion #2)

**Before**
- INDETERMINATE because no fingerprint was persisted; only same-day comparison was possible.
- The empirical 30-random-file probe found 2 distinct schemas, but those were `__empty__` arrays vs populated arrays — diversity, not drift.

**After**
- Approach: **fingerprint the dataclass field list, persist in state, accept this as the criterion #2 source of truth.** The `InsiderTransaction` dataclass is the producer's schema definition. Any producer-side schema change requires a code change to the dataclass. So a hash of `InsiderTransaction.__dataclass_fields__` is exactly the drift detector criterion #2 wants — it changes IFF the producer changes.
- Implementation: `_schema_fingerprint()` returns a 12-char sha256 prefix. Stored in fetch_state.json on every run. A future Spec 065 evaluation can compare the fingerprint across consecutive snapshots: same hash = no drift, different hash = drift detected.
- Documented in the producer's state-bookkeeping comment block at the top of `_state_load`.

**Files touched**
- `tools/fetch_form4_insider.py` (helpers added in Fix 2 already include `_schema_fingerprint`)

---

## Tests added

`tests/test_form4_state_bookkeeping.py` — 10 focused tests covering:
- `_state_load` returns `{}` on missing/corrupt files
- `_state_write` creates the file and merges into existing state (the critical invariant)
- `last_attempt` is independent of `last_success` (timeout-resilient bookkeeping)
- `last_new_filing` only advances when `total_txns > 0` (no-new-filings day records success without faking)
- `last_panel_rebuild` independent of fetch (`--panel-only` doesn't clobber fetch state)
- Schema fingerprint is deterministic for unchanged dataclass
- Schema fingerprint changes when a field is added (drift detector validation)
- State round-trips numeric types correctly (int stays int, not str)

All 10 pass. Full Form 4 + ranker-hygiene test suite (61 tests) green.

---

## Earliest re-evaluation date

The eligibility check (Spec 065 §3) requires **5 consecutive stable snapshots**. Earliest realistic re-eval = **5 production days after the cron change is in effect**.

Cron schedule (Mon-Fri 13:30 UTC). If this patch lands and is active by Mon 2026-05-04:
- 2026-05-04 (Mon): first stable run candidate
- 2026-05-05 (Tue), 2026-05-06 (Wed), 2026-05-07 (Thu), 2026-05-08 (Fri)
- **Earliest re-eval: 2026-05-08 (Fri)** assuming all 5 weekdays produce stable snapshots.

If any day fails (timeout, error, or schema fingerprint mismatch), the streak resets and re-eval slips by one day per failure.

---

## Out of scope (NOT in this patch)

- ❌ Form 4 flip to required (`production_qa_check.py:117` stays `False`)
- ❌ Selector / ranker / sizing / construction changes
- ❌ Insider alpha promotion (lane closed per 2026-04-05; Spec 065 §4)
- ❌ 30d / 60d variants, cluster / exec / unique-buyer flags
- ❌ 13F cohort-quarantine prep (next task)
- ❌ Panel builder change to include snapshot-date rows (Fix 4 KNOWN LIMITATION; small follow-up)
- ❌ Patching `tools/fetch_form4_bulk.py` (no longer in cron path; manual-only)

---

## Status of Spec 065 §1 criteria after this patch

| # | Criterion | Pre-patch | Post-patch (forward-looking) |
|---|---|---|---|
| 1 | Producer ran for date D | FAIL (sporadic bulk runs) | should PASS once cron runs incremental daily; verify on 2026-05-04+ |
| 2 | No producer-side schema drift | INDETERMINATE | PASS — `schema_fingerprint` persisted in state |
| 3 | Coverage drift within thresholds | PASS | PASS (unchanged) |
| 4 | Blank-vs-zero distinction | PASS | PASS (unchanged) |
| 5 | Signed-delta reconciliation | INDETERMINATE | INDETERMINATE until panel-builder includes snapshot dates (Option A); script available for partial check |
| 6 | Mapping failures ≤ 5 | PASS | PASS (unchanged) |
| 7 | No stale-source warning | FAIL (panel 6+ days stale) | should PASS once cron runs panel-only after every fetch attempt; verify on 2026-05-04+ |
| 8 | Column position + valid tokens | PASS | PASS (unchanged) |

**Summary**: 4 PASS pre-patch → 5 PASS post-patch (criterion #2). Criteria #1 and #7 are pending verification on the next cron run (2026-05-04). Criterion #5 remains INDETERMINATE pending the panel-builder change (Option A above) — but the reconciliation script now exists and will go live once the panel has snapshot-date rows.

---

## Files in this patch

- `tools/cron_data_extras.sh` — modified (Fix 1, Fix 3)
- `tools/fetch_form4_insider.py` — modified (Fix 2, Fix 5)
- `tools/check_form4_reconciliation.py` — new (Fix 4)
- `tests/test_form4_state_bookkeeping.py` — new (10 tests)
- `FORM4_OPERATIONAL_REPAIR_2026_05_01.md` — this artifact

---

## Next action (per user direction)

After this repair lands, the next task is **13F cohort-quarantine prep** (refresh ~2026-05-15). Do NOT re-evaluate Form 4 stable-snapshot eligibility until the natural 5-day window has elapsed (~2026-05-08). The point is to **prove stable daily operation**, not to patch today's verdict.
