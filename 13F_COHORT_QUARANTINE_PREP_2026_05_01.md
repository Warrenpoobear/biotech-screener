# 13F Cohort-Quarantine Prep — 2026-05-01

**Refresh window:** Q1 2026 13F filings expected ~2026-05-15 (first Friday after the 45-day post-quarter SEC deadline of 2026-05-15).

**Scope:** Diagnostic/prep only. **No selector / ranker / scoring / manager-registry change in this prep pass.** The framework defined here is to be exercised when the refresh lands; nothing is wired into production cron until results are reviewed.

**Trigger memos:**
- `regime_post_cohort_change_distortion_2026_04_28` — current `inst_delta_z` byte-identical 04-25/04-27/04-28 because no new 13F data; self-heal expected at refresh
- `incomplete_production_run_fallback_2026_05_01` — when `institutional_summary_delta.json` is missing, `inst_delta_z=0` for all rows and looks like a regime (it's a data outage, not signal)
- `feedback_cohort_change_quarantine` — first snapshot post-13F-add has contaminated `inst_delta_z`/`rank_delta`
- `policy_coinvest_context_layer_2026_04_25` — coinvest is gate/context; do NOT strip without audited replacement

---

## 1. Source files / producers / consumers map

### Inputs (registry + summaries)

| File | Path | Mtime | Schema | Purpose |
|---|---|---|---|---|
| Manager registry | `production_data/manager_registry.json` | 2026-04-25 13:51 ET | `{elite_core: list[42], conditional: list[6], metadata}` per-manager keys: `cik, name, aum_b, style` | Defines which 13F filers count as "elite" |
| Institutional summary | `production_data/institutional_summary.json` | 2026-04-25 13:57 ET (FROZEN) | top-level: `schema_version, as_of_date, cache_as_of_date, elite_managers_total=37, tickers_with_signal=279, tickers_in_universe=342, signal_coverage_pct=81.58, tickers={342 entries}` | Per-ticker manager-overlap features at the most recent 13F cache date |
| Institutional summary delta | `data/snapshots/<date>/institutional_summary_delta.json` | per-snapshot | top-level: `as_of_date, prior_date, tickers_in_current/prior/common, tickers={n entries}`. **Currently `prior_date=2025-12-31`** (Q4 2025 13F filings); refreshes when Q1 2026 lands | Per-ticker change in manager overlap vs prior 13F refresh |

### Producers (write the inputs)

| Tool | Path | Trigger | Memory |
|---|---|---|---|
| Manager onboarding | `tools/onboard_manager.py` | manual; **never hand-edit registry** | `feedback_manager_acceptance_test` |
| 13F backfill | `tools/backfill_13f_history.py` | manual on registry change | — |
| 13F cache warm | `tools/warm_13f_cache.py` | scheduled (weekend or pre-refresh) | — |
| Institutional summary build | (embedded; rebuilds `institutional_summary.json`) | runs after 13F cache update | `regime_post_cohort_change_distortion_2026_04_28` notes mtime |
| Per-snapshot delta build | (embedded in `run_screen.py` rankings-assembly path; writes `institutional_summary_delta.json` to snapshot dir) | every production snapshot | `incomplete_production_run_fallback_2026_05_01` |

### Consumers (read inputs into rankings)

| Field | Computed from | Producer in run_screen | Consumer downstream |
|---|---|---|---|
| `coinvest_score_z` | `institutional_summary.json` per-ticker manager-overlap counts/AUM | `module_5_alpha_cohort.py` | selector (0.65 × coinvest), ranker_v2 (+0.02 coef), gate per Spec 072 vNext |
| `inst_delta_z` | `institutional_summary_delta.json` (current vs prior 13F) | `run_screen.py:5089-5092` (with default 0.0 fallback) | selector (0.35 × inst_delta), ranker prune logic, attribution |
| `inst_delta_net/new/exit` | same delta JSON | run_screen | diagnostic |

### Existing tools (reusable)

| Tool | Path | Notes |
|---|---|---|
| Cohort-expansion diff (manual) | `tools/diff_cohort_expansion_artifact.py` | Hard-coded for 04-25/04-27 transition (Saturday rebuild + 4 new managers). 238 LOC. **Useful template** for the generalized 13F-refresh diff but should not be reused as-is — it pins specific manager CIKs and ticker lists. |
| Manager integration test | `tools/test_manager_integration.py` | Pre-flight before onboarding new managers |

---

## 2. Pre/post 13F refresh comparison contract

A 13F refresh shifts `prior_date` in `institutional_summary_delta.json` from `2025-12-31` (Q4 2025) to `2026-03-31` (Q1 2026). Mechanically every per-ticker `inst_delta_z` recomputes; the SIGNAL_ALERT regime should clear.

The diff harness compares **State A** (last clean pre-refresh snapshot) vs **State B** (first post-refresh snapshot).

### Required diff sections

#### A. Manager-level diff (registry + summary cross-check)

| Field | Source | Pre | Post | Delta |
|---|---|---|---|---|
| Total elite_core managers | `manager_registry.json:elite_core` | n_pre | n_post | new+removed |
| Total conditional managers | `manager_registry.json:conditional` | n_pre | n_post | new+removed |
| Total elite_managers_with_filing | `institutional_summary.json` | 37 (now) | n_post | filings landed/dropped |
| Manager total AUM (sum aum_b) | `manager_registry.json` | $X B | $Y B | $Δ |
| New managers (cik in post but not pre) | registry diff | — | list[(cik, name, aum_b, style)] | — |
| Removed managers | registry diff | list | — | — |
| Managers with AUM change >5% | registry diff | — | — | list[(cik, pre_aum, post_aum, %Δ)] |
| Managers with style change | registry diff | — | — | list[(cik, pre_style, post_style)] |

#### B. Coverage diff

| Field | Pre | Post | Delta |
|---|---|---|---|
| `tickers_in_universe` | 342 | n_post | rebalance evidence |
| `tickers_with_signal` | 279 | n_post | coverage drift |
| `signal_coverage_pct` | 81.58% | n_post | drift in pp |
| Tickers gaining coverage (no signal pre, signal post) | — | list[ticker] | — |
| Tickers losing coverage (signal pre, no signal post) | list[ticker] | — | — |
| `cache_as_of_date` advanced | 2026-04-13 | should be 2026-03-31 (Q1 close) or later | mtime advance check |

#### C. Per-ticker score diff (rankings.csv)

For each ticker present in both A and B:

| Field | Statistic |
|---|---|
| `coinvest_score_z` | mean Δ, max abs Δ, n with \|Δ\| > 0.5 |
| `inst_delta_z` | mean Δ, max abs Δ, n with \|Δ\| > 1.0 |
| `coinvest_score_z` distribution shift | KS-stat A vs B |
| `inst_delta_z` distribution shift | KS-stat A vs B |
| `inst_delta_z` constant? | sd (must be > 0.1 — see §4 guardrails) |

#### D. Top-30 churn

| Metric | Value |
|---|---|
| Pre top-30 set | {tickers} |
| Post top-30 set | {tickers} |
| Jaccard(pre, post) | float |
| Names entering | list with attribution: which factor (Δcoinvest, Δinst_delta, Δfinancial_score) most explains entry |
| Names leaving | same with attribution |
| Rank movement |max\|rank_post − rank_pre\| over common names | int |
| Median rank movement | float |

Interpretation framework per `interp_framework_forward_shadows_2026_04_28`:
- HL Jaccard > 0.70 → coherent, real refresh
- HL Jaccard 0.40–0.70 → partial refresh, expected during cohort window
- HL Jaccard < 0.40 → weak/incoherent, requires manual investigation (could indicate registry corruption or producer misbehavior)

#### E. Sector / market-cap / development-stage skew

| Field | Pre top-30 distribution | Post top-30 distribution | Chi-sq stat |
|---|---|---|---|
| `industry_group` (or sector proxy) | bucket counts | bucket counts | drift indicator |
| `market_cap_bucket` | counts | counts | drift |
| `stage_bucket` (Spec 068 cohort key) | counts | counts | drift |

#### F. Attribution — true manager change vs producer artifact

Critical from April outage: distinguish **structural shift** from **incomplete-run fallback**.

For each top-30 entry/exit, classify:
- **REFRESH-DRIVEN**: explained by manager-level change (new/removed manager affecting that ticker) — expected, no quarantine
- **WINDOW-DRIVEN**: explained by `prior_date` shift (Q4→Q1) recomputing `inst_delta_z` — expected, fades over ~10 trading days
- **PRODUCER-ARTIFACT**: `institutional_summary_delta.json` missing OR `inst_delta_z` constant — NOT a regime, treat as outage per April memo
- **UNEXPLAINED**: doesn't fit above categories — investigate

---

## 3. Quarantine decision rules + thresholds

### Hard guardrails (run BEFORE any quantitative interpretation)

These are gates from the April outage. Per `incomplete_production_run_fallback_2026_05_01`, fail any of these and the diff is aborted as "data integrity, not regime":

- ❌ **Pre-snap missing `institutional_summary_delta.json`**: snapshot pre-state invalid; do not run diff
- ❌ **Post-snap missing `institutional_summary_delta.json`**: snapshot post-state invalid; do not run diff
- ❌ **`inst_delta_z` sd ≤ 0.1 in either snap** (effectively constant, indicating producer skipped): post-refresh diff is invalid; trigger producer audit, NOT cohort interpretation
- ❌ **`institutional_summary.json` mtime not advanced past 2026-04-25 13:57 ET**: producer didn't refresh; cannot diff
- ❌ **`prior_date` in delta JSON unchanged**: still pointing at 2025-12-31; the refresh hasn't landed; not yet time to diff

### Quarantine triggers

Once guardrails pass, run the quantitative diff. **Quarantine a name (or the whole top-30) when**:

| Trigger | Threshold | Quarantine action |
|---|---|---|
| Top-30 Jaccard (pre vs post) | < 0.70 | Mark all top-30 as **cohort-contaminated** for ~10 trading days; attribution-only; no rank-driven decisions |
| Top-30 Jaccard | 0.70–0.85 | Standard cohort window; attribution review of new entries/exits but no full freeze |
| Top-30 Jaccard | ≥ 0.85 | Normal — no special handling beyond per-name attribution |
| Manager count Δ | new + removed > 5 | Cohort-contaminated for ~3 weeks (matches the regime memo's prior 04-25 → ~05-15 window) |
| Coverage % drop | ≥ 10pp | Producer audit before any consumer reads inst_delta_z (likely producer fault, not regime) |
| `inst_delta_z` distribution shift | KS-stat ≥ 0.30 vs pre | Expected for refresh; flag but don't block — this IS the refresh's purpose |
| `coinvest_score_z` distribution shift | KS-stat ≥ 0.20 vs pre | Manual review — coinvest shouldn't shift drastically from a single 13F refresh; if it does, registry change suspected |
| Stage-bucket drift in top-30 | ≥ 5 names changed bucket between A and B | Verify whether Spec 068 stage_bucket re-classification happened in parallel (would confound the diff) |

### Standard cohort window

Once a refresh-driven cohort change is confirmed:
- Length: **10 trading days post-refresh** (default; matches regime memo's ~3-week window from 04-25 → ~05-15 ≈ 14 trading days but with the "first snapshot is most contaminated" decay implicit)
- During window: top-30 changes are **attribution-only**. No promotion of rank-driven decisions. No retraining.
- ic_health_monitor SIGNAL_ALERT for `inst_delta_z` is **expected** during window. Do NOT suppress.
- Selector behavior is **biased but not broken**. Do NOT retrain.
- Rank-change monitor noise is **partly artifact**. Do NOT recalibrate hysteresis.

### Quarantine exit criteria

To exit quarantine and resume normal interpretation:
1. **Distribution stable**: rolling 3-day sd of `inst_delta_z` distribution is within 20% of pre-refresh sd
2. **Top-30 stable day-over-day**: Jaccard(D, D-1) ≥ 0.85 for 3 consecutive snapshots
3. **No new manager onboarding pending**: registry mtime hasn't advanced since refresh
4. **SIGNAL_ALERT cleared**: ic_health heartbeat shows `inst_delta_z` back in normal range

If any criterion fails, the window extends to the next eligible exit check.

---

## 4. April-outage guardrails (operational, baked in)

These prevent re-running the 04-07/08/11/12 mistake of reading a producer outage as a regime. The diff harness **must** apply these before any interpretation:

### G1 — Snapshot completeness

```
∀ snap ∈ {A, B}:
  data/snapshots/<snap>/rankings.csv exists
  data/snapshots/<snap>/institutional_summary_delta.json exists
  inst_delta_z column has sd > 0.1
  Spearman computation guards against tied-constant inputs (already fixed 2026-05-01, commit 7213b2ef)
```

If any fails → **STOP**. Output: `INCOMPLETE_RUN_FALLBACK` verdict with the failure reason. Do NOT compute Jaccard or distribution stats.

### G2 — Producer freshness

```
production_data/institutional_summary.json mtime > pre-snap date
prior_date in B's delta JSON > prior_date in A's delta JSON
cache_as_of_date in institutional_summary.json advanced
```

If any fails → **STOP**. Output: `REFRESH_NOT_LANDED` verdict. Wait for next snapshot.

### G3 — Distinguish manager-level cause vs window-level cause

For every top-30 entry/exit attribution, the diff harness must check:
- Is the ticker in the new-managers' holdings? (manager-level cause)
- OR is the change explained by `prior_date` shift in `institutional_summary_delta.json` only? (window-level cause)

If MOSTLY window-level (>70% of changes attributable to prior_date roll, not new manager filings), the change is "expected refresh churn" not "regime."

---

## 5. Exact command/checklist for refresh day

When the refresh lands (~2026-05-15 or first weekday after Q1 2026 13F filings appear in EDGAR):

### Step 1 — Pre-refresh capture (do BEFORE refresh lands; latest weekday before)

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener

# Pin the pre-refresh state (capture by copy, not git — these are large/runtime)
PRE_DATE=$(date -d "yesterday" +%Y-%m-%d)  # or last weekday with a clean snapshot
mkdir -p data/snapshots/_13f_quarantine_2026q1
cp production_data/institutional_summary.json \
   data/snapshots/_13f_quarantine_2026q1/institutional_summary.PRE.json
cp production_data/manager_registry.json \
   data/snapshots/_13f_quarantine_2026q1/manager_registry.PRE.json
cp data/snapshots/$PRE_DATE/rankings.csv \
   data/snapshots/_13f_quarantine_2026q1/rankings.PRE.csv
cp data/snapshots/$PRE_DATE/institutional_summary_delta.json \
   data/snapshots/_13f_quarantine_2026q1/institutional_summary_delta.PRE.json
echo "$PRE_DATE" > data/snapshots/_13f_quarantine_2026q1/PRE_DATE.txt
```

### Step 2 — Detect refresh landed

Cron runs the bulk 13F producer (or `warm_13f_cache.py`). Check:

```bash
# Verify institutional_summary.json mtime advanced past pre-refresh date
stat -c '%y' production_data/institutional_summary.json
# Verify cache_as_of_date in JSON advanced
python3 -c "import json; print(json.load(open('production_data/institutional_summary.json'))['cache_as_of_date'])"
# Should be 2026-03-31 or later (Q1 2026 close)
```

### Step 3 — First post-refresh snapshot capture

After production cron runs and emits the first post-refresh `data/snapshots/<date>/rankings.csv`:

```bash
POST_DATE=$(ls -1 data/snapshots/2026-05-* | tail -1 | cut -d'/' -f3)
cp production_data/institutional_summary.json \
   data/snapshots/_13f_quarantine_2026q1/institutional_summary.POST.json
cp production_data/manager_registry.json \
   data/snapshots/_13f_quarantine_2026q1/manager_registry.POST.json
cp data/snapshots/$POST_DATE/rankings.csv \
   data/snapshots/_13f_quarantine_2026q1/rankings.POST.csv
cp data/snapshots/$POST_DATE/institutional_summary_delta.json \
   data/snapshots/_13f_quarantine_2026q1/institutional_summary_delta.POST.json
echo "$POST_DATE" > data/snapshots/_13f_quarantine_2026q1/POST_DATE.txt
```

### Step 4 — Run the diff harness (skeleton in §6)

```bash
python -m tools.check_13f_cohort_quarantine \
    --pre-date "$(cat data/snapshots/_13f_quarantine_2026q1/PRE_DATE.txt)" \
    --post-date "$(cat data/snapshots/_13f_quarantine_2026q1/POST_DATE.txt)" \
    --output artifacts/13f_cohort_quarantine_diff_2026_05_15.md
```

### Step 5 — Apply quarantine if triggers fire

If §3 quarantine triggers fire:
- Log the quarantine in a memory entry: `cohort_change_quarantine_2026_05_15.md`
- Update `MEMORY.md` Active Monitoring Windows with the new quarantine end date (D + 10 trading days)
- Wait for §3 exit criteria

If no triggers fire (clean refresh):
- Log "no quarantine" outcome and the actual diff numbers
- Resume normal interpretation; no special handling

### Step 6 — Exit-quarantine verification (~2026-05-29 if 10-day window)

Run a stability check at the proposed exit date:

```bash
python -m tools.check_13f_cohort_quarantine \
    --pre-date "$(cat data/snapshots/_13f_quarantine_2026q1/POST_DATE.txt)" \
    --post-date "$(date -d 'today' +%Y-%m-%d)" \
    --exit-check
```

Pass = exit criteria all met → memory updated `[resolved]`; quarantine ends.
Fail = quarantine extends; re-run on next eligible day.

---

## 6. Read-only harness skeleton

Provided as `tools/check_13f_cohort_quarantine.py` (skeleton; see file). Per scope discipline, this is a **diagnostic-only read** that:
- Loads pre + post snapshot rankings + summary + delta
- Applies G1/G2/G3 guardrails first; exits with explicit verdict if any fail
- Computes the §2 diff sections
- Compares against §3 thresholds
- Emits Markdown diff report (and JSON for downstream tools)

**The skeleton does NOT**:
- Modify any production state
- Write to `production_data/`
- Touch `manager_registry.json`
- Change scoring/ranker/selector
- Apply quarantine — that's a memory write, not a producer write

The skeleton is a starting point. Refinement before refresh-day use:
- Validate the §2.A manager-level diff against the actual registry shape (currently approximated)
- Calibrate `inst_delta_z` distribution-shift KS thresholds against historical refresh data (would need historical 13F refresh observations — only one cohort change has happened in the recent regime, on 04-25)

---

## 7. Explicit non-goals

This prep does NOT and the harness MUST NOT:

- ❌ Modify `production_data/manager_registry.json` (use `tools/onboard_manager.py` if a registry change is approved separately)
- ❌ Modify `production_data/institutional_summary.json` (producer-only)
- ❌ Modify any `data/snapshots/<date>/` rankings or delta JSON
- ❌ Wire into production cron (run manually on refresh day per §5)
- ❌ Promote insider data (Form 4 stays diagnostic per `Spec 065`)
- ❌ Change selector / ranker / scoring weights
- ❌ Re-run the Form 4 stable-snapshot eligibility check (separate task on ~2026-05-08)
- ❌ Tune coinvest_score_z computation or the 0.65/0.35 selector weights
- ❌ Force a re-quarantine on the existing 04-25 → ~05-15 window (it's already in effect; this prep is for the EXIT/refresh, not a re-entry)
- ❌ Merge or cherry-pick commits `7213b2ef` (Spearman hygiene) or `470987df` (Form 4 operational repair) — those are separate decisions

---

## 8. Queue (per user direction at this milestone)

| Date | Task | Status |
|---|---|---|
| 2026-05-01 (today) | 13F cohort-quarantine prep | in progress (this artifact) |
| ~2026-05-08 (Fri) | Form 4 stable-snapshot re-evaluation | pending — requires 5 stable producer days post-`470987df` |
| ~2026-05-15 (Fri) | Run §5 diff when Q1 2026 13Fs land | scheduled by this artifact |
| ~2026-05-22 (Fri) | vNext D7/D8/D9 verification + production-vs-coinvest forward-return re-test | scheduled (remote agent `trig_017s1kczCPEzp4ecNaPP4vYr`) |

---

## 9. Open questions (resolve before refresh-day, not in this prep)

1. **Exact quarantine window length**: 10 trading days is a default; the regime memo's prior 04-25 → ~05-15 window was ~14 trading days. Calibrate after one observed cycle.
2. **KS-stat thresholds (§3)**: 0.30 for `inst_delta_z` shift and 0.20 for `coinvest_score_z` shift are heuristic. Calibrate on the first refresh cycle's actual distributions.
3. **Whether to wire the diff harness into cron**: probably no — this should stay manual + on-demand. A cron run risks misfiring on partial refresh days.
4. **Sector/stage skew thresholds (§2.E)**: Chi-sq significance is sample-dependent; 30 names is borderline. Consider Fisher's exact for small bucket counts.
5. **Coordinate with Form 4 re-eval (~2026-05-08)**: if Form 4 re-eval also lands during cohort window, the eligibility verdict may itself be affected by inst_delta_z noise. Worth checking the dependency.

---

## 10. Files in this prep

- `13F_COHORT_QUARANTINE_PREP_2026_05_01.md` — this artifact
- `tools/check_13f_cohort_quarantine.py` — skeleton (next file)
- (memory) `13f_cohort_quarantine_prep_2026_05_01.md` — pointer entry, written separately

No production code changes. No tests added in this pass — tests should accompany the harness once it's elevated from skeleton to operational tool.
