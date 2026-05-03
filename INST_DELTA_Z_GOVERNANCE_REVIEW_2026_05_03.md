# inst_delta_z Governance Review — 2026-05-03

**Status:** Diagnose-only memo. No ruleset / selector / ranker / weight change is proposed or wired in this pass.
**Author:** Hermes (read-only triage), at operator request following 2026-05-01 fleet receipt RED verdict.
**Active ruleset at time of writing:** `2a3e79eb` (v1.13.0) — B6 selector + pairwise_minimal ranker + EW Top-30, adopted 2026-04-03.
**Trigger:** `ic_health_monitor` attention=HIGH on 2026-05-01, SIGNAL_ALERT on `inst_delta_z`.

---

## 1. FACTS (from artifacts, cited)

### 1.1 IC dashboard reading

Source: `artifacts/ic_dashboard/2026-05-01_dashboard.json` (schema `ic_dashboard.v1`, generated 2026-05-01T12:46:48Z).

Dashboard parameters:
- `lookback`: 60 prior dates
- `horizon`: 20 (forward-return window)
- `n_prior_dates`: 60
- `date_range`: 2026-02-18 → 2026-04-30 (the as_of_date is 2026-05-01 but the IC computation window ends 2026-04-01 because 20-day forward returns must mature)

Per-signal headline (`signals.<name>` block):

| Signal | Mean IC | Hit Rate | Latest IC | Health | n Dates |
|---|---|---|---|---|---|
| score_rank_pct | +0.0011 | 38.9% | -0.0316 | WEAK | 36 |
| clinical_optionality_pct_dev | +0.1074 | 91.7% | -0.0371 | HEALTHY | 36 |
| **inst_delta_z** | **-0.1006** | **8.6%** | **-0.0092** | **ALERT** | **35** |

Thresholds (`thresholds` block): healthy ≥ +0.030, warn ≥ 0.0, alert ≤ -0.030. inst_delta_z mean IC -0.1006 is ~3.4× below the alert threshold.

### 1.2 inst_delta_z is a load-bearing signal

`agents/ic_health_monitor/SOUL.md:39-41` declares the load-bearing set:
> Load-bearing signals (current DEM):
> - `clinical_optionality_pct_dev` — optionality anchor, primary sort driver
> - `inst_delta_z` — institutional delta, w=0.3 sort weight

Code-side weights (note: three different numbers exist, only one is the live ranker weight):

| Location | Value | What it is |
|---|---|---|
| `ranker_engine.py:117` | `RankerSignalSpec("inst_delta_z", 0.45)` | Live ranker weight in pairwise_minimal scorer |
| `decision_engine.py:279` | `binary_91_180_institutional_weight: float = 0.3` | Bucket-specific scale (quality_plus_institutional path only) |
| Selector spec per `13F_COHORT_QUARANTINE_PREP_2026_05_01.md` | `0.35 × inst_delta` | B6 selector second component |

FACT: the signal flagged ALERT participates in (a) the live ranker score, (b) the B6 selector, and (c) the 91-180d bucket institutional weight. It is not informational-only.

### 1.3 The 35-date IC sample is not 35 independent observations

Three contiguous date clusters carry byte-identical IC values in `inst_delta_z.per_date`:

| Cluster | IC value | Dates |
|---|---|---|
| C1 | -0.1309 | 2026-02-28, 2026-03-01, 2026-03-02 |
| C2 | -0.1380 | 2026-03-08, 2026-03-09 |
| C3 | -0.1561 | 2026-03-14, 2026-03-15, 2026-03-16 |

INFERENCE (separated below in §2): consistent with the documented 13F-cadence pattern in `13F_COHORT_QUARANTINE_PREP_2026_05_01.md` ("current `inst_delta_z` byte-identical 04-25/04-27/04-28 because no new 13F data"). These 8 dates likely represent ~3 independent inst_delta_z snapshots replayed against rolling forward returns.

INFERENCE: effective independent-sample size is closer to 30, not 35. This does not flip the conclusion (mean IC remains strongly negative) but it does mean the headline n is mildly overstated.

### 1.4 Outage-day contamination is NOT present in this dashboard

`RANKER_HYGIENE_NOTE_2026_05_01.md:21-29` documents 4 incomplete production days (2026-04-07/08/11/12) where `inst_delta_z=0.0` for all 297 rows. Spearman against a constant produces a meaningless number (the validation-table Spearman bug was fixed there).

FACT: those 4 dates do NOT appear in the inst_delta_z `per_date` array. The dashboard window ends 2026-04-01 (20-day forward-return horizon caps the trailing edge). The headline -0.1006 is not contaminated by the outage-day artifact.

### 1.5 Comparator behavior over the same window

Same 36/35-date window:
- `clinical_optionality_pct_dev` mean IC +0.1074, hit rate 92%, HEALTHY — primary alpha source is intact.
- `score_rank_pct` mean IC +0.0011, hit rate 39%, WEAK — composite is essentially noise but not anti-predictive.
- `inst_delta_z` mean IC -0.1006, hit rate 9% — the only signal that is reliably wrong.

A 9% hit rate (3 of 35 dates positive) at this magnitude is well past random. The signal is not "decaying"; it is sign-flipped relative to design intent over the observed window.

### 1.6 Q1 2026 13F refresh is imminent

`13F_COHORT_QUARANTINE_PREP_2026_05_01.md` line 3:
> Refresh window: Q1 2026 13F filings expected ~2026-05-15 (first Friday after the 45-day post-quarter SEC deadline of 2026-05-15).

FACT: the data feeding `inst_delta_z` is currently a stale Q4-2025-vs-prior delta. A material refresh lands in ~12 calendar days. The "self-heal" expectation in the prep memo is that the regime-distortion finding from 04-28 will resolve at refresh — but that finding was about within-snapshot ordering identity, not about the 60-date IC trajectory.

### 1.7 Production trail is healthy

- `agents/qa/memory/2026-05-02.md`: PASS, 5/01 artifacts re-verified intact (rankings.csv 827,948 bytes, sha256 match, screen exit 0, ruleset 2a3e79eb).
- `agents/sentinel/memory/2026-05-01.md`: drift OK, Phase-2 catalyst_7d FULLY RESOLVED 5→0, ranker bebe73f8 healthy.
- `agents/ops/memory/2026-05-01.md`: HOLD on bucket_drift_vs_policy only; does NOT escalate inst_delta_z.

FACT: ic_health_monitor is the only agent flagging this. The pipeline is otherwise green.

---

## 2. INFERENCE (clearly separated from facts)

1. **Substance, not artifact.** The -0.1006 mean IC is not driven by the known outage-day Spearman artifact (those dates fall outside the window) and is not driven by the calendar-cohort-quarantine artifact (those dates fall in April, only 04-01 is in window). The negative reading reflects 7 weeks of real-data behavior between mid-Feb and end-March 2026.

2. **Independent-sample correction is small.** Adjusting for the C1/C2/C3 byte-identical clusters drops effective n from 35 to ≈30. Mean IC and hit rate remain strongly negative either way; the headline number is robust.

3. **Coherent shape, not noise.** Of 35 dates, 32 are negative and only 3 positive (none above +0.08). This is not a "few bad days hurt the mean" pattern — it is a sustained sign-flip.

4. **The signal's job is being done by clinical_optionality_pct_dev.** With clinical_optionality_pct_dev at +0.107 mean IC and 92% hit rate over the same window, removing or down-weighting inst_delta_z is unlikely to leave the ranker without an alpha source.

5. **Refresh is a confound, not a fix.** The Q1 2026 13F refresh on ~05-15 will replace the stale Q4 cohort delta with a fresh one. Whether that fixes the sign-flip depends on whether the underlying mechanism (institutional buying as a contrarian indicator over 20-day horizons) is regime-driven or structural. The dashboard cannot answer that until ~30 days post-refresh — call it 2026-06-15 for the first IC reading that contains majority post-refresh data.

6. **The weights are not the same number in three places.** Per §1.2, ranker spec carries 0.45, the bucket weight is 0.3, the selector blend is 0.35. None of these are wrong on their own, but the SOUL doc claim ("w=0.3 sort weight") does not match the live ranker weight. Worth a separate ticket to reconcile naming/value drift across docs vs code, regardless of the disposition of this review.

7. **Carrying a load-bearing signal at -0.10 IC for 12+ days while waiting on the refresh has a cost.** With cum excess at +12.18% over 87 periods (`shadow_monitor` 2026-05-01) the headline P&L is healthy, but the +0.45 ranker weight on a sign-flipped signal is dragging at the margin. How much, exactly, is the kind of question that should be answered before the disposition is chosen — see Option C below.

---

## 3. OPTIONS (no recommendation embedded)

### Option A — Drop inst_delta_z from the ranker entirely

What:
- Set RankerSignalSpec weight from 0.45 → 0.0 in `ranker_engine.py:117`.
- Decide separately whether to also strip the 0.35 selector contribution and the 0.3 91-180d bucket weight, or leave them (different mechanism, different evidence base).
- Bump the ruleset id (2a3e79eb → next).

Trade-offs:
- ✅ Cleanest response to a -0.1006 IC reading on a load-bearing signal.
- ✅ Removes the most-contaminated input from rankings before the refresh.
- ❌ Requires a ruleset bump + governance log entry + re-run of golden snapshots.
- ❌ Loses the signal entirely; if the post-refresh data flips it back to predictive, you have to re-promote it.
- ❌ Locks in a decision before the refresh data arrives; the IC measurement window predates the refresh, so the option is being chosen with strictly stale evidence.

### Option B — Reduce weight (e.g., 0.45 → 0.10) and hold

What:
- Soft-cut the ranker weight; leave selector and bucket weights alone for now.
- Governance log entry; ruleset bump.
- Re-evaluate at 2026-06-15 (first post-refresh IC reading with majority new data).

Trade-offs:
- ✅ Caps the P&L drag from the sign-flip without burning the signal.
- ✅ Reversible — easy to step back to 0.45 if the post-refresh window shows recovery.
- ❌ Still requires a ruleset bump and a re-golden.
- ❌ "Reduce by some amount" is the option that most invites an arbitrary-magnitude decision; without an explicit IR-impact estimate the new weight is a guess.

### Option C — Watch and document, no production change before refresh

What:
- No ruleset change.
- File this memo as the formal governance log entry: signal flagged, evidence summarized, disposition pending data refresh.
- Add a calendar entry: re-evaluate when 60-date IC dashboard for ~2026-06-15 lands (first window with majority post-Q1-2026 13F data).
- Optional: stand up a shadow-portfolio variant with `inst_delta_z` weight=0 to quantify the actual P&L drag in real time. (Requires `backtest never modifies production` discipline; would need a separate spec ticket.)

Trade-offs:
- ✅ No production change made on stale evidence.
- ✅ Cheapest to execute; reversible by definition (nothing is changed).
- ✅ The shadow-variant sub-option produces the IR-impact estimate that Option B's "reduce by some amount" needs anyway.
- ❌ Carries the -0.10 IC ranker contribution for 12+ days while drag accumulates.
- ❌ Defers a decision rather than making one; if you intended to act this weekend, this is not the option.

---

## 4. WHAT THIS MEMO DOES NOT ANSWER

- IR / Sharpe / drawdown sensitivity to the 0.45 → 0 step. Would need a backtest re-run.
- Whether the sign-flip is a regime symptom (would mean it is shared with `coinvest_score_z`) or signal-specific. Worth a 5-minute check against the same dashboard windowed to coinvest before any decision lands.
- Whether the SOUL.md "w=0.3" / ranker_engine.py "0.45" mismatch is documentation drift or a real lineage issue. Separate ticket.
- Whether the byte-identical IC clusters in C1/C2/C3 trace cleanly to no-new-13F-data days on those dates (they likely do, but neither this memo nor the dashboard explicitly maps each repeat to its inst_delta_z source mtime).

---

## 5. NEXT STEP

Operator's call. This memo is intentionally quiet about which option to take. If a decision is wanted now, the next reasonable artifacts to produce — none of them touching production — are:

1. The coinvest-vs-inst_delta_z shared-regime check (one IC dashboard windowed query, no code change).
2. A governance log entry recording whichever option is chosen.
3. If Option B: an explicit IR-impact estimate from a backtest sweep of the new weight.
4. If Option C with the shadow-variant: a spec ticket for the variant configuration.

---

## 6. PROVENANCE

- Source artifact: `artifacts/ic_dashboard/2026-05-01_dashboard.json`
- Cross-checks: `agents/qa/memory/2026-05-02.md`, `agents/sentinel/memory/2026-05-01.md`, `agents/ops/memory/2026-05-01.md`, `agents/shadow_monitor/2026-05-01_monitor.md`
- Signal-spec citations: `ranker_engine.py:117`, `decision_engine.py:279`, `agents/ic_health_monitor/SOUL.md:39-41`
- Context memos: `13F_COHORT_QUARANTINE_PREP_2026_05_01.md`, `RANKER_HYGIENE_NOTE_2026_05_01.md`
- Author: Hermes / Claude — read-only triage session 2026-05-03
- No file outside this memo was created or modified by this review.
