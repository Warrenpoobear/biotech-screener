# Governance Report — IC Council Decision-Outcome Ledger + Rotating Edge-Advocate

**Session date:** 2026-06-29 (Monday)
**Author:** Town assistant (Warrenpoobear)
**Operator:** Darren Schulz (Director of Investments, Wake Robin)
**Class:** documentation / governance close-out. NO change to scoring, ranker, selector, sizing, final_score, gates, event-EV math, cron, or production wiring.
**Verification basis:** repo state read live from `Warrenpoobear/biotech-screener` on 2026-06-29 ~21:20 ET (commit log, file contents, PR #449, CI checks). Every status claim below is grounded in that read, not in prior session assumptions.

> **IMPORTANT CORRECTION (vs. earlier draft of this report).** An earlier draft described the repo deliverables (schema doc, SKILL.md edge-advocate patch, ledger artifact, spec doc, Town mirror) as DEFERRED pending green CI. **That is wrong.** Repo verification shows they were committed to `main` earlier today. This report supersedes that draft. The deferral language applied to the Town-side spec's *own* sequencing; the actual repo already moved ahead of it.

---

## 1. Purpose

Memorialize, in one auditable record, the IC-council alpha-alignment work as it actually stands in the repo and in Town routines: what is committed, what was changed Town-side this session, the divergences that need reconciliation, and the open gates. The companion live spec is the Town Doc "Spec — IC Council Decision-Outcome Ledger + Rotating Edge-Advocate Role (v2)": https://www.town.com/content/document/nx7fqka8kcxaecx53rpp6babns89j7kv

---

## 2. Problem statement (why this work happened)

The `biotech-ic-council` skill was structurally **anti-false-positive only**: all five seats exist to reject a bad merge; none existed to catch a false *negative* (a real edge discarded because it is noisy or concentrated — the DEM rally-participation case). Separately, the recursive loop learned from review debates, not from whether past council calls were right about the market. Two in-bounds additions close that gap: a **Decision-Outcome Ledger (DOL)** and a **rotating edge-advocate role**.

---

## 3. Repo state — what is ALREADY COMMITTED to main (verified)

These were committed today, 2026-06-29 (commits `d2edd52`, `ce7f2ac`, `3f3888a`):

| Repo deliverable | Path | State |
|---|---|---|
| DOL schema + protocol doc | `skills/biotech-ic-council/references/decision-outcome-ledger.md` | COMMITTED |
| SKILL.md edge-advocate patch (rotating role, recommendation taxonomy incl. FORWARD_SHADOW_MANDATE / CONDITIONAL_EDGE_TRACKING, false-negative matrix row, DOL reference) | `skills/biotech-ic-council/SKILL.md` | COMMITTED |
| Overarching spec | `docs/COUNCIL_DOL_ALPHA_SENSITIVITY_SPEC.md` | COMMITTED (status line still says "DRAFT — awaiting operator approval") |
| Ledger artifact | `artifacts/ic_council/decision_outcome_ledger.jsonl` | COMMITTED — **live, one real row** |
| Town mirror | `docs/hermes_skills/biotech-ic-council.md` | COMMITTED (refreshed) |

**The DOL is not empty.** It holds one row:
- `ICD-20260629-001` — DEM YTD bootstrap control evidence; recommendation `FORWARD_SHADOW_MANDATE`; edge advocate = Seat 1, position `support`, `edge_advocate_false_negative_risk_flag: true`; shadow mandate `SM-20260629-001`; `evaluation_window_type: MODEL_EVALUATION`; `operator_confirmed: false`; `outcome_status: pending`; window start moved to 2026-06-29, due 2026-09-30.

**The shadow mandate has already been executed** in PR #449 ("forward bootstrap SM-20260629-001 — 118 windows, INCONCLUSIVE"):
- Median percentile 60.6 (gross), 54.0 (net of 25bps); 57.6% of windows >=50th; 37.3% >=75th; NON_RALLY median 62.2, RALLY 56.8.
- Success gate (median >=75 AND >50% windows >50th): NOT MET. Failure gate: NOT MET. Verdict **INCONCLUSIVE**.
- A follow-up commit (`1f8938e`) reframed the result as a **backfilled current-model replay (baseline only), NOT forward proof** — TRUTH_CARDs for Jan–Jun 2026 were generated 2026-06-28 by replaying the current frozen model on historical snapshots. SM-20260629-001 **remains OPEN**; resolution requires >=20 *post-mandate* forward windows from 2026-06-29 onward.

---

## 4. What was changed Town-side this session (applied, observe-only)

These are routine-prompt edits on the Town platform; none touches repo code or production logic.

### Self-Learning Loop Review (Monthly) — `jn791t221c7cgj6an8drvhaeed898k4e`
IC-council efficacy section now reads the DOL as primary source (metrics only over operator-confirmed + resolved rows), with a reconstruction fallback. Reports run count, triage split, flip rate, hold/reject & approve precision, false-negative catch rate, edge-advocate rotation check, LRN->promotion rate, gate-coverage bypasses, and stale-pending / unscoped rows, plus a load-bearing-vs-decorative verdict.

### PDUFA/Catalyst Resolution Tracker — `jn75pgxw2w3nnh24d0dwegtdqs86w4v8` (new Step 3b)
### Biotech Earnings Post-Mortem Tracker — `jn7766gymewr55pdza9x9m2j8h86ck4k` (new Step 4b)
Each adds a **read-only, propose-only** DOL stamp step: surface matching open rows in the existing alert/summary email as "PROPOSED — awaiting operator commit," never write or commit the repo file.

### Spec v2 Town Doc
Rewritten to v2 with all seven operator amendments + appendix amendment trace.

---

## 5. DIVERGENCES requiring reconciliation (the honest gaps)

This session's Town-side work was authored against the v2 *spec* field names. The **committed repo schema uses different names**. They are the same design but will not line up at runtime until reconciled.

| Concept | Town-side prompts / v2 spec | Committed repo schema (authoritative) |
|---|---|---|
| Window class | `eval_window_class` | `evaluation_window_type` |
| Window confirm flag | `eval_window_confirmed_by` (enum) | `operator_confirmed` (bool) + `set_by` |
| Advocate assignee | `edge_advocate_assignee` | `edge_advocate_seat` (+ `edge_advocate_assigned`) |
| Advocate stance | `edge_advocate_stance` | `edge_advocate_position` |
| Shadow text field | `shadow_mandate` | `forward_shadow_mandate_id` + `forward_shadow_metric` (+ separate `SM-*.md`) |
| Resolution commit attribution | `resolved_committed_by` | (not present; repo uses `operator_confirmed` + `outcome_evidence_refs`) |
| Shadow result | (implied) | `forward_shadow_result`, `advocate_call_resolved_correct` |

**Stale guard.** The Step 3b / 4b prompts contain a "deferred-aware no-op: skip if the ledger file does not exist" instruction. **The ledger now exists**, so that guard is stale and would mislabel the live ledger. Both prompts also still reference the v2 field names above.

**Recommended fix (NOT applied — needs operator go-ahead):** update the three routine prompts to (a) drop the "ledger does not exist" no-op and replace it with "match against the committed schema field names," and (b) use `evaluation_window_type` / `operator_confirmed` / `edge_advocate_seat` / `forward_shadow_result` / `advocate_call_resolved_correct`. Not changed in this session because reconciling to the repo schema is a distinct task and the repo is the authoritative side.

---

## 6. CI status (verified) — the active gate

CI is **RED**. On PR #449 (head `1f8938e`), all checks are failing as of 2026-06-29 23:12 ET:
- `pytest (3.10)` — failure
- `pytest (3.12)` — failure
- `lint` — failure
- `type-check` — failure
- `secret-scan` — failure
- `dep-audit` — failure
- `smoke` — failure
- Combined state: pending/failing; PR `mergeable: false`, `mergeable_state: dirty`.

This is consistent with the long-standing CI-RED condition (since 2026-05-08). Town cannot confirm the root cause from here (no host access); this needs host verification.

---

## 7. Governance posture on the "commit to git" instruction

The operator asked to commit this governance report to git. Per repository governance (no unsafe acceleration; do not merge/commit into a failing-CI mainline without explicit approval; observe-only defaults), and given CI is RED, the report is committed to a **dedicated docs PR** (option C) rather than pushed to `main` — keeping it off the red mainline and giving it its own review surface. The PR is docs-only (single file under `docs/governance/`) and touches no code, model, ranker, selector, sizing, or production logic.

---

## 8. Verified facts vs. recommended next steps

**Verified facts (repo + Town, this session):**
- Repo deliverables (schema doc, SKILL.md patch, spec doc, ledger artifact, Town mirror) are committed to `main`.
- DOL is live with one row (ICD-20260629-001), `operator_confirmed: false`, `outcome_status: pending`.
- Shadow mandate SM-20260629-001 executed in PR #449 → INCONCLUSIVE, then reframed as backfilled baseline; remains OPEN; forward window restarts 2026-06-29, >=20 post-mandate windows required.
- CI is RED (failing checks on #449); #449 not mergeable.
- Town-side: Loop Review + two trackers + spec v2 updated this session.

**Recommended next steps (none taken without operator clearance):**
1. Review/merge this docs PR once acceptable.
2. Reconcile the three routine prompts to the committed repo schema field names and remove the stale "ledger does not exist" guard (Section 5).
3. Set `operator_confirmed: true` (or edit) on ICD-20260629-001's evaluation window — currently false; metrics will not count it until confirmed.
4. Address CI RED (needs host verification) before any further repo writes are merged.
5. First DOL efficacy read at the next monthly Self-Learning Loop Review.

---

## 9. Artifact index

- Spec v2 (Town): https://www.town.com/content/document/nx7fqka8kcxaecx53rpp6babns89j7kv
- Governance report (Town Doc): https://www.town.com/content/document/nx7av50wc1eaqqbfy6kzwq0hmd89n7ej
- Committed spec (repo): `docs/COUNCIL_DOL_ALPHA_SENSITIVITY_SPEC.md`
- DOL schema (repo): `skills/biotech-ic-council/references/decision-outcome-ledger.md`
- Ledger (repo): `artifacts/ic_council/decision_outcome_ledger.jsonl` (row ICD-20260629-001)
- Shadow execution: PR #449 `research/forward-bootstrap-shadow-2026-06-29`
- Routines touched (Town): Self-Learning Loop Review (Monthly); PDUFA/Catalyst Resolution Tracker; Biotech Earnings Post-Mortem Tracker.
