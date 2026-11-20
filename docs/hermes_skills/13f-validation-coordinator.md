---
name: 13f-validation-coordinator
triggers:
  - weekly 13F validation gate (Fridays 6:22 PM ET, starting 2026-05-31)
  - 55-manager cohort monitoring (post-h20d override)
  - Jaccard stability tracking (target ≥0.70 by 2026-06-15)
  - re-evaluation gate (2026-07-01)
description: >
  Monitor 55-manager cohort stability post-h20d override (OPTION_B_OVERRIDE_2026_05_26).
  Weekly validation starting 2026-05-31. Reads Jaccard similarity, inst_delta distortion,
  filing progress, and Top-30 churn. Reports stabilization trends. Tracks escalation
  triggers (Jaccard < 0.40 or inst_delta > 1.50). Re-evaluation gate at 2026-07-01.
---

# 13F Validation Coordinator — Weekly Cohort Monitoring (Updated 2026-05-26)

## Purpose

Monitor 55-manager institutional registry cohort stability post-h20d override authorization (2026-05-26). Execute weekly validation starting 2026-05-31 to track Jaccard similarity and inst_delta distortion trends. Report stabilization progress toward re-evaluation gate (2026-07-01).

**Status Update (2026-05-26):**
- ✅ 13F quarantine: CLEARED (48-manager cohort, Jaccard 0.875)
- ✅ Registry expansion: AUTHORIZED (55-manager cohort, Jaccard 0.463 baseline)
- ⚠️ 55-manager validation: FAILED (below threshold, but override approved)
- 🔄 Weekly monitoring: ACTIVE (starting 2026-05-31)

**Current status note (2026-11-01): the 55-manager cohort and `--pre-date 2026-05-15` baseline below are historical.**
- Registry v4.0 is now 30 `elite_core` + 6 `conditional`, with 26 `retired` managers excluded.
- Coinvest was rebuilt from Q2 2026 13F under an operator-authorized change with an out-of-sample clock reset.
- The next quarantine check is `python3 tools/check_13f_cohort_quarantine.py --pre-date 2026-10-30 --post-date <first post-merge snapshot>`, followed by a 5-day observation window.
- Do not compare Jaccard against the 48/55-manager baselines.
- See `docs/governance/2026-11-01-coinvest-13f-integrity-clock-reset.md` and `docs/MODEL_DOCUMENTATION.md`.

**Trigger conditions (weekly):**
- Every Friday 6:22 PM ET starting 2026-05-31
- Run: `python3 tools/check_13f_cohort_quarantine.py --pre-date 2026-05-15 --post-date [FRIDAY_DATE]`
- Artifact: `artifacts/13f_validation_verdict_55manager_weekly_[DATE].md`

---

## Validation Gates Reference (6 gates)

| Gate | Threshold | Status (May 19) | Checked |
|------|-----------|-----------------|---------|
| **G1: Filed Count** | ≥34 managers | **PASS** (46/48) | ✅ Pre-validation |
| **G2: Producer Freshness** | cache_as_of_date > pre_date | PENDING | May 20 snapshot |
| **G3: Manager Composition** | no unexpected churn | PENDING | May 20 snapshot |
| **G4: Position Completeness** | no Q4 stale | PENDING | May 20 snapshot |
| **G5: Top-30 Stability** | KS-stat inst_delta < 0.30, coinvest < 0.20 | PENDING | May 20 snapshot |
| **G6: Coverage/Diversity** | drop < 10pp | PENDING | May 20 snapshot |

**Critical decision gate:**
- **Cohort Jaccard ≥ 0.70** (Top-30 overlap target)
- Current (May 19): **0.536** (pre-refresh, not final)

---

## Decision Tree (Simplified)

```
START (May 20 post-validation)
├─ All 6 gates PASS + Jaccard ≥ 0.70
│  └─ PATH A: CLEAR → Quarantine LIFTS
│     ├─ Record clearance date + gate results
│     ├─ Unlock Spec 089 KG pilot
│     ├─ Unlock Spec 100 IC battery
│     ├─ Unlock Spec 094 + Spec 072
│     └─ Phase 2 Step 4 unblocks (May 21)
│
├─ Any gate FAIL or Jaccard < 0.70
│  └─ PATH B: EXTEND → Analyze failures
│     ├─ Identify which gates failed
│     ├─ Root cause analysis (data/modeling)
│     ├─ Options: re-baseline or remediate
│     └─ Set re-validation trigger
│
└─ Ambiguous result (mixed pass/fail)
   └─ PATH C: MANUAL → Governance review
      ├─ Summarize ambiguous gates
      ├─ Present decision options
      └─ Await human sign-off
```

---

## Step 1: Pre-Validation Checklist (May 20, ~4:00 PM ET)

Before validation runs, verify environment readiness:

```bash
# Verify production_data is fresh (wait for snapshot if not)
ls -lh production_data/universe.json
ls -lh production_data/institutional_summary.json

# Verify 13F validation harness exists and is ready
ls -lh tools/check_13f_cohort_quarantine.py

# Verify prior validation artifacts (if any)
ls -lh artifacts/13f_validation_* 2>/dev/null | tail -5
```

**Stop condition:** If production_data files are not fresh (timestamp < May 20 4:00 PM ET), WAIT for snapshot refresh before proceeding.

---

## Step 2: Read Validation Verdict (May 20, ~5:30 PM ET)

Once validation completes, read the verdict artifact:

```bash
# Find the latest validation artifact
VERDICT=$(ls -t artifacts/13f_validation_*.md 2>/dev/null | head -1)

# Extract key metrics
cat "$VERDICT" | grep -A 5 "Overall Validation Result\|Verdict:"
cat "$VERDICT" | grep "Cohort Jaccard\|Gate [0-9]:"
```

**Record these values:**
- Cohort Jaccard (numeric value)
- Gate 1–6 status (PASS/FAIL each)
- Overall verdict (CLEAR / EXTEND / MANUAL)

---

## Step 3: Route to Decision Path

### PATH A: CLEAR (if Jaccard ≥ 0.70 AND all 6 gates PASS)

**Actions (in order, all require explicit approval):**

1. **Record closure**
   ```bash
   # Capture git HEAD
   git log --oneline -1 > artifacts/13f_quarantine_lift_evidence.txt
   
   # Capture validation date
   date +%Y-%m-%d >> artifacts/13f_quarantine_lift_evidence.txt
   ```

2. **Commit clearance**
   ```bash
   git add artifacts/13f_quarantine_lift_evidence.txt
   git commit -m "13F quarantine lifted: Jaccard=[X.XX], all gates PASS
   
   Decision: Spec 089 KG pilot unlocked for May 21 launch
   Phase 2 Step 4 unblocks post-approval
   Validation artifact: 13f_validation_[DATE].md
   
   Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
   ```

3. **Memory update** (update `operating_state_post_spec_100_2026_05_17.md`)
   - Quarantine status: LIFTED
   - Date: [May 20-21]
   - Jaccard: [X.XX]
   - Unlock decision: Spec 089 ready to launch

4. **Notify governance** (document in memory)
   - Unlock timestamp
   - Which specs now unblock
   - Phase 2 Step 4 launch eligibility

5. **Proceed to Spec 089 launch checklist**
   - Reference: `specs/changes/spec_089_*.md`
   - Pre-launch gate: Phase 2 Step 3 verification COMPLETE
   - Launch condition: governance approval recorded
   - Timeline: May 21 start, May 23–24 completion

---

### PATH B: EXTEND (if any gate FAIL or Jaccard < 0.70)

**Root cause analysis (in order):**

1. **Identify failed gates**
   ```bash
   VERDICT=$(ls -t artifacts/13f_validation_*.md | head -1)
   cat "$VERDICT" | grep -E "FAIL|Gate [0-9]:" | grep FAIL
   ```

2. **Classify failure type**
   - **G2 (Freshness):** Cache not advanced — usually temporary, re-run when data ready
   - **G3 (Composition):** Manager churn — assess severity (how many new/gone)
   - **G4 (Completeness):** Stale positions — manager data gap (tactical)
   - **G5 (Stability):** KS-stat drift — potential signal distortion (strategic)
   - **G6 (Coverage):** Coverage drop — signal loss impact (strategic)

3. **Evaluate remediation paths**
   - **Re-baseline:** if failure is data/timing (G2, partial G4)
   - **Remediate:** if failure is modeling (G5, G6)
   - **Waive:** if failure isolated to few managers and nonessential

4. **Set re-validation trigger**
   - If re-baseline: +3 days after fix lands
   - If remediate: post-fix +2 days snapshot, +1 day rerun
   - If waive: document waived managers, proceed with caution

5. **Commit memo**
   ```bash
   # Create failure memo
   cat > artifacts/13f_quarantine_extended_memo_[DATE].md <<EOF
   # 13F Quarantine Extension Memo — [DATE]
   
   ## Failed Gates
   - [Gate name]: [reason]
   - ...
   
   ## Root Cause
   [Data issue / modeling issue / manager-specific]
   
   ## Remediation Path
   [Re-baseline / Remediate / Waive]
   
   ## Re-validation Trigger
   [Date/condition when validation re-runs]
   
   ## Expected Clearance
   [Date range]
   EOF
   
   git add artifacts/13f_quarantine_extended_memo_[DATE].md
   git commit -m "13F quarantine extended: [gate name] FAIL
   
   Reason: [root cause]
   Remediation: [path chosen]
   Re-validation: [trigger date]
   
   Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
   ```

6. **Update memory** (extend `13f_q1_2026_monitoring_live_2026_05_15.md`)
   - Quarantine status: EXTENDED
   - Failed gates: [list]
   - Remediation path: [choice]
   - Re-validation target: [date]
   - Impact: Specs 089/100/094/072 remain BLOCKED

7. **Monitor remediation progress**
   - If re-baseline: watch for next snapshot
   - If remediate: monitor fix execution
   - If waive: document decision and proceed with caveats

---

### PATH C: MANUAL (if result is ambiguous)

**Governance review needed:**

1. **Summarize ambiguity**
   ```bash
   VERDICT=$(ls -t artifacts/13f_validation_*.md | head -1)
   cat "$VERDICT" | grep -E "borderline|unclear|mixed|conflicting"
   ```

2. **Present decision options**
   - **Option 1:** Waive specific gates (list tickers/managers)
   - **Option 2:** Extend validation window (new target date)
   - **Option 3:** Remediate specific issue (identify root cause)
   - **Option 4:** Escalate (defer decision, gather more data)

3. **Governance sign-off** (required before proceeding)
   - Document chosen option
   - Record approval timestamp
   - File memo with decision rationale

4. **Proceed per chosen option**
   - If waive: route to Path A (proceed with caveats)
   - If extend: route to Path B (monitor remediation)
   - If remediate: route to Path B (fix + rerun)
   - If escalate: hold until resolution

---

## Step 4: Execute Path Actions

**All path actions require explicit user approval before execution.**

Common approval pattern:
```
User: "Proceed with PATH A (CLEAR)"
Agent: Execute Step 1–5 in sequence, confirm each step
User: "Next" (proceed to next step)
```

**Do NOT:**
- Skip governance approval
- Combine multiple steps into single commit
- Proceed beyond PENDING gates
- Waive gates without documented justification

---

## Reference: Prior Decisions

- **Quarantine created:** 2026-05-01 (Spec memo)
- **Last status:** 2026-05-19, 44/46 filed, Jaccard 0.536 (pre-refresh)
- **Decision tree:** `artifacts/13f_decision_tree_post_clearance_2026_05_19.md`
- **Validation runbook:** `docs/13f_q1_2026_refresh_runbook.md`
- **Phase 2 blocker:** Specs 089/100/094/072 locked until clearance

---

## Known Pitfalls

1. **Jaccard may jump sharply on refresh** — from 0.536 (6/48 filed) to higher value (44+/48 filed). This is expected. Threshold is still ≥0.70 (not higher).

2. **Gate 2 (Freshness) fails until cache advanced** — production_data cache_as_of_date must be ≥ May 20 snapshot date. This is a timing issue, not a defect. Re-run once cache updates.

3. **Gate 5/6 failures may require modeling review** — KS-stat drift suggests cohort composition impact. Escalate to governance if uncertain whether it's signal degradation or transient churn.

4. **Don't commit clearance before filing validation results** — ordering: (1) read verdict, (2) route to path, (3) execute path actions, (4) commit memo. Committing out of order causes audit trail confusion.

5. **Re-validation timing** — if extending, don't re-run validation until the condition that triggered extension is actually resolved. Premature re-runs will fail again.

---

## Deliverables

Upon completion (any path):

- **PATH A:** Clearance memo + spec unlock notification + Phase 2 Step 4 readiness
- **PATH B:** Extension memo + remediation tracking + re-validation trigger
- **PATH C:** Ambiguity summary + governance decision + implementation path

All delivered via committed artifact + memory update.

---

## Approval Gate Summary

```
Do NOT proceed unless:
☐ May 20 snapshot is fresh (timestamp ≥ May 20 4:00 PM ET)
☐ Validation verdict artifact exists and is readable
☐ Cohort Jaccard value is extracted
☐ All 6 gate results are documented
☐ Decision path (A/B/C) is unambiguous
☐ User has approved proceeding with chosen path
```

**Stop condition:** Any missing checkbox → WAIT or ESCALATE.
