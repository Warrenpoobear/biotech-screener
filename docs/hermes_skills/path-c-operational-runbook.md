---
name: path-c-operational-runbook
triggers:
  - "daily governance monitoring"
  - "path c window close 2026-06-03"
  - "ic unobservable decision"
  - "emergency exit conditions"
description: >
  Operational runbook for Path C temporary policy override (2026-05-28 to 2026-06-03).
  Daily monitoring checklist, decision automation, and emergency procedures.
---

# Path C Operational Runbook

**Effective period:** 2026-05-28 (snapshot) through 2026-06-03 (window close decision)

**Operator:** dschulz@wakerobin.co

---

## Quick Reference

| Action | Command | When | Owner |
|--------|---------|------|-------|
| **Daily Check** | `bash tools/daily_path_c_monitoring.sh` | Post-snapshot, ~10 AM ET | Operator (manual or cron) |
| **IC Status** | `python3 tools/monitor_forward_eval_ic.py` | Anytime, check window status | Operator (monitoring) |
| **Window Close Decision** | `python3 tools/path_c_window_close_decision.py` | 2026-06-03, after 10 AM ET | Operator (interactive) |
| **Emergency Exit** | See section 4 | If drawdown >2pp or cohort Jaccard <0.70 | Operator (immediate) |

---

## 1. Daily Monitoring (2026-05-28 through 2026-06-03)

### 1.1 Automated Daily Checklist

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener
bash tools/daily_path_c_monitoring.sh
```

This runs four checks post-snapshot:
1. **Forward Eval IC Status** — Queries `artifacts/forward_eval_ic_ledger.jsonl`; shows latest IC value against floor (0.0200)
2. **Portfolio Drawdown vs XBI** — Checks `final/portfolio_summary.json` for drawdown_vs_xbi_pp; warns if > 1pp, critical if > 2pp
3. **13F Cohort Stability** — Verifies governance memo exists; documents Jaccard >= 0.70 requirement
4. **Emergency Exit Conditions** — Lists the two hard triggers (drawdown, cohort quarantine)

**Expected output (early window):**
```
[PATH_C_MONITOR] 2026-05-28 10:15:22 — Daily governance monitoring

[1/4] Forward Eval IC Status
[IC_MONITOR] No IC data yet in window
[IC_MONITOR_JSON] {"status": "NO_DATA", ...}

[2/4] Portfolio Drawdown vs XBI
Drawdown vs XBI: 0.34pp ✓ NORMAL

[3/4] 13F Cohort Stability
✓ Path C governance memo exists
  Cohort target: Jaccard >= 0.70
  Last clearance: 2026-05-24 (Jaccard 0.875)

[4/4] Emergency Exit Conditions
  Hard trigger 1: Portfolio drawdown > 2pp relative to XBI → revoke immediately
  Hard trigger 2: 13F cohort Jaccard < 0.70 or new quarantine → escalate

[PATH_C_MONITOR] Daily check complete. Window closes 2026-06-03.
```

### 1.2 Manual IC Monitoring

Check the IC trend anytime via:
```bash
python3 tools/monitor_forward_eval_ic.py
```

Output includes:
- Latest observation date
- Latest mean_ic value
- Status: ABOVE_FLOOR / BELOW_FLOOR / NO_DATA
- JSON blob with full observations array

**Interpretation:**
- **Status = NO_DATA** (expected through ~2026-06-17): Forward eval gate is in cold-start. PIT cache doesn't yet have filled 20-day forward-return horizons. This is normal.
- **Status = ABOVE_FLOOR** (if IC appears before window close): mean_ic >= 0.0200; institutional consensus is generating positive predictive power
- **Status = BELOW_FLOOR** (if IC appears before window close): mean_ic < 0.0200; institutional consensus is NOT translating to return predictability

### 1.3 Continuous Monitoring (Optional Cron)

To automate daily checks via cron (runs post-snapshot, e.g., 10 AM ET):
```bash
# Add to crontab (or equivalent)
0 10 * * * cd /mnt/c/Projects/biotech_screener/biotech-screener && bash tools/daily_path_c_monitoring.sh >> /tmp/path_c_daily.log 2>&1
```

---

## 2. The IC Measurement Gap (Expected)

**Key fact:** PIT cache will NOT have filled 20-day forward-return horizons until mid-June (~2026-06-17).

Forward eval IC monitor will likely show **IC_UNOBSERVABLE** (NO_DATA) through window close on 2026-06-03.

This is **not a failure**. It's a natural consequence of:
- Forward eval gate requires 10+ prior snapshots with filled 20-day return horizons
- Observation date T requires return data through T+20
- Late May observations require return data into early June
- Real-time availability lag: ~2 weeks after observation date

**Implication:** The 2026-06-03 decision point will likely encounter IC_UNOBSERVABLE, requiring operator choice (extend or revert).

---

## 3. Window Close Decision (2026-06-03)

On **2026-06-03**, after production snapshot completes and daily monitoring runs:

```bash
python3 tools/path_c_window_close_decision.py
```

This automation guides you through the decision tree:

### 3.1 Scenario A: IC is Observable (Unlikely by 2026-06-03)

If mean_ic is available (filled 20-day horizons):

**Path A1: mean_ic >= 0.0200 (ABOVE_FLOOR)**
- ✓ **Decision:** Path C remains valid
- **Action:** Window closes successfully; continue with next governance cycle
- **Next:** Begin Path A design work post-freeze

**Path A2: mean_ic < 0.0200 (BELOW_FLOOR)**
- ✗ **Decision:** Path C governance override REVOKED
- **Action:** Revert to HOLD pending Path A
- **Next:** Escalate with evidence summary; begin Path A design work

### 3.2 Scenario B: IC is Unobservable (Expected)

If mean_ic is unavailable (PIT cache missing filled horizons):

**Operator decision required. Choose ONE:**

**Option B1: Extend Window (Recommended if conviction remains)**
- Extend Path C until first observable IC print (~2026-06-17)
- Evaluate against floor at that date
- Document extension rationale in governance ledger
- **Why:** Gives institutional signal time to accumulate predictive evidence; conviction on catalysts remains high (COGT, RVMD, SYRE, PRAX)

**Option B2: Revert to HOLD (Conservative if uncertain)**
- Revert to HOLD pending Path A design
- Closes override immediately
- Trigger Path A portfolio timing gate design (post-freeze)
- **Why:** Closes exception safely; moves to durable structural fix; retains governance discipline

**How to document:**
1. Run the automation to capture the decision logic
2. Create file `artifacts/readiness/WINDOW_CLOSE_DECISION_2026_06_03.md`
3. Record: date, IC status, decision (extend vs revert), rationale, next actions
4. Timestamp and sign off

---

## 4. Emergency Exit Conditions (Any Time Before 2026-06-03)

If either trigger fires before window close, revoke Path C immediately and escalate.

### 4.1 Hard Trigger 1: Portfolio Drawdown > 2pp vs XBI

**Indicator:** `portfolio_summary.json` shows `drawdown_vs_xbi_pp > 2.0`

**Action:**
1. Revoke Path C immediately
2. Revert to HOLD (policy constraints re-engaged)
3. Escalate to governance review with drawdown evidence
4. Document in governance ledger: timestamp, drawdown value, decision

**Who detects:** Daily monitoring or Hermes watchdog

### 4.2 Hard Trigger 2: Cohort Instability

**Indicator:** 13F cohort Jaccard < 0.70 OR new quarantine triggers

**Action:**
1. Escalate to governance review immediately
2. Investigate cohort change (manager additions/removals, filing dates)
3. Options: (a) re-validate cohort if change is benign, (b) revoke Path C if distortion is material
4. Document: what changed, why, decision, rationale

**Who detects:** Hermes 13F monitoring agent or manual 13F refresh check

---

## 5. Escalation Contacts

| Scenario | Contact | Action |
|----------|---------|--------|
| **Drawdown breach** | dschulz@wakerobin.co | Immediate revocation + governance review |
| **IC below floor (if observable)** | dschulz@wakerobin.co | Revert to HOLD + Path A design |
| **IC unobservable (expected)** | dschulz@wakerobin.co | Operator decision on extend vs revert |
| **Cohort instability** | dschulz@wakerobin.co + ops | Governance escalation + cohort validation |

---

## 6. Tools & Artifacts

| Tool | Location | Purpose |
|------|----------|---------|
| Daily checklist | `tools/daily_path_c_monitoring.sh` | Post-snapshot monitoring (4 checks) |
| IC monitor | `tools/monitor_forward_eval_ic.py` | IC status anytime (used by daily checklist + manual) |
| Window close automation | `tools/path_c_window_close_decision.py` | 2026-06-03 decision tree |
| IC ledger | `artifacts/forward_eval_ic_ledger.jsonl` | Extracted IC values (auto-populated post-snapshot) |
| Governance memo | `artifacts/readiness/GOVERNANCE_DECISION_PATH_C_2026_05_28.md` | Full decision rationale (committed) |
| Window close record | `artifacts/readiness/WINDOW_CLOSE_DECISION_2026_06_03.md` | Operator decision on extend/revert (to create) |

---

## 7. Timeline Summary

| Date | Action | Owner | Status |
|------|--------|-------|--------|
| 2026-05-28 | Path C governance decision APPROVED | Operator | ✓ DONE (commit 8cbe1648) |
| 2026-05-28 → 2026-06-03 | Daily monitoring (post-snapshot) | Operator | ⏳ ACTIVE |
| 2026-05-28 → 2026-06-03 | Forward eval IC ledger auto-populates | Pipeline | ⏳ ACTIVE (cold-start expected) |
| 2026-06-03 | Window close decision point | Operator | ⏳ PENDING (extend vs revert) |
| ~2026-06-17 | First observable IC expected (if waiting) | Market data | ⏳ EXPECTED |
| 2026-06-03+ | Path A design begins post-freeze | Product | ⏳ PENDING |

---

## 8. Key Insights

1. **Cold-start IC is expected.** The forward eval gate has been in production only since 2026-05-28. It needs 10+ prior snapshots with filled 20-day horizons to compute IC. By 2026-06-03, you'll likely see NO_DATA or cold-start warnings. This is normal and documented.

2. **IC_UNOBSERVABLE is a valid measurement state, not a failure.** If IC is unavailable at window close, it's because PIT cache real-time lag is real. The governance decision anticipated this; the operator decision tree accommodates it.

3. **Path C is not indefinite.** It's a controlled exception with hard exit conditions (floor or revert). Either observable IC validates the override, or operator chooses explicitly to extend/revert.

4. **Path A is the durable fix.** Post-freeze, portfolio timing gates will enforce max 30% in 0–7d, min 40% in 90+d, decoupling institutional signal strength from portfolio timing policy. Design spec: `docs/governance/PATH_A_PORTFOLIO_TIMING_GATES_SPEC_106_2026_06_25.md` (Spec 106).

---

## 9. Questions & Troubleshooting

**Q: The daily monitoring shows "No IC data yet in window". Is this a problem?**
A: No. This is expected through mid-June. The forward eval gate is computing IC correctly; PIT cache horizons just haven't filled yet.

**Q: What if the IC never appears before 2026-06-03?**
A: Then you hit Scenario B (IC_UNOBSERVABLE). Run the window close automation, which prompts you to choose: extend (wait until ~2026-06-17 for first IC) or revert (go back to HOLD now).

**Q: How do I know if drawdown is too high?**
A: Daily checklist reports it. Warning if 1–2pp, critical if > 2pp. Anything > 2pp is a hard trigger for revocation.

**Q: What's the difference between cohort quarantine (13F gate) and cohort instability (Path C trigger)?**
A: Two separate state machines. Quarantine (Jaccard < 0.70) is a Phase 2 health gate. Path C monitors the same cohort but triggers escalation if it changes mid-window. Both use same threshold (0.70).

---

## 10. References

- **Governance Decision:** `artifacts/readiness/GOVERNANCE_DECISION_PATH_C_2026_05_28.md`
- **IC Monitoring Framework:** `docs/hermes_skills/path-c-governance-monitoring.md`
- **Catalyst Concentration Diagnosis:** `artifacts/readiness/CATALYST_CONCENTRATION_DIAGNOSIS.md`
- **Forward Eval IC Tool:** `tools/forward_eval_ic_ledger.py`

---

**Last updated:** 2026-05-28  
**Operator:** dschulz@wakerobin.co  
**Status:** Active, ready for window open through close
