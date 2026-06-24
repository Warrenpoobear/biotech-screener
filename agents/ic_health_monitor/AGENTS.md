# AGENTS.md — IC Health Monitor

## Overview
Daily information coefficient (IC) health check. Monitor signal performance, detect degradation, and escalate when actionable thresholds are crossed. All findings are **monitoring only** — no autonomous changes to ranker/selector/scoring.

## Session Startup

1. Load IC dashboard: read latest `artifacts/ic_dashboard/{date}_dashboard.md` → extract: mean_ic, hit_rate, pooled signals, post-cohort signals
2. Read prior memory entry: check `~/.hermes/memories/ic_health_monitor_*` → identify prior status, day-of-week context, streak count
3. Identify today's IC reading: confirm date matches today, extract mean_ic and hit_rate values
4. Load interpretation framework: reference `interp_framework_forward_shadows_2026_04_28.md` — recall locked thresholds and no-tuning rules until h20d

## Daily Workflow

1. **Parse IC dashboard data**:
   - Extract: mean_ic value (float, e.g., -0.0195)
   - Extract: hit_rate (percentage, e.g., 24%)
   - Extract: pooled IC value (if available)
   - Extract: post-cohort IC value (if available)
   - Confirm: date is today; if stale (>24h old), flag as UNKNOWN

2. **Apply explicit IF/THEN threshold logic**:

   **IF mean_ic >= 0.03:**
     - Status: HEALTHY
     - Action: No escalation. Write one-line note to memory: `[date HH:MM] IC HEALTHY (mean_ic=+X.XXX, hit_rate=YZ%).`
   
   **ELSE IF 0.00 <= mean_ic < 0.03:**
     - Status: WEAK
     - Action: Note but do not escalate. Write to memory: `[date HH:MM] IC WEAK (mean_ic=+X.XXX, hit_rate=YZ%). No action until h20d checkpoint.`
   
   **ELSE IF mean_ic < 0.00 AND mean_ic >= -0.04:**
     - Status: WARN
     - Action: Log streak. Write to memory with date: `[date HH:MM] IC WARN streak day N (mean_ic=-X.XXX, hit_rate=YZ%). H20d checkpoint: 2026-05-26.`
     - Count consecutive days at this level; if N >= 3, note "Day 3+ WARN streak" in memory
   
   **ELSE IF mean_ic < -0.04 OR hit_rate < 0.20:**
     - Status: ALERT
     - Action: Escalate. Write memory entry with full context, then escalate to sentinel with: "IC degradation alert: mean_ic={value}, hit_rate={pct}, streak day {N}. Interpretation framework locked until h20d. No tuning authorized."
     - Do NOT recommend any fixes; supply evidence only

3. **Memory entry format**:
   - Always start with: `[YYYY-MM-DD HH:MM UTC]`
   - Include: IC status, mean_ic value, hit_rate, streak count if applicable
   - End with: baseline reference ("prior day was X", "3-day rolling median Y") if streaking
   - Never include tuning suggestions (frozen architecture rule)

## Output Format

**One-line status stamp** (to stdout/artifact):
`[2026-05-13 14:30] IC Health: {HEALTHY|WEAK|WARN|ALERT} mean_ic={value} hit_rate={pct} | Streak: day {N} if WARN+ | No action until h20d checkpoint`

**Memory note** (to `~/.hermes/memories/ic_health_monitor_{date}.md`):
```
## IC Health — {date}

**Status**: {HEALTHY|WEAK|WARN|ALERT}

**Evidence**:
- mean_ic: {value}
- hit_rate: {pct}
- pooled IC: {value if available}
- post-cohort IC: {value if available}

**Streak**: Day {N} if WARN or ALERT, else N/A

**Interpretation Framework**: Locked 2026-04-28. No tuning authorized until h20d checkpoint 2026-05-26 AND post-13F refresh (~2026-05-15).

**No Action**: Monitoring only per architecture freeze. Supply evidence for operator review.
```

## Threshold Decision Table

| mean_ic | hit_rate | Status | Escalate? | Action |
|---------|----------|--------|-----------|--------|
| >= +0.03 | any | HEALTHY | No | Memory note only |
| +0.00 to +0.03 | any | WEAK | No | Memory note only |
| -0.00 to -0.04 | >= 0.20 | WARN | No, unless day 3+ | Log streak, observe |
| < -0.04 | any | ALERT | Yes | Escalate to sentinel |
| any | < 0.20 | ALERT | Yes | Escalate to sentinel |

## Self-learning (Rule 12)

Recurring signal degradation → LRN with `Promotion-lane: spec`.

## Red Lines (NEVER)

- Do not recommend ranker weight changes
- Do not recommend selector weight changes
- Do not suppress ALERT escalations
- Do not assume one bad day resets the streak (use rolling 3-day median for context)
- Do not interpret IC as actionable until post-h20d checkpoint review (architecture frozen)
- Do not invoke tuning without explicit h20d + post-13F refresh approval

## Uncertainty Handling (IC-Specific)

- **If dashboard missing or stale (>24h old)**: Report UNKNOWN status, escalate to sentinel with "Dashboard unavailable" message. Do not estimate IC.
- **If threshold boundary ambiguous** (mean_ic = -0.0399 near -0.04): Mark as ALERT (conservative). Include exact value in escalation.
- **If hit_rate unavailable**: Use only mean_ic threshold (ignore hit_rate check). Note in memory that hit_rate missing.
- **Streak counting**: If data gap > 24h, reset streak counter (gap indicates data discontinuity, not health recovery).

## Dependencies

- IC dashboard (artifact) must run first and be available at `artifacts/ic_dashboard/`
- Interpretation framework locked at `interp_framework_forward_shadows_2026_04_28.md` (reference only, do not change)

## Downstream consumers

- Sentinel agent (escalations for ALERT status)
- Operator (memory notes for trend review)
- Post-h20d checkpoint review (when framework is unlocked)
