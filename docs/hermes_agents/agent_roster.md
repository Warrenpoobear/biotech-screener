# Hermes Agent Roster
Last updated: 2026-05-18
Total jobs: 19 (17 recurring, 2 one-shot) + 1 planned (not yet created)

All jobs deliver locally (Hermes job history). None push to external channels.
To inspect output: ask Hermes "show last run of <job name>".
To manage: ask Hermes "pause/resume/remove <job name>".

---

## Daily / Intraday

### crontab-integrity-check
- **ID:** 862557978653
- **Schedule:** Mon-Fri 08:00 ET
- **Toolsets:** terminal
- **Purpose:** Verify crontab is parseable and all 5 critical entries are present
  (cron_daily_production.sh, agent_heartbeat_checks.py, cron_data_extras.sh,
  cron_data_refresh.sh, cron_watchdog.sh). Detects silent crontab REPLACE wipes
  before they cause missed production runs.
- **Alert:** CRONTAB ALERT if any entry missing or REPLACE event in last 24h

### openclaw-fleet-triage daily
- **ID:** 4f360d005436
- **Schedule:** daily 18:00 ET
- **Skills:** openclaw-fleet-triage
- **Toolsets:** terminal, file, skills
- **Workdir:** /mnt/c/Projects/biotech_screener/biotech-screener
- **Purpose:** Read-only OpenClaw fleet health check — fleet_steward receipt,
  per-agent stale/fail triage, task audit. Also runs memory-write watchdog:
  flags agents where memory mtime > 7d older than artifact mtime (code bug,
  not infrastructure). Known standing cases: herald, shadow_monitor, ic_health_monitor.

### openclaw auth sync
- **ID:** 4cfe9fb5d466
- **Schedule:** every 6h
- **Toolsets:** terminal
- **Purpose:** Runs ~/.local/bin/openclaw-auth-sync to refresh per-agent
  auth-profiles.json from ~/.claude/.credentials.json. Prevents the OAuth
  drift pattern where all agents fail simultaneously with FailoverError.
- **Known issue:** Hermes scheduler stalls after WSL2 sleep and misses cycles.
  Confirmed 2026-05-05: stalled 39h, all 31 agents EXPIRED+DRIFT. Manual fix:
  run ~/.local/bin/openclaw-auth-sync then kick job via Hermes cronjob run.

### morning-briefing
- **ID:** a955f533907b
- **Schedule:** Mon-Fri 12:00 ET
- **Toolsets:** terminal, file
- **Workdir:** /mnt/c/Projects/biotech_screener/biotech-screener
- **Purpose:** Wake Robin morning briefing — live artifacts digest from the
  screener (rankings, shadow portfolio, signal health, fleet status).

### pdufa-proximity-alert
- **ID:** e84535b22a2a
- **Schedule:** Mon-Fri 08:15 ET
- **Toolsets:** terminal, file
- **Workdir:** /mnt/c/Projects/biotech_screener/biotech-screener
- **Purpose:** Checks for upcoming PDUFA/action dates and cross-checks against
  current portfolio holdings. Flags names within proximity window.

### pr-review-daily
- **ID:** 51537fae7635
- **Schedule:** Mon-Fri 14:00 ET
- **Toolsets:** terminal
- **Purpose:** Automated PR governance reviewer. Reviews PRs that touch
  production integrity (screener pipeline, ruleset, scoring). Read-only.

### hermes-run-ledger-supervisor
- **ID:** eaea558faaf1
- **Schedule:** Mon-Fri 08:00 ET
- **Toolsets:** terminal, file
- **Workdir:** /mnt/c/Projects/biotech_screener/biotech-screener
- **Purpose:** Verifies every scheduled Hermes job and OpenClaw cron job
  has run within its expected window. Catches silent scheduler stalls.

### biotech-output-contract-check
- **ID:** 90fd1ba6606f
- **Schedule:** Mon-Fri 19:00 ET
- **Skills:** biotech-screener-output-qa
- **Toolsets:** terminal, file
- **Workdir:** /mnt/c/Projects/biotech_screener/biotech-screener
- **Purpose:** End-to-end contract check on today's production snapshot.
  Validates rankings schema, signal distributions, top-30 composition.

### event-outcome-binder-watch
- **ID:** f7635b487132
- **Schedule:** Mon 10:00 ET
- **Toolsets:** terminal, file
- **Workdir:** /mnt/c/Projects/biotech_screener/biotech-screener
- **Purpose:** Checks coverage of realized event outcomes in the binder.
  Flags gaps between CRT resolutions and event_feedback artifacts.

### alpha-verdict-ledger
- **ID:** 131d000821c2
- **Schedule:** Fri 20:00 ET
- **Toolsets:** terminal, file
- **Workdir:** /mnt/c/Projects/biotech_screener/biotech-screener
- **Purpose:** Maintains and reports current verdicts for every signal arm.
  Weekly accounting of signal status (ACTIVE/SHADOW/RETIRED/HOLD).

### llm-token-usage-monitor
- **ID:** 2a37afd91266
- **Schedule:** daily 21:30 ET
- **Toolsets:** terminal, file
- **Workdir:** /mnt/c/Projects/biotech_screener/biotech-screener
- **Purpose:** Accounting and anomaly detection for LLM token usage across
  OpenClaw agents. Flags unusual spend spikes. Read-only.

### llm-token-usage-weekly
- **ID:** 4bb8509d2d8f
- **Schedule:** Sun 18:30 ET
- **Toolsets:** terminal, file
- **Workdir:** /mnt/c/Projects/biotech_screener/biotech-screener
- **Purpose:** Weekly LLM token usage digest and accounting rollup.



### aa-model daily tracker
- **ID:** 3d1e09988873
- **Schedule:** daily 18:30 ET
- **Skills:** aa-model-tracker
- **Toolsets:** terminal, file, skills
- **Workdir:** /mnt/c/Projects/asset allocation/asset-allocation
- **Purpose:** MODE A repo health — git status, pytest, ruff, latest run.
  Patches HERMES_TRACKING.md auto sections. Preflight check detects phase
  drift: diffs git log since last tracker sync and surfaces "N commits since
  last sync" as the first line so phase drift is never buried.

---

## Weekly

### biotech-screener weekly audit
- **ID:** ccb9b8e16844
- **Schedule:** Mon 07:00 ET
- **Skills:** biotech-screener-audit
- **Toolsets:** terminal, file, skills
- **Workdir:** /mnt/c/Projects/biotech_screener
- **Purpose:** Full read-only audit of the biotech screener — snapshot integrity,
  ruleset pinning, signal health, pipeline health. Runs before crontab check
  and production window.

### 91-180d-bucket-watch
- **ID:** d653cbc61a15
- **Schedule:** Mon 08:30 ET
- **Toolsets:** terminal, file
- **Purpose:** Tracks 91-180d portfolio bucket pct vs 55% policy target.
  Thresholds: >= 55% = RESOLVED (HOLD blocker cleared), >= 40% = REVIEW
  (approaching target, consider rebalance), < 40% = HOLD with current value
  and delta. Baseline as of 2026-05-01: 20.0%.

### weekly-signal-regime-sweep
- **ID:** 7e79501afb6e
- **Schedule:** Sun 14:00 ET
- **Skills:** signal-shared-regime-check, openclaw-fleet-triage
- **Toolsets:** terminal, file
- **Workdir:** /mnt/c/Projects/biotech_screener/biotech-screener
- **Purpose:** IC regime check across all load-bearing signals. Detects shared
  regime vs signal-specific degradation. Runs before inst-delta-z-recovery-watcher.

### inst-delta-z-recovery-watcher
- **ID:** 4013ddd98c6d
- **Schedule:** Sun 14:30 ET
- **Toolsets:** terminal, file
- **Workdir:** /mnt/c/Projects/biotech_screener/biotech-screener
- **Purpose:** Checks reinstatement conditions for inst_delta_z (zeroed in
  selector 2026-05-04, ruleset v1.14.0). Conditions: (A) rolling mean_ic of
  latest 10 dates > +0.02 AND (B) calibration_evidence event-IC positive.
  If both met: emits governance reopen alert. Governance log:
  INST_DELTA_Z_GOVERNANCE_LOG_2026_05_04.md

### forward-shadow-weekly-digest
- **ID:** 120e89e8edbb
- **Schedule:** Fri 19:00 ET
- **Toolsets:** terminal, file
- **Purpose:** Weekly shadow portfolio performance digest from
  artifacts/shadow_monitor/. Reports: cumulative PnL, excess vs XBI, max
  drawdown, win rate, per-sleeve attribution (0_30, 31_90, 91_180, less_binary).
  Flags RED if cumulative excess < -5% or max drawdown > 20%.
  Flags YELLOW if scorecard=HOLD or drawdown streak > 3.

---

## One-shot (auto-expire)

### ruleset-v1.14.0-first-run-sentinel
- **ID:** 5ab49c070c88
- **Schedule:** once 2026-05-04 18:15 ET
- **Toolsets:** terminal, file
- **Purpose:** Validates first production run under ruleset v1.14.0 (canonical id=8887576e; initial commit produced phantom hash 622edb77, corrected by commit `bd91b523d` 2026-05-05).
  Confirms metadata.json shows new ruleset_id, finds last v1.13.0 snapshot,
  compares selector_score distribution and top-25 overlap. Flags RED if
  overlap < 60% or mean shifts > 2 std devs. Report written to
  data/ruleset_v1.14.0_sentinel_report.txt.

### 13f-q1-cycle-inst-delta-check
- **ID:** aee119860782
- **Schedule:** once 2026-05-19 17:00 ET
- **Toolsets:** terminal, file
- **Workdir:** /mnt/c/Projects/biotech_screener/biotech-screener
- **Purpose:** 13F Q1 2026 cycle probe (filings due 2026-05-15). Reads
  history.jsonl for inst_delta_z mean_ic on dates after 2026-05-14, compares
  to pre-filing degradation baseline (-0.097). If mean_ic > -0.05: emits
  improvement alert and recommends reinstatement check. Hypothesis: inst_delta_z
  degraded on Q4 2025 13F filing date (2026-02-28); Q1 refresh may resolve it.

---

## Planned (not yet created)

### hermes-held-spec-ledger
- **ID:** —
- **Schedule:** TBD (proposed: daily after `build_hermes_knowledge_layer.py` completes)
- **Toolsets:** terminal, file
- **Purpose:** Phase B trigger for Spec 090 (Town-Hermes Bridge). Emits a
  `held_spec_ledger` event via `common.operator_delivery.send_operator_event`
  pointing at `artifacts/ops/held_spec_ledger/latest.md`. Required for Phase B
  activation; not yet created. Phase A (`OPERATOR_DELIVERY_DRY_RUN=1`) is the
  current state — no live job exists.
- **Blocker:** Spec 090 Phase B not started. TOWN_WEBHOOK_URL +
  TOWN_WEBHOOK_SECRET must be set in `.env` and `OPERATOR_DELIVERY_DRY_RUN=0`
  before this job can be wired.

---

## Governance cross-references

| Agent | Governance doc |
|---|---|
| inst-delta-z-recovery-watcher | INST_DELTA_Z_GOVERNANCE_LOG_2026_05_04.md |
| ruleset-v1.14.0-first-run-sentinel | INST_DELTA_Z_SIGNAL_HEALTH_GOVERNANCE_REVIEW_2026_05_04.md |
| 13f-q1-cycle-inst-delta-check | INST_DELTA_Z_GOVERNANCE_LOG_2026_05_04.md |
| weekly-signal-regime-sweep | artifacts/ic_dashboard/history.jsonl |
| forward-shadow-weekly-digest | artifacts/shadow_monitor/ |

## Skills backing recurring agents

| Agent | Skills loaded |
|---|---|
| biotech-screener weekly audit | biotech-screener-audit |
| openclaw fleet triage daily | openclaw-fleet-triage |
| aa-model daily tracker | aa-model-tracker |
| weekly-signal-regime-sweep | signal-shared-regime-check, openclaw-fleet-triage |

## Debug skills (loaded on demand, not wired to cron)

Stored in docs/hermes_skills/ and ~/.hermes/skills/devops/:
- openclaw-cron-scheduler-debug — 7-class cron/scheduler failure taxonomy
  (Class A: crontab REPLACE, B: WSL2 sleep, C: Hermes scheduler stall,
   D: watchdog loop, E: weekend false-positive, F: LLM/tool mismatch,
   G: announce/webchat delivery errors)
- openclaw-auth-sync — OAuth drift workaround; sync script + cron 4cfe9fb5d466
- openclaw-agent-scope-audit — 30-agent SOUL.md scope table + registry reference
- openclaw-session-routing-debug — auth drift, zombies, delivery channel failures
- openclaw-data-pipeline-debug — press release contamination, IC ALERT protocol

## Known operational issues (2026-05-05)

| Issue | Status | Fix applied |
|-------|--------|-------------|
| Hermes scheduler stalls after WSL2 sleep | Recurring | Manual: run auth-sync + kick cron 4cfe9fb5d466 |
| OpenClaw announce/webchat channel not resolvable in isolated sessions | Fixed | bestEffort:true on all 7 affected jobs |
| Auth-sync missed 39h (31 agents EXPIRED+DRIFT) | Resolved 2026-05-05 | Manually synced; cron kicked |

