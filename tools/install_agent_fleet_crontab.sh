#!/usr/bin/env bash
# install_agent_fleet_crontab.sh — print recommended WSL crontab lines for agent fleet.
#
# Does NOT modify crontab automatically. Operator runs: crontab -e
# and pastes the block printed by this script.
#
# Usage:
#   bash tools/install_agent_fleet_crontab.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cat <<EOF
# --- Agent fleet crontab reference (paste into crontab -e on WSL host) ---
# Repo: ${REPO_ROOT}
# TZ: cron inherits America/Detroit on operator WSL unless overridden.

# Herald health (weekdays 14:40 ET — after data refresh herald stage)
40 14 * * 1-5 cd ${REPO_ROOT} && python3 tools/herald_health_check.py >> logs/herald_health.log 2>&1

# Intraday mover — see tools/cron_intraday_mover.sh for full schedule
30 10 * * 1-5 ${REPO_ROOT}/tools/cron_intraday_mover.sh poll >> ${REPO_ROOT}/logs/intraday_mover.log 2>&1
0,30 11-15 * * 1-5 ${REPO_ROOT}/tools/cron_intraday_mover.sh poll >> ${REPO_ROOT}/logs/intraday_mover.log 2>&1
15 16 * * 1-5 ${REPO_ROOT}/tools/cron_intraday_mover.sh digest >> ${REPO_ROOT}/logs/intraday_mover.log 2>&1

# Weekly self-learning digest (Friday 19:30 ET)
30 19 * * 5 cd ${REPO_ROOT} && bash tools/cron_weekly_skills_review.sh >> logs/weekly_skills_review.log 2>&1

# Data auditor (weekdays 17:30 ET — after production; matches cron_data_auditor.sh)
30 17 * * 1-5 cd ${REPO_ROOT} && bash tools/cron_data_auditor.sh --daily-only >> logs/data_auditor.log 2>&1

# Agent heartbeat (weekdays 17:30 ET — fleet receipt + artifact escalation)
30 17 * * 1-5 cd ${REPO_ROOT} && python3 tools/agent_heartbeat_checks.py >> logs/heartbeat_checks.log 2>&1

# Hermes knowledge layer + contradiction detector (evening, post-screen)
0 18 * * 1-5 cd ${REPO_ROOT} && python3 tools/build_hermes_knowledge_layer.py >> logs/hermes_knowledge.log 2>&1
5 18 * * 1-5 cd ${REPO_ROOT} && python3 agents/hermes-contradiction-detector/run_job.py --from-build >> logs/hermes_contradiction.log 2>&1

# Evening catch-up safety net (weekdays 22:00 ET — WSL sleep/wake gaps)
0 22 * * 1-5 ${REPO_ROOT}/tools/cron_evening_catchup.sh >> ${REPO_ROOT}/logs/evening_catchup.log 2>&1

# Production watchdog (reboot + weekday safety net for missed cron)
@reboot ${REPO_ROOT}/tools/cron_watchdog.sh >> ${REPO_ROOT}/logs/watchdog.log 2>&1
30 12 * * 1-5 ${REPO_ROOT}/tools/cron_watchdog.sh >> ${REPO_ROOT}/logs/watchdog.log 2>&1

# Operator one-shot triage: python3 tools/fleet_ops_status.py --write
# Wiring audit: python3 tools/fleet_completion_audit.py --write
# Host checklist: bash tools/run_fleet_operator_checklist.sh

# Close F-2026-005/006 on host before enabling:
#   export SELFIMPROVE_GATES_MET=1
# Herald dark pipeline recovery:
#   bash tools/herald_recovery.sh
#   python3 tools/herald_health_check.py --recover
# in cron_weekly_skills_review.sh environment (pattern_to_skillpatch drafts).
# Rule 12 checklist: docs/governance/RULE_12_PROMOTION_CHECKLIST.md
EOF
