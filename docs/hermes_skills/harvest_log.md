# Hermes Skill Harvest Log

Auto-maintained by the daily skill harvester cron job (Hermes).
Each entry records git activity reviewed, sessions searched, and skill patches applied.

---

## 2026-05-24

### Git activity (manual update)
- **biotech-screener**: Hermes MCP startup fix reviewed
  - `7c04e6a9` Fix Hermes MCP startup
  - `58954ad0` Normalize Hermes MCP config key

### Sessions reviewed
- Cursor Cloud session diagnosing Hermes MCP discovery failure.

### Skill patches
- **cursor-hermes-mcp-debug**: New skill documenting repo-native Hermes MCP startup,
  expected read-only tools, direct JSON-RPC smoke tests, Cursor reload requirement,
  and common failure classes.
- **screener_ops**: Added Cursor Hermes MCP quick-reference section with config,
  expected tools, reload guidance, and detailed runbook link.

### New skills created
- `cursor-hermes-mcp-debug.md`

### Files synced to docs/hermes_skills/
- cursor-hermes-mcp-debug.md

---

## 2026-05-06

### Git activity (past 24h)
- **biotech-screener**: 30 commits reviewed
  - `8aa6d5f0` docs(specs): add investment logic audit follow-up backlog (specs 078-082)
  - `67efd1b2` Add hermes-operator Claude Code subagent for job/scheduler management
  - `ee1c9970` chore: disk cleanup batch 1+2A — remove stale artifacts, orphaned worktree, safe cache
  - `fee9b10e` Backup Hermes memory-steward skill to repo (.hermes/skills/)
  - `5c284ab7` fix: correct Morningstar scores_by_ticker inner key (second ms_* bug)
  - `dd002964` Add memory-steward Claude Code subagent for audit-gated cleanup
  - `ff4b7c64` chore: remove catalyst_source_filed_at dead schema field
  - `e70ae626` fix: restore Morningstar enrichment field mapping
  - `feba3a64` docs(audit): spec_076 schema prune audit — audit only, no removals
  - `7f02f79e` docs: spec_074 financial_score evidence memo + spec_075 inst_delta checkpoint
  - `df5a59f0` feat(ev): bind event_ev_p_hit into resolution records (shadow-only)
  - `d97a5cc7` ops(heartbeat): fix production_qa false-alarm STALE at 17:30 ET
  - `70049dd0` ops(heartbeat): fix review_queue_steward + policy_shadow_watch receipt noise
  - `9f3b91ea` ops(agents): fix memory-write for catalyst_delta, event_analyst, shadow_monitor
  - `7ac2ceee` ops(sentinel): re-anchor to v1.14.0 (8887576e)
  - (+ 15 more: populate_ir_sources, price data audit, financial data fixes, EV docs, etc.)
- **asset-allocation**: 1 commit
  - `5977b19` docs(tracking): MODE A sync 2026-05-05 — 386 tests, 1 ruff error, governance flag on 3 fix() commits

### Sessions reviewed: 8
- `20260506_081523_d685ae` — IR URL population scoping (Option 3 EDGAR-sourced recommended)
- `20260506_071508_448d23` — Fleet triage + policy_shadow backfill + grok email credential incident
- `20260505_214736_9bfa43` — Mode B audit: score_rank_pct WARN Day 2, sentinel misanchored
- `20260505_131955_90c115` — Class C + G confirmed for openclaw-cron-scheduler-debug
- `cron_4f360d005436_20260505_180021` — Fleet triage: memory-write watchdog run, v1.14.0 stability spike diagnosed
- `20260504_143458_159c55` — Fleet triage: 2026-05-02 production miss root-caused (crontab REPLACE)
- `cron_4f360d005436_20260503_024000` — Fleet triage: WSL2 sleep + 8 OAuth expired profiles
- `20260503_110735_3d31d4` — inst_delta_z governance memo, coinvest shared-regime check

### Skill patches
- **openclaw-data-pipeline-debug**: Class I added — Morningstar silent double-bug (`run_screen.py` wrong inner key `.get("scores")` vs `.get("scores_by_ticker")`). Confirmed 2026-05-06 commits `e70ae626` + `5c284ab7`. Both bugs independent + both silent.
- **openclaw-data-pipeline-debug**: Class G extended — score_rank_pct WARN Day 3+ escalation protocol added. Confirmed structural degradation from mid-February; escalation packet template with GOVERNANCE CEILING rule added. ic_health_monitor cron dependency noted (no standalone cron).
- **openclaw-data-pipeline-debug**: Routing table updated with Class I + WARN signal escalation path.
- **memory-steward**: Orphaned `.claude/worktrees/` added to known bloat patterns. Confirmed 235 MB reclaimed 2026-05-06 (`ee1c9970`). Audit step 7 updated with worktree orphan detection commands.
- **openclaw-agent-scope-audit**: Class G added — dead schema field / spec-076 pruning pattern. `catalyst_source_filed_at` as confirmed instance (commit `ff4b7c64`). Includes diagnostic recipe + safe-to-cut classification rules.

### New skills created: none

### Files synced to docs/hermes_skills/
- openclaw-data-pipeline-debug.md (patched)
- memory-steward.md (patched)
- openclaw-agent-scope-audit.md (patched)
- references/ir_url_population.md (from ~/.hermes/skills/devops/openclaw-data-pipeline-debug/references/)
