# Hermes Skill Harvest Log

Auto-maintained by the daily skill harvester cron job (Hermes).
Each entry records git activity reviewed, sessions searched, and skill patches applied.

---

---

---

## 2026-06-24 (agent fleet) — PR #399 fleet self-learning completion (phases 2–2g)

### Scope
Tier 0 observability/plumbing only. No ranker, selector, sizing, or `final_score` changes.

### Fleet wiring (merged on branch `cursor/agent-fleet-phase2-dfb7`)
- **Telemetry:** `log_agent_run` on all daily builders, qa/supervisor/herald/auditor, Hermes job exits
- **Outcome feedback:** `attach_outcome_verdict` on all builders + governance checks (policy_shadow overlap ≥80%, universe zero stale-alert, grok zero high alerts, etc.)
- **Herald recovery:** `tools/herald_recovery.py` / `.sh`; `herald_health_check.py --recover` (F-2026-005)
- **Evening catchup:** fully deterministic — retired all `run_agent_direct` LLM HEARTBEAT paths (Class F); ops_digest, ruleset sentinel, CRT, postmortem, supervisor_sentinel, Hermes knowledge/contradiction, catalyst_delta, price_action_watch
- **Registry:** `merged_into` for deprecated agents; `install_agent_fleet_crontab.sh` WSL reference
- **Rule 12:** `docs/governance/RULE_12_PROMOTION_CHECKLIST.md` — weekly workflow + stalled-loop gates (F-2026-005/006 block `SELFIMPROVE_GATES_MET`)

### Skill patches
- **self-improving** (`skills/self-improving/SKILL.md`): link to Rule 12 governance checklist
- **operational_health_baselines** (`skills/operational_health_baselines/SKILL.md`): Herald recovery commands + SLA table (new Hermes mirror)

### Sync
- Ran `python3 tools/sync_hermes_skills.py` (self-improving, operational_health_baselines)
- Ran `python3 tools/sync_hermes_skills.py --register-meta`
- Ran `python3 tools/audit_hermes_skills.py`

### Operator close (host — not cloud-verifiable)
- F-2026-005: run `bash tools/herald_recovery.sh` on WSL
- F-2026-006: restore GitHub Actions budget
- Crontab: `bash tools/install_agent_fleet_crontab.sh`

### Governance
- Tier 0. Efficacy back-check for this batch blocked until F-2026-005/006 RESOLVED (Rule 12).

---

## 2026-06-24 (loop review) — Trim list, contradiction gate, unified digest

### Tooling
- **skills_loop_review.py**: trim candidates (0 loads / 30d), efficacy overdue parser, contradiction check, stalled-loop parser
- **pattern_to_skillpatch.py**: pre-draft `CONTRADICTION_REVIEW` gate (F-2026-001 class)
- **hermes_skills_learning_loop_v2.py**: monthly report append trim + efficacy + stalled sections
- **weekly_skills_digest.py**: unified operator digest for Town monthly routine
- **audit_learnings.py**: `spec_lane_blocked` + `Promotion-lane` parsing

### Tests
- `tests/test_skills_loop_review.py` — 7 tests

### Governance
- Tier 0. Advisory-only — no auto-delete, no auto-merge.

---

## 2026-06-24 (Herald) — TOOLS.md fix + health check + classify path fix

### Agent docs
- **agents/herald/TOOLS.md**: replaced ops-agent copy-paste with Herald fetch/dedupe/classify/digest commands
- **agents/herald/AGENTS.md**: Herald daily sequence (was ops agent)
- **agents/herald/HEARTBEAT.md**: health check as first checklist item

### Tooling
- **herald_health_check.py**: read-only pipeline health; writes `artifacts/herald/health_check_YYYY-MM-DD.json`; exit 0/1/2
- **classify_press_releases.py**: fix output naming when input is `deduped_*.jsonl` → `classified_*.jsonl` (supervisor done predicate)
- **fetch_company_press_releases.py**: fix import order (`sys.path` before `tools.*`) — cron/supervisor no longer require `PYTHONPATH`
- **cron_data_refresh.sh**: `stage_herald` now runs dedupe before classify (matches supervisor done predicate)
- **agent_heartbeat_checks.py**: stale-source check uses `classified/classified_*.jsonl` not top-level dir sort

### Tests
- `tests/test_herald_health_check.py` — 4 tests

### Governance
- Tier 0 (agent docs + plumbing). No scoring changes.

---

## 2026-06-24 (stalled-loop + telemetry) — Operator verdicts filled; Cursor implementations

### Stalled-loop verdicts (Rule 12 efficacy gate)
- **F-2026-005 Herald**: OPEN — last repo classified JSONL 2026-02-26; host recovery unconfirmed; target 2026-07-01
- **F-2026-006 CI**: OPEN — GitHub Actions failing ~3–4s (budget pattern); target 2026-07-01
- Filled in `.learnings/memory.md` + `docs/FAILURE_PATTERN_LIBRARY.md`

### Tooling
- **skills_execution_logger.py**: environment-tagged JSONL (`execution_log_{env}_{month}.jsonl`); `record_feedback` gated
- **skills_telemetry_monthly_report.py**: CLI wrapper over `hermes_skills_learning_loop_v2`
- **pattern_to_skillpatch.py**: `infer_promotion_lane`, `refuse_spec_lane_entries`, Rule 12 lane gate logging
- **tests**: `test_pattern_to_skillpatch_lane.py`, `test_skills_execution_logger.py`

### Research
- Checklist v2 vs `final_score`: **blocked in cloud** (no snapshots/price_history) — runbook `docs/research/CHECKLIST_V2_FINAL_SCORE_BLOCKER_2026_06_24.md`

### Governance
- Tier 0 (learnings/docs/tools/tests). No ranker/selector/sizing changes.

---

## 2026-06-24 (Rule 12) — Promotion checklist canonicalized (shared Town bar)

### Skill patches
- **self-improving** (`skills/self-improving/SKILL.md`): Rule 12 promotion checklist — shared `>=3` threshold with Town (7-day behavioral / all-time failure modes); candidate feeds (Hermes LEARNINGS + failure-patterns + Town Correction Ledger); lane gate (`Promotion-lane: spec` refused); propose-only path; efficacy back-check (2-week harvest_log verification; stalled-loop F-2026-005/F-2026-006 block until RESOLVED)
- **self-improving REFERENCE**: `Promotion-lane`, `promotion_status`, Rule 12 summary, efficacy template

### Tooling
- **pattern_to_skillpatch.py**: `Promotion-lane` / `Skill-Path` parsing; spec-lane BLOCKED drafts; gate in `main()` only; efficacy note in draft output
- **audit_learnings.py**: Rule 12 promotion checklist section in output

### Sync
- Ran `python3 tools/sync_hermes_skills.py --only self_improving`
- Ran `python3 tools/audit_hermes_skills.py` — 32/32 registered, no drift
- Ran `pytest tests/test_audit_learnings.py -p no:warnings`

### Governance
- Tier 0 (skills/docs/tools). No ranker, selector, sizing, or scoring changes. F-2026-001: threshold not forked.

---

## 2026-06-24 (full apply) — All skills and learning synced to Hermes

### Learning tier updates
- **memory.md**: `cron_sys_path_isolation` promoted to HOT Ops compact
- **projects/biotech_screener.md**: Pipeline recovery Classes M–P summary table
- **agent_ops.md**: M–P pattern-key cross-ref table
- **LEARNINGS.md**: `[LRN-20260624-002]` yfinance date, `[LRN-20260624-003]` universe leak, `[LRN-20260624-004]` argparse default mask (all promoted)

### Skill patches
- **self-improving**: Rule 11 selfimprove FENCE gates (`SELFIMPROVE_IMMEDIATE_VERDICT`, `SELFIMPROVE_GATES_MET`)
- **screener-ops**: Pipeline recovery M–P table under Infrastructure
- **openclaw-cron-scheduler-debug**: Class J (cron sys.path isolation) + quick-ref triage line
- **hermeslink-state-capture**: Town operator triage cross-ref
- **operator_host_skills.md**: Full sync workflow (`audit_learnings` + `sync --register-meta`)

### Sync
- Ran `python3 tools/sync_hermes_skills.py --register-meta` (all cursor mirrors)
- Ran `python3 tools/audit_hermes_skills.py` — 32/32 registered, no drift
- Ran `python3 tools/audit_learnings.py` — tiers within limits

### Governance
- Skills/docs/learnings only. No ranker, selector, sizing, or scoring changes.

---

## 2026-06-24 (evening) — Town skills + learning applied to Hermes

### Town skills applied
- **memory-steward** (`docs/hermes_skills/memory-steward.md`): Town audit steps (memories, `town_ls skills://`, content library, people docs); Town forbidden/cleanup lists; `CLEAN_STALE_MEMORIES` decision option; Town execution rules for `delete_memory`
- **town-operator-bridge**: Operator triage table mapping `event_type` → root cause; Classes M–P cross-ref from pipeline recovery session
- **screener-ops**: Town-Hermes bridge status refreshed (2026-06-24); `cron_missed` ↔ Class P note

### Learning promoted
- **agent_ops.md**: Class P cron `sys.path` isolation pattern
- **LEARNINGS.md**: `[LRN-20260624-001] cron_sys_path_isolation` (promoted)

### Sync
- Ran `python3 tools/sync_hermes_skills.py --only screener_ops`
- Ran `python3 tools/audit_hermes_skills.py` — 32/32 registered, no drift

### Governance
- Skills/docs/learnings only. No ranker, selector, sizing, or scoring changes.

---

## 2026-06-24 (afternoon) — Pipeline recovery: XBI re-fetch, delisted universe, cache warm, cron import, selfimprove FENCE

### Patterns added to openclaw-data-pipeline-debug (Classes M–P)

- **Class M** — `datetime.isoformat()` → yfinance parse failure. XBI re-fetch crashed with
  "unconverted data remains: T00:00:00"; fix: `strftime("%Y-%m-%d")`. (399e674c)
- **Class N** — Multi-path universe leak. Delisted ticker appeared in screen after fixing one
  loader; must patch all consumers: refresh_prices, run_screen.py, run_screen_from_bundle.py,
  coverage ratio denominators. (merge 5b3225696)
- **Class O** — argparse CLI default masking function default. `--warm-sources` CLI default
  included slow registries (EUCTR/CTIS/ISRCTN), overriding the function's essential-only
  default. Production timed out 1800s on every run. (ebb33da5)
- **Class P** — Cron sys.path isolation. `from tools.*` import fails in cron (no virtualenv
  activation, no PYTHONPATH). Fix: insert PROJECT_ROOT onto sys.path before repo-relative
  imports. Fired 42× in agents.log before fix. (735ac3f7)

### Governance
- selfimprove FENCE applied: `record_feedback()` gated behind `SELFIMPROVE_IMMEDIATE_VERDICT=1`;
  `pattern_to_skillpatch.py` exits 0 unless `SELFIMPROVE_GATES_MET=1`. (735ac3f7)
- Memo: artifacts/governance/selfimprove_audit_2026-06-24.md

---

## 2026-06-24 (Shadow monitor immutability + financial periodicity)

### Git activity (past 24h)
- **biotech-screener** (51 commits):
  - `18e7cc75` feat: selfimprove Steps 3-4 + Nous Gateway research tools
  - `9b0d1fe8` chore: manager registry + CUSIP static map cleanup
  - `679d8f6e` sci-cart: mechanism alias pack v0.1 T2D drug classes + normalizer tests
  - `c6e1700c` fix(m2-v2): correct 4× burn overstatement in NetIncome + R&D fallback paths
  - `501fd56d` sci-cart map UX v0.3: visual QA audit memo
  - `b8f4c66b` audit(sci-cart): v0.3 scope hygiene check
  - `aa9c8512` audit: full calculation audit 2026-06-23 + forward shadow checkpoints
  - `5e1eaf45` Scientific Cartography Map UX v0.3 — poster-style layout
  - `20da2dd0` feat(selfimprove): wire immediate verdict in run_agent_direct (Step 2)
  - `614561e7` sci-cart mechanism alias pack v0.1: T2D drug classes
  - `49d78b00` feat(selfimprove): wire skill_exec_id capture in run_agent_direct
  - `e8f1d475` chore(claude): add governance reviewer, sci-cart skill, Claude config + memos
  - `379499d3` sci-cart map UX v0.2d: D3 expansion + asset-name canonicalization
  - `e5052aa8` feat(sci-cart): Map UX v0.2c — D1 dedup + D3 non-drug filter
  - `91eec238` docs(sci-cart): Map UX v0.2b visual QA memo
  - `544da12d` feat(sci-cart): Map UX v0.2b — static disease-map generator
  - `40a621f2` sci-cart Phase 13.7: add CT.gov uppercase/underscore stage aliases
  - `dc1aaed6` fix(sci-cart): Phase 13.5 R2b — stage parser singular phase field compatibility
  - `597f3aa7` sci-cart Phase 13.6 R6: mechanism coverage design memo (DESIGN_ONLY)
  - `f6c78fcc` sci-cart Phase 13.4 R5: wire therapeutic_area from MONDO disease normalizer
  - `c6f68861` sci-cart Phase 13.3 R3: fix confidence collapse for unresolved asset aliases
  - `fb244e43` audit(sci-cart): Phase 13.2 R4 normalization sample review — PASS proceed to R3
  - `c7cc77c1` audit(sci-cart): Scientific Cartography map UX repo audit + RA-style design spec
  - `9f3e0f6b` fix(data): refresh market data + add run skill + EES log entries
  - `f4a32df2` fix(sci-cart): Phase 13.1 R2 — add trial_records.json to input discovery order
  - `96733236` design(ees): Phase 3 CT_PRIMARY_COMPLETION left-tail guardrail design memo
  - `fb52071f` audit(ees): EES v2 Phase 3 attribution review — CT_PRIMARY_COMPLETION left-tail avoidance signal
  - `376d9e9d` fix(shadow): harden settled-row immutability against non-boolean truthy forms
  - `60876b11` feat(shadow): EES v2 Phase 3 shadow monitor — diagnostic-only, append-only ledger
  - `c35fc1ba` spec(ees): EES v2 Phase 3 shadow monitor design spec
  - `e80c3ff2` audit(ees): EES forward validation — PASS diagnostic, Phase 3 concentrated signal
  - `35e662f8` data(herald): replace common-word ticker keywords with company-name Globe searches
  - `0d41dbd3` data(herald): fill IR URLs for 38 tickers with no press-release source
  - `692eded0` audit(pit): PIT gap forward return evidence review — PASS diagnostic
  - `83724549` docs(model): document Semgrep, LangGraph, and CodeGraph developer tooling
  - `7255cf48` Merge branch 'main' of github.com:Warrenpoobear/biotech-screener
  - `f55942cf` research(pit): fresh PIT gap forward-return assembly (Method A + B) (#389)
  - `cef457d3` docs(sci-cart): Phase 13 remediation plan — R2→R4→R3→R5→R6 sequence
  - `6488ddf0` Merge branch 'main' of github.com:Warrenpoobear/biotech-screener
  - `b0d70950` docs(audit): Semgrep MCP post-registration audit memo
  - `d5f15a0b` diag(event-ev): Event EV shadow diagnostic — calibration analysis only (#388)
  - `7afbd1db` docs(sci-cart): Phase 12.1 Disease Map Operational Review — 2026-06-23
  - `934b5389` feat(mcp): register Semgrep MCP server for governance scanning
  - `4592fb6f` fix(event-ev): correct EES status labels — PREDICTIVE_STATUS: UNPROVEN
  - `c7406c48` feat(event-ev): Event EV shadow diagnostic — market-implied vs base rate
  - `8555ef25` docs(governance): expectation layer field coverage verification — PASS
  - `f96ba4d0` data: expression decision log — 2026-06-23 cron run
  - `f90a1665` feat(mcp): add read-only scientific cartography tools (get_atlas_data)
  - `38edb0ab` fix(sci-cart): tighten disease normalizer synonym matching + ingester name column
  - `5a69fcd1` research: assemble PIT gap forward returns (Jan 16 - May 7, 2026) (#382)
- **asset-allocation** (0 commits)

### Sessions reviewed: 4
- cron_a15dbdcb6f41_20260624_082230 (weekly-skill-harvester): prior harvest run
- cron_4013ddd98c6d_20260624_082703 (inst_delta_z governance monitor): routine check
- cron_7e79501afb6e_20260624_084501 (weekly-signal-regime-sweep): routine check
- cron_a15dbdcb6f41_20260624_083822 (aa-model-tracker): no AA activity

### Skill patches
- **openclaw-data-pipeline-debug**: Added Class K (shadow monitor truthy-form immutability — post-merge audit pattern). Confirmed 2026-06-23 commit `376d9e9d`: `forward_complete_20d is True` identity check missed manually edited values like `1` or `"true"`. Added `_is_settled()` helper pattern accepting all truthy forms. +17 tests (54 total).
- **openclaw-data-pipeline-debug**: Added Class L (financial calculation unit-mismatch — periodicity confusion). Confirmed 2026-06-23 commit `c6e1700c`: Module 2 fallback paths hardcoded `/3` (quarterly) for annual data (should be `/12`). Burn rate overstated 4×, runway understated 4×. Fix uses `_ytd_months_from_date()` with `NetIncome_date` / `R&D_date` fields. +2 golden tests.

### New skills created: none

### Notable: Scientific Cartography expansion
- 15+ commits advancing sci-cart phases 12.1–13.7 (disease map UX, mechanism alias pack, normalizer hardening)
- Phase 13 adds MONDO disease normalizer integration for therapeutic_area
- Mechanism alias pack v0.1: T2D drug classes (first domain-specific alias normalization)
- Map UX v0.2b→v0.3: static disease-map generator with poster-style layout
- All diagnostic-only, no production scoring impact

### Notable: Selfimprove loop wiring
- `18e7cc75` + `20da2dd0` + `49d78b00`: selfimprove Steps 3-4 wired into `run_agent_direct.py`
- `tools/record_skill_feedback.py`: deferred ground-truth reward signal
- `tools/pattern_to_skillpatch.py`: auto-drafts skill promotions from LEARNINGS.md patterns
- `tools/research_ticker.sh` + `tools/research_landscape.sh`: Nous Gateway research tools (read-only)

### Notable: EES v2 Phase 3 shadow monitor
- `60876b11` + `376d9e9d`: diagnostic-only append-only ledger for EES v2 Phase 3
- Requires `--as-of-date`, no cron, no scheduler
- Settled rows immutable (truthy-form guard added post-merge)
- Observation gate: 20 completed 5d + 20d observations before IC computation
- 54 tests passing

### Governance
- Docs/skills/research tools only. No selector, ranker, sizing, final score, decision engine, production KG, cron, or runtime scoring changes.
- Class L fix (`c6e1700c`) is a bug fix to Module 2 financial calculations — affects burn/runway for tickers reaching fallback paths with annual filings. Not a governance event (silent bug, no behavior change intended).

---

## 2026-06-19 (Hermes context refresh + registry tombstone split)

### Follow-up contract fixes (2026-06-20 UTC)
- **Lane A jobs:** Added contract tests for builder-output schemas and fixed `hermes-held-spec-ledger` to read `items` from `held_spec_ledger/latest.json`.
- **First-fire routing:** `hermes-first-fire-validator` now treats builder `FAIL_*` eval statuses as failures and routes `first_fire_fail`.
- **Registry lint:** Added `suppressed` as an explicit status and made directory-less registry entries legal only for fully marked deprecated tombstones.
- **Direct dispatch guard:** `run_agent_direct.py` blocks `suppressed` agents as well as deprecated/shadow agents.
- **Lane A heartbeat contract:** Removed stub `HEARTBEAT.md` files from deterministic `hermes-*` jobs; heartbeat checks skip these on-demand jobs by design.
- **MCP guardrail:** Docs now explicitly forbid substituting upstream write-capable `hermes mcp serve` for the repo-native read-only Cursor MCP.

### Governance
- Hermes docs/tooling/Lane A plumbing only. No selector, ranker, sizing, final score, decision engine, production KG, cron, or runtime scoring changes.

---

### Hermes sync/audit
- Ran `python3 tools/audit_hermes_skills.py` and found the dated skills audit report was being counted as an unregistered skill doc.
- Updated `tools/audit_hermes_skills.py` so `SKILLS_AUDIT_*.md` reports remain audit artifacts, not skill registry entries.
- Synced `screener-ops` through `tools/sync_hermes_skills.py --only screener_ops`; cursor-synced mirrors remain source-driven.

### Registry and runtime authority
- **agent_roster.md / agent-registry-reference.md / hermes_tools_map.md / hermeslink-state-capture.md:** refreshed counts to registry `as_of` 2026-06-19: 34 entries = 29 active, 1 suppressed, 4 deprecated; 31 directories on disk.
- **Known registry invariant gap:** `biotech_news_digest`, `company_news_ingest`, and `policy_shadow_watch` are deprecated registry tombstones without directories. This is an operator-owned registry/directory reconciliation, not a scoring or runtime change.
- **Cloud limitation:** live Hermes CLI/OpenClaw scheduler state remains operator-WSL-only; repo docs do not assert current job IDs or last-run status beyond the existing 2026-05-05 snapshot.

### Governance
- Docs/tooling only. No selector, ranker, sizing, final score, decision engine, production KG, cron, or runtime scoring changes.

---

## 2026-06-17 (Fleet monitoring blind spot + Scientific Cartography + memory-write bugs)

### Git activity (past 24h)
- **biotech-screener** (13 commits):
  - Scientific Cartography Layer phases 0/1–7B (11 commits): new `scientific_cartography/` module with competitive clustering, landscape features, export layer, diagnostic pipeline. Phase 7B adds optional disabled-by-default diagnostic hook to `tools/run_daily_production.py` (non-blocking, controlled via `--run-scientific-cartography` flag).
  - Fix: MapIndexExporter handles None disease_name in sort (0058b940)
  - Day 7 pre-market monitoring snapshot (b2c4bb1c)
  - PR #350 merge (61c70db7)
- **asset-allocation** (1 commit):
  - docs(tracking): MODE A sync 2026-06-17 — fix self-ref, stale gov flag escalation (32b2ab3)

### Sessions reviewed: 3
- cron_4f360d005436 (fleet triage 2026-06-17): FLEET VERDICT RED — receipt 5d stale, sentinel ROLLBACK_RECOMMENDED, drift pipeline broken, multiple agent STALE
- cron_876bb90e5295 (memory steward 2026-06-17): ~/.hermes/ grew to 3.4 GB, 2 Hermes cron jobs with stale errors (weekly-skill-harvester 30d, morning-briefing 24d)
- cron_3d1e09988873 (aa-model tracker 2026-06-17): routine MODE A sync, no new findings

### Skill patches
- **openclaw-cron-scheduler-debug**: Added Class H (heartbeat checker stall — monitor-of-monitors blind spot) and Class I (Hermes cron job silent failure — NameError/RuntimeError unnoticed for weeks). Updated quick-reference triage with both new classes. Confirmed instance: fleet receipt 5d stale (2026-06-17), 2 Hermes cron jobs failing 24-30d silently.
- **openclaw-agent-scope-audit**: Added Class H (cron task ruleset ID mismatch — cron parameter layer, not SOUL.md). SOUL.md correct but cron task passes stale ruleset ID. Confirmed instance: sentinel cron task passing 2a3e79eb while SOUL.md correctly shows 8887576e (2026-06-17, Day 9+). Also added bioshort_watch and qa to Class C confirmed memory-write bug instances.
- **openclaw-data-pipeline-debug**: Added Class J (drift report pipeline broken — sentinel blinded on overlap/rank-shift). drift_guardrails/ missing from snapshots 5+ days, ruleset_health_history.jsonl stale 6+ weeks. Updated cross-skill routing section.

### New skills created: none
### No new findings skipped: all actionable findings patched

### Notable: Scientific Cartography Layer
- New `scientific_cartography/` module built across 11 commits (phases 0/1 through 7B)
- Phase 7B integrates into `tools/run_daily_production.py` as optional diagnostic hook
- Disabled by default (`--run-scientific-cartography` flag required)
- Non-blocking: failures logged but do not halt pipeline
- No governance concern: diagnostic-only, does not modify production scoring or rankings

### Notable: Hermes cron job health gap
- weekly-skill-harvester (a15dbdcb6f41): NameError, 30d stale — this is the job that runs THIS harvest
- morning-briefing (a955f533907b): RuntimeError, 24d stale
- No monitoring exists for Hermes cron job failures — they fail silently until manually discovered
- Recommended: add Hermes cron job health check to fleet steward heartbeat

---

## 2026-06-16 (Hermes registry + Cloud staleness refresh)

### Hermes Agent v0.16.0 update check
- Public release metadata confirms Hermes Agent v0.16.0 / v2026.6.5 ("The Surface Release") is available upstream.
- Cloud checkout has no `hermes` CLI or `~/.hermes/hermes-agent`; live version/update remains operator-WSL authority.
- Added operator-safe checklist to `docs/HERMES_GATEWAY_SETUP.md`: `hermes update --check`, backup `~/.hermes/config.yaml` + `~/.hermes/skills/`, then run update/config/doctor/gateway checks.
- Added v0.16.0 caution for first config persistence possibly rewriting `config.yaml` and dropping hand-curated provider blocks; compare against backup before gateway restart.

### Agent profiles / skills / memory follow-up
- **screener-ops:** registry profile line updated to 31 directories = 29 active + 2 deprecated; mirror synced.
- **memory layer:** HOT/WARM learnings updated for CodeGraph v0.9.9 and registry bidirectional invariant.
- **memory-steward:** deployment/profile guidance reconciled across `docs/hermes_skills/memory-steward.md` and `.hermes/skills/devops/memory-steward.SKILL.md`.
- Ran `python3 tools/audit_learnings.py` — no stale hints or compaction issues.

### Hermes sync/audit
- Ran `python3 tools/sync_hermes_skills.py --register-meta` and `python3 tools/sync_hermes_skills.py` — 32 registered Hermes docs, 19 cursor mirrors unchanged/skipped except `screener-ops` after CodeGraph pin refresh.
- Ran `python3 tools/audit_hermes_skills.py` — all Hermes `.md` files registered, source_authority complete, no mirror drift.

### Registry and runtime authority
- **agent_roster.md / agent-registry-reference.md:** registry-aligned counts now show 31 directories = 29 active + 2 deprecated (`bioshort_watch`, `shadow_watch`).
- **openclaw-session-routing-debug.md:** retired gateway-zombie guidance now handles retained `status=deprecated` directories.
- **HERMES_OPERATIONAL_PROFILE.md:** distinguishes repo audit counts (32 skill docs, 29 active agents + 2 deprecated workspaces) from conceptual operator-host layers.
- **screener-ops:** CodeGraph pin refreshed to v0.9.9 to match `.cursor/environment.json` and `docs/CODEGRAPH_RUNBOOK.md`.

### Governance
- Docs/metadata/heartbeat plumbing only. No selector, ranker, sizing, final score, decision engine, KG, cron, or runtime scoring changes.

---

## 2026-06-07 (Hermes taxonomy + learning loop refresh)

### Hermes-native docs
- **hermes_tools_map.md**: baseline `ec4b2726`; `audit_learnings`, skills telemetry + learning loop v2; 32-skill audit expectation; WSL sequence updated
- **hermeslink-state-capture.md**: fleet count fixed (29 active); skills learning tools; PR #322 merged state
- **agent_roster.md**: date refresh; registry count aligned
- **operator_host_skills.md**: 32/32 audit; skills learning telemetry index

### Skill mirror
- **screener-ops**: baseline `ec4b2726`; checklist 32/32

### Governance
- Docs/tooling only. No ranker, selector, sizing, or scoring changes.

---

## 2026-06-07 (repo scale + skills refresh)

### Skill sources refreshed
- **screener-ops**: Repo scale orientation table (~4k engines / ~176k production / ~750k total); baseline `main` @ `31a74d42`; `audit_learnings.py` in WSL gate Phase 1 + sync workflow
- **selector-ranker**: Codebase context — compact high-leverage engines; governance tier unchanged
- **codegraph**: baseline `31a74d42`, index file count note
- **self-improving**: Distill step → memory bootstrap block
- Ran `sync_hermes_skills.py` — updated mirrors

### Governance
- Skills/docs only. No ranker, selector, sizing, or scoring changes.

---

## 2026-06-01 (HOT memory optimization)

### Memory layer
- **`memory.md`**: bootstrap-first table (recursion, CodeGraph, host, governance); deduped ops; Pattern-Key tags
- **`archive/README.md`**: COLD tier demotion policy
- **LRN-20260329-001**: Status → promoted (raw_count_size_confound already in HOT)
- **`audit_learnings.py`**: bootstrap line count, HOT Pattern-Keys, LRN/HOT mismatch, compaction hints
- **README.md** + **self-improving** Rule 6: memory optimization principles

### Governance
- Memory/docs/tooling only.

---

## 2026-06-01 (knowledge recursion stack)

### Knowledge layer
- **`.learnings/README.md`**: agent knowledge stack map, load order, promotion rules
- **`.learnings/domains/agent_ops.md`**: WARM Hermes/Cursor/Cloud authority patterns
- **`tools/audit_learnings.py`**: read-only tier audit, Pattern-Key promotion candidates, stale hints
- **`tests/test_audit_learnings.py`**: parser + report smoke
- Updated `memory.md`, `LEARNINGS.md` (LRN-20260601-001..003), `projects/biotech_screener.md`
- **self-improving** skill/REFERENCE: `audit_learnings.py` in session-end workflow

### Governance
- Knowledge/docs/tooling only. No ranker, selector, sizing, or scoring changes.

---

## 2026-06-01 (recursive self-improvement loop)

### Skill sources refreshed
- **self-improving** (`skills/self-improving/SKILL.md`): biotech-screener recursive loop (Observe→Log→Distill→Promote→Skill-patch→Sync→Verify); Rule 10 governance; session-end trigger
- **self-improving-reference** (`skills/self-improving/REFERENCE.md`): LRN/harvest templates, session-end audit, skill-patch checklist
- **screener_ops**, **codegraph**, **openclaw-agent-optimize**: cross-links to recursion loop
- **operator_host_skills.md**: index row for self-improving + REFERENCE mirror
- **sync_hermes_skills.py**: register `self-improving-reference.md` in REFERENCE_MAP
- `.learnings/projects/biotech_screener.md`: skill recursion meta + codegraph 0.9.7 note
- Ran `sync_hermes_skills.py` — updated mirrors

### Governance
- Skills/docs only. No ranker, selector, sizing, or scoring changes.

---

## 2026-06-01 (WSL acceptance gate + CodeGraph bounds)

### Skill sources refreshed
- **screener-ops** (`skills/screener_ops/SKILL.md`): full operator WSL acceptance gate (Phases 0–4); gateway model check; Spec 087 B1b cron/artifact criteria; first-fire seed note; printable checklist; baseline `main` @ `0bac216a` (#332–#334)
- **codegraph** (`skills/codegraph/SKILL.md`): surface split table; four-step operating rule; practical CLI examples; baseline `0bac216a`
- **operator_host_skills.md**: WSL acceptance gate index row
- Ran `sync_hermes_skills.py` — updated `screener-ops.md`, `codegraph.md`

### Governance
- Skills/docs only. No ranker, selector, sizing, or scoring changes.

---

## 2026-05-31 (Cursor skills knowledge)

### Skill sources refreshed
- **screener-ops** (`skills/screener_ops/SKILL.md`): Hermes model routing table (MCP / Lane A / gateway / SOUL / `run_agent_direct.py`); plumbing baseline `main` @ `8dbd1b9c` (#326–#331); Cursor sync workflow table
- **codegraph** (`skills/codegraph/SKILL.md`): baseline `8dbd1b9c`; pointer to `hermes_tools_map.md`
- **operator_host_skills.md**: Cursor skills knowledge index (sync, audit, model doc links)
- Ran `sync_hermes_skills.py` — updated `screener-ops.md`, `codegraph.md`

### Governance
- Skills/docs only. No ranker, selector, sizing, or scoring changes.

---

## 2026-05-30

### Sync (Cursor Cloud agent)
- Ran `python3 tools/audit_hermes_skills.py` — 31/31 Hermes docs registered in `_meta.json`
- Ran `python3 tools/sync_hermes_skills.py --register-meta` — 19 cursor mirrors unchanged; `memory-steward` skipped (Hermes-authoritative)
- **screener-ops**: Town-Hermes bridge status refreshed (Phase B wiring complete 2026-05-27; live delivery pending)
- **town-operator-bridge**: Phase B call-site table updated (4 DONE, 2 TODO)
- **operational-state.md**: Hermes Skills Hub sync state block updated

### Governance
- Tooling/docs only. No ranker, selector, sizing, or scoring changes.

---

## 2026-05-31 (Hermeslink)

### hermeslink-state-capture skill refresh
- Updated `docs/hermes_skills/hermeslink-state-capture.md` for 2026-05-31: host authority table, Phase B Town egress, full run cycle, 34-agent fleet, separation from `build_knowledge_graph.py`
- Ran `build_hermes_knowledge_layer.py` on Cloud (UNKNOWN_CLOUD_ENV; 0 hard contradictions)

---

## 2026-05-31 (Hermes update)

### Code fixes (PR #322)
- Fixed `build_hermes_knowledge_layer.py` repo `sys.path` for `town_bridge_events` import
- Fixed MCP `knowledge_read(contradiction_ledger)` to prefer `latest.md`
- MCP regression test + harvest log

---

## 2026-05-30 (OpenClaw doc refresh)

### OpenClaw / Hermes taxonomy
- **hermes_tools_map.md:** §5 OpenClaw gateway (Lanes A/B/C, `run_openclaw.sh`, debug skills, post-#326 cleanup)
- **openclaw-session-routing-debug.md:** 29-agent fleet; Class G updated for repo-removed agents
- **openclaw-agent-scope-audit.md:** `policy_shadow_watch` marked resolved → `shadow_monitor`
- **hermeslink-state-capture.md:** agent counts 29 active (registry-aligned)
- **agent_roster.md:** debug skills line 29-agent; tools map cross-link

### Agent & bridge gaps addressed (earlier same day)
- **agent_roster.md:** Split repo fleet (29 agents after #326, `AGENT_REGISTRY.json`) vs Hermes scheduler jobs
- **Town-Hermes Phase B:** `common/town_bridge_events.py`; `cron_missed` from `ops_supervisor` + `cron_watchdog`; `contradiction_detected` from knowledge layer + `hermes-contradiction-detector`
- **Registry:** Added `hermes-contradiction-detector` (31 active agents)
- **Tests:** `tests/test_town_bridge_events.py`
- **Docs:** `town-operator-bridge.md` operator live-email checklist; `hermes-context.mdc` fleet count

### Governance
- Plumbing/ops only. No ranker, selector, sizing, or scoring changes.

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
