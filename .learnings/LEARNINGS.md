# Learnings Log

<!-- Self-improving agent learning entries. Format: [LRN-YYYYMMDD-XXX] -->
<!-- Entries logged during conversations; promoted to CLAUDE.md / memory when verified. -->

## [LRN-20260329-001] size_confound_in_raw_event_counts

**Logged**: 2026-03-29T19:00:00Z
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Raw event counts (graveyard burden, catalyst density, neg_reg count) all correlate positively with forward returns — capturing "well-covered large company" not genuine alpha.

### Details
Both graveyard burden (IC=+0.046 at 63d) and catalyst history features (event_density IC=+0.141, neg_reg IC=+0.052) showed wrong-sign results. The same size/coverage confound appeared in PI trial count earlier. Any count-based feature needs size-orthogonal decomposition before promotion.

### Suggested Action
Residualize all count-based research features against pipeline size (n_total_trials) or market_cap_bucket before testing for signal. Do not promote raw count features.

### Metadata
- Source: research_backtest
- Related Files: scripts/research/backtest_graveyard_signal.py, scripts/research/backtest_catalyst_history_signal.py
- Tags: size_confound, research, signal_quality
- Pattern-Key: raw_count_size_confound
- Recurrence-Count: 3

## [LRN-20260329-002] portfolio_drag_is_construction_not_ranking

**Logged**: 2026-03-29T20:00:00Z
**Priority**: critical
**Status**: promoted
**Area**: portfolio

### Summary
Shadow portfolio drag comes from flat 3% weights on C-tier names and no exit rule for persistent headwind+drawdown, not from ranking model defects.

### Details
C-tier P&L/weight-day: -0.78% (2x worse than A-tier). Headwind bleed: 2.3x non-headwind. Tier-weighted policy (A=4/B=2.5/C=1/D=0) improved returns by +1.60pp over 18 days. Adding headwind+drawdown exit improved by +1.82pp. SLN and EYPT were held as C-tier headwind names at full weight — not rebalance lag, but policy design.

### Suggested Action
Run tier-weighted policy as formal shadow candidate (Spec 035). Already implemented and wired into run_screen.py. Current verdict: PROMISING (4/4 gates pass).

### Metadata
- Source: shadow_attribution_audit
- Related Files: tools/build_policy_shadow_compare.py, tools/eval_policy_candidate.py, specs/changes/035_tier_weighted_policy.md
- Tags: portfolio_construction, policy, shadow_candidate
- Skill-Path: Spec 035

## [LRN-20260329-003] open_targets_api_search_type_mismatch

**Logged**: 2026-03-29T17:00:00Z
**Priority**: medium
**Status**: resolved
**Area**: data_pipeline

### Summary
Open Targets GraphQL search endpoint returns generic SearchResult types — inline fragments like `... on Drug` or `... on Disease` are silently ignored. Must use two-step: search → get ID → fetch details by ID.

### Details
This bug caused zero results from the Open Targets enrichment tool. The fix required both a two-step query pattern AND correcting the field name `maxPhaseForIndication` → `maxClinicalStage`. After fix: 218/309 tickers enriched with 14,833 disease associations.

### Suggested Action
Always use two-step query pattern for Open Targets API. The search endpoint does not support type-specific inline fragments.

### Metadata
- Source: bug_fix
- Related Files: tools/enrich_open_targets.py, tools/enrich_indication_master.py
- Tags: api, open_targets, graphql, enrichment
- Pattern-Key: ot_search_type_mismatch

## [LRN-20260330-005] sec_8k_collapse_gate_works

**Logged**: 2026-03-30T13:00:00Z
**Priority**: medium
**Status**: resolved
**Area**: data_pipeline

### Summary
SEC 8-K cache refresh gate correctly rejected corrupted fetch (117 vs prior 473, ratio 0.25 < 0.3 threshold) when EDGAR returned 500 errors on several search queries. Prior cache preserved automatically.

### Details
CCFT data quality infrastructure validated in production. The collapse ratio gate in warm_caches.py detected the bad fetch and kept good data. No manual intervention needed.

### Suggested Action
No action needed — gate worked as designed. Good validation example of cache resilience.

### Metadata
- Source: production_ops
- Related Files: warm_caches.py
- Tags: data_quality, validation, positive_example

## [LRN-20260329-004] f_string_no_placeholder_flake8

**Logged**: 2026-03-29T18:00:00Z
**Priority**: low
**Status**: resolved
**Area**: code_quality

### Summary
Flake8 F541 fires on f-strings used for markdown table headers with no interpolation. Use plain strings for static table headers.

### Details
Recurred across bioshort_watch, catalyst_history_diagnostics, eval_policy_candidate, build_policy_shadow_compare. Every new markdown formatter hits this.

### Suggested Action
Use plain strings for static markdown table headers, f-strings only when interpolating values.

### Metadata
- Source: pre_commit_hook
- Tags: flake8, code_style
- Pattern-Key: f_string_no_placeholder
- Recurrence-Count: 5

## [LRN-20260525-001] cursor_cloud_runtime_deps

**Logged**: 2026-05-25T22:30:00Z
**Priority**: medium
**Status**: resolved
**Area**: cloud_environment

### Summary
Fresh Cursor Cloud agents may have CodeGraph but not the Python runtime/test dependencies needed by `run_screen.py` and pytest.

### Details
Manual Friday screen check initially failed on `ModuleNotFoundError: No module named 'dotenv'`. Installing `requirements.txt` fixed `run_screen.py`; installing `pytest-xdist` was also required while main still has pytest addopts `-n auto --dist worksteal`. Environment setup should install Python deps before CodeGraph checks so agents can run screens/tests without ad hoc pip installs.

### Suggested Action
Keep `.cursor/environment.json` idempotent and include `python3 -m pip install --user -r requirements.txt pytest-xdist` before CodeGraph initialization until pytest addopts no longer require xdist.

### Metadata
- Source: cursor_cloud_run_screen_check
- Related Files: .cursor/environment.json, requirements.txt, pyproject.toml
- Tags: cursor_cloud, environment, pytest, run_screen
- Pattern-Key: cursor_cloud_missing_python_deps

## [LRN-20260525-002] ci_actions_budget_not_code_failure

**Logged**: 2026-05-25T18:40:00Z
**Priority**: medium
**Status**: resolved
**Area**: ci

### Summary
GitHub Actions failures with annotation "The job was not started because an Actions budget is preventing further use" are provider budget/quota failures, not repository regressions.

### Details
PR #300 showed 15 failing checks across smoke, replay, lint, type-check, secret-scan, dep-audit, and pytest. Every check completed in seconds with the same provider annotation before any job steps started. No tests, linters, scans, or workflow commands executed.

### Suggested Action
Do not patch PR code for this signal. Restore/wait for Actions budget availability, then rerun CI.

### Metadata
- Source: ci_investigation_pr_300
- Tags: ci, github_actions, budget, quota
- Pattern-Key: actions_budget_pre_start_failure

## [LRN-20260528-004] codegraph_hermes_guard_implemented

**Logged**: 2026-05-28T03:48:00Z
**Priority**: high
**Status**: resolved
**Area**: tooling

### Summary
All five Hermes acceptance gates for codegraph are now implemented in `common/codegraph_guard.py`. Hermes agents should use `CodegraphGuard` instead of calling the codegraph CLI directly. Runbook updated: Hermes registration is no longer deferred.

### Details
- Gate 1: Dynamic-dispatch break → warns, instructs fallback, never halts silently
- Gate 2: Ambiguous symbol → warns, requires `file_hint` before proceeding
- Gate 3: File-path literal → returns UNVERIFIED + grep fallback instruction
- Gate 4: Cron/shell boundary → warns to inspect subprocess/crontab manually
- Gate 5: Partial proof → emits `[PARTIAL PROOF]` on empty results or dynamic-dispatch gaps
- `tier3_gate(symbol)` → convenience method for blast-radius governance check
- 21 unit tests in `tests/test_codegraph_guard.py`, all passing
- CODEGRAPH_RUNBOOK.md Hermes section updated from "deferred" to "registered"

### Suggested Action
Hermes agents that need structural code analysis should `from common.codegraph_guard import CodegraphGuard` and use `cg.callers()`, `cg.impact()`, `cg.tier3_gate()` etc.

### Metadata
- Source: codegraph_hermes_integration_2026-05-28
- Related Files: common/codegraph_guard.py, tests/test_codegraph_guard.py, docs/CODEGRAPH_RUNBOOK.md
- Tags: codegraph, hermes, governance, acceptance_gates
- Pattern-Key: codegraph_hermes_guard

## [LRN-20260528-005] codegraph_snapshot_columns_contract_already_exists

**Logged**: 2026-05-28T03:48:00Z
**Priority**: low
**Status**: resolved
**Area**: tooling

### Summary
CODEGRAPH_RUNBOOK.md called SNAPSHOT_COLUMNS drift "the next recommended codegraph application" but Contract 6 in `tests/test_contract_output_schemas.py` already covers it (added 2026-05-24). Runbook updated to note this.

### Details
Contract 6 enforces: (6a) bundle is strict subset of live, (6b) shared parity columns, (6c) live retains scoring/expectation columns. All 3 sub-tests pass. The runbook note was just stale.

### Metadata
- Source: codegraph_skill_audit_2026-05-28
- Related Files: tests/test_contract_output_schemas.py, docs/CODEGRAPH_RUNBOOK.md
- Tags: codegraph, snapshot_columns, contract_test
- Pattern-Key: always_check_existing_tests_before_writing_new_ones

## [LRN-20260528-001] codegraph_mcp_vs_cli_naming

**Logged**: 2026-05-28T03:38:00Z
**Priority**: medium
**Status**: resolved
**Area**: tooling

### Summary
`skills/codegraph/SKILL.md` Standard Workflow used CLI command names (`codegraph query`, `codegraph callers`, etc.) while IDE/agent sessions use MCP tool names (`codegraph_search`, `codegraph_callers`, etc.). Agents were falling back to shell calls instead of the faster in-process MCP path.

### Details
The CLI (`codegraph query`, `codegraph callers`, `codegraph impact`, `codegraph context`) and MCP tools (`codegraph_search`, `codegraph_callers`, `codegraph_impact`, `codegraph_context`) are functionally equivalent but differ in name. MCP tools are sub-millisecond in-process; CLI requires a shell spawn. `SKILL.md` now shows MCP names as primary with CLI equivalents inline. The `.cursor/rules/codegraph.mdc` workspace rule already used MCP names correctly.

### Suggested Action
When writing skills or rules for codegraph, use MCP tool names as primary with CLI equivalents in parentheses for bash use.

### Metadata
- Source: codegraph_skill_audit_2026-05-28
- Related Files: skills/codegraph/SKILL.md, .cursor/rules/codegraph.mdc
- Tags: codegraph, mcp, tooling, skill
- Pattern-Key: codegraph_mcp_vs_cli_naming

## [LRN-20260528-002] codegraph_env_pin_and_python_deps

**Logged**: 2026-05-28T03:38:00Z
**Priority**: high
**Status**: resolved
**Area**: cloud_environment

### Summary
`.cursor/environment.json` used `@colbymchenry/codegraph@latest` (non-deterministic) and did not install Python runtime deps. Every new Cloud Agent VM lacked PyYAML, pytest, python-dotenv, etc. until manually fixed. Also: LRN-20260525-001's claim that `pytest-xdist` is required was stale — `pyproject.toml` uses `-q -m 'not network'`, not `-n auto --dist worksteal`.

### Details
- Installed codegraph version is `0.9.6`; runbook had `0.9.4` (also stale).
- `pyproject.toml` `addopts = "-q -m 'not network'"` — xdist is NOT required.
- `requirements.txt` does not include `pytest-xdist`.
- Fixed: pinned to `@0.9.6`, added `pip install -r requirements.txt`.
- `CODEGRAPH_RUNBOOK.md` updated: version, index counts, removed xdist claim, removed hardcoded WSL paths from rollback section.

### Suggested Action
environment.json install command now: `npm install -g @colbymchenry/codegraph@0.9.6 && pip install -r requirements.txt && ...`
When upgrading codegraph, update the pinned version explicitly. Do not use `@latest`.

### Metadata
- Source: codegraph_env_audit_2026-05-28
- Related Files: .cursor/environment.json, docs/CODEGRAPH_RUNBOOK.md, requirements.txt, pyproject.toml
- Tags: codegraph, cursor_cloud, environment, pytest, pinning
- Pattern-Key: codegraph_env_pin_and_python_deps
- Supersedes: LRN-20260525-001 (xdist detail only — rest of that entry remains valid)

## [LRN-20260528-003] codegraph_not_wired_into_governance_skills

**Logged**: 2026-05-28T03:38:00Z
**Priority**: medium
**Status**: resolved
**Area**: tooling

### Summary
The `selector_ranker` skill (Tier 3, highest governance) had no codegraph preflight section. The `AGENT_ROUTING_POLICY.md` Tier 2 trigger was labelled "mechanical" but had no concrete tool to verify it.

### Details
- `skills/selector_ranker/SKILL.md`: Added mandatory 5-step codegraph preflight (search → node → callers → callees → impact) with explicit BLOCKED gate: impact reaching selector/ranker/decision_engine/final_score/rankings.csv requires operator approval.
- `governance/AGENT_ROUTING_POLICY.md`: Added `codegraph_impact --depth 2` as the verification step for the Tier 2 Claude Code review trigger. Impact list must appear in the PR description.

### Suggested Action
When writing new production-path skills (Tier 2+), always include a codegraph preflight section referencing `skills/codegraph/SKILL.md`.

### Metadata
- Source: codegraph_skill_audit_2026-05-28
- Related Files: skills/selector_ranker/SKILL.md, governance/AGENT_ROUTING_POLICY.md
- Tags: codegraph, governance, selector_ranker, preflight
- Pattern-Key: codegraph_missing_from_governance_skills

## [LRN-20260525-003] hermes_cloud_runtime_distinction

**Logged**: 2026-05-25T18:31:00Z
**Priority**: high
**Status**: promoted
**Area**: hermes_ops

### Summary
Repo-native Hermes MCP can be healthy in Cursor Cloud while production Hermes/Hermes Link runtime is not visible; cloud-generated knowledge artifacts may be stale relative to the active branch/runtime.

### Details
Hermes MCP exposed read-only tools and could read generated knowledge artifacts. However, `hermes`/`hermeslink` commands and expected ports were absent in Cursor Cloud. Knowledge artifacts recorded an older branch/head, so C2/C3/C5/first-fire warnings were quarantined pending fresh local/production runtime refresh.

### Suggested Action
Treat repo-native MCP as verified, Hermes Link/runtime absence as a cloud environment limitation, and artifact warnings as "needs local/production confirmation" until regenerated on the correct host/branch.

### Metadata
- Source: hermes_mcp_check
- Related Files: .cursor/mcp.json, mcp_server/hermes_server.py, docs/hermes_skills/hermeslink-state-capture.md
- Tags: hermes, mcp, hermeslink, cursor_cloud, knowledge_layer
- Pattern-Key: hermes_mcp_vs_runtime_visibility
