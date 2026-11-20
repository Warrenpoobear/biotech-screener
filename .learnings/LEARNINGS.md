# Learnings Log

<!-- Self-improving agent learning entries. Format: [LRN-YYYYMMDD-XXX] -->
<!-- Entries logged during conversations; promoted to CLAUDE.md / memory when verified. -->

## [LRN-20260329-001] size_confound_in_raw_event_counts

**Logged**: 2026-03-29T19:00:00Z
**Priority**: high
**Status**: promoted
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

## [LRN-20260601-001] knowledge_skill_recursion_loop

**Logged**: 2026-06-01T12:00:00Z
**Priority**: high
**Status**: promoted
**Area**: hermes_ops

### Summary
Durable agent improvement requires a closed loop across `.learnings/`, `skills/`, and `harvest_log.md` — not skills alone.

### Details
Sessions produced ops and tooling lessons (WSL gate, CodeGraph bounds, CI budget vs code) that belonged in tiered learnings before mirroring to skills. Without README map and `audit_learnings.py`, agents re-discovered the same stack each session.

### Suggested Action
Session-end: run `audit_learnings.py`; promote Pattern-Key ≥3 to `memory.md`; patch eligible skills; sync + harvest_log.

### Metadata
- Source: knowledge_recursion_session_2026-06-01
- Related Files: .learnings/README.md, tools/audit_learnings.py, skills/self-improving/SKILL.md
- Tags: knowledge, recursion, skills, harvest
- Pattern-Key: knowledge_skill_recursion_loop
- Recurrence-Count: 1
- Skill-Path: self-improving

## [LRN-20260601-002] codegraph_bounded_not_authority

**Logged**: 2026-06-01T12:00:00Z
**Priority**: medium
**Status**: promoted
**Area**: tooling

### Summary
CodeGraph is healthy for structural navigation but must not be treated as authority for cron, runtime artifacts, or governance truth.

### Details
Surface split: MCP (Cursor), CLI (shell), Hermes MCP (fleet only), grep/read (literals, subprocess). Ranker/selector/scoring paths stay gated even when impact looks small.

### Suggested Action
Keep preflight in skills/codegraph; log tooling gaps to LEARNINGS with Skill-Path codegraph.

### Metadata
- Source: codegraph_review_2026-06-01
- Related Files: skills/codegraph/SKILL.md, .cursor/rules/codegraph.mdc
- Tags: codegraph, governance, bounded_proof
- Pattern-Key: codegraph_bounded_not_authority
- Recurrence-Count: 2
- Skill-Path: codegraph

## [LRN-20260624-001] cron_sys_path_isolation

**Logged**: 2026-06-24T21:00:00Z
**Priority**: high
**Status**: promoted
**Area**: hermes_ops

### Summary
Hermes cron entry scripts fail with `ModuleNotFoundError: No module named 'tools'` when using repo-relative imports without inserting `PROJECT_ROOT` onto `sys.path`. Interactive shells mask the bug via virtualenv or `PYTHONPATH`.

### Details
Confirmed 2026-06-24 (735ac3f7): `agents_direct` cron fired 42× before fix. Town `cron_missed` events are often the operator's first signal. Pattern: `PROJECT_ROOT = Path(__file__).resolve().parent.parent` then `sys.path.insert(0, str(PROJECT_ROOT))` before any `from tools.*` or `from common.*` imports.

### Suggested Action
Audit all cron entry scripts for repo-relative imports. Cross-ref `openclaw-data-pipeline-debug` Class P and `town-operator-bridge` triage table.

### Metadata
- Source: pipeline_recovery_2026-06-24
- Related Files: agents/*/run_job.py, tools/run_daily_production.py
- Tags: cron, sys_path, hermes_ops, town_bridge
- Pattern-Key: cron_sys_path_isolation
- Recurrence-Count: 1
- Skill-Path: town-operator-bridge, openclaw-data-pipeline-debug

## [LRN-20260624-002] yfinance_isoformat_date_parse

**Logged**: 2026-06-24T21:30:00Z
**Priority**: medium
**Status**: promoted
**Area**: data_pipeline

### Summary
Passing `datetime.isoformat()` dates to yfinance produces `T00:00:00` suffix that breaks `history()` parsing.

### Details
Confirmed 2026-06-24 (399e674c): XBI re-fetch crashed with "unconverted data remains: T00:00:00". Use `strftime("%Y-%m-%d")` for all yfinance date arguments.

### Suggested Action
Audit all yfinance call sites for datetime vs date string formatting.

### Metadata
- Source: pipeline_recovery_2026-06-24
- Pattern-Key: yfinance_isoformat_date_parse
- Skill-Path: openclaw-data-pipeline-debug

## [LRN-20260624-003] multi_path_universe_leak

**Logged**: 2026-06-24T21:30:00Z
**Priority**: high
**Status**: promoted
**Area**: data_pipeline

### Summary
Fixing delisted-ticker filtering in one universe loader does not remove the ticker from screen output — multiple loaders must be patched.

### Details
Confirmed 2026-06-24 (5b3225696): TERN still appeared after refresh_prices fix. Consumers: refresh_prices, run_screen.py, run_screen_from_bundle.py, coverage ratio denominators.

### Suggested Action
When changing universe.json semantics, audit all consumers with `codegraph_impact` or grep for universe load paths.

### Metadata
- Source: pipeline_recovery_2026-06-24
- Pattern-Key: multi_path_universe_leak
- Skill-Path: openclaw-data-pipeline-debug, screener-ops

## [LRN-20260624-004] argparse_cli_default_masks_function_default

**Logged**: 2026-06-24T21:30:00Z
**Priority**: high
**Status**: promoted
**Area**: ops

### Summary
argparse CLI defaults override function-parameter defaults when both exist — production cache warm timed out because CLI included slow registries.

### Details
Confirmed 2026-06-24 (ebb33da5): `--warm-sources` CLI default included EUCTR/CTIS/ISRCTN, masking essential-only function default. Production step 1.5 hit 1800s every run.

### Suggested Action
Align argparse defaults with function defaults; keep slow registries on dedicated cron with per-source timeouts.

### Metadata
- Source: pipeline_recovery_2026-06-24
- Pattern-Key: argparse_cli_default_masks_function_default
- Skill-Path: openclaw-data-pipeline-debug, screener-ops

## [LRN-20260601-003] sync_reference_mirror_source

**Logged**: 2026-06-01T12:00:00Z
**Priority**: medium
**Status**: resolved
**Area**: tooling

### Summary
`sync_hermes_skills.py` synced REFERENCE mirrors from SKILL.md when both exist, duplicating full skill bodies into `*-reference.md`.

### Details
Fixed `_source_path()` to prefer REFERENCE.md when target is in REFERENCE_MAP. Prevents mirror drift and context bloat in Hermes copies.

### Suggested Action
After adding skills/*/REFERENCE.md, always run sync + audit_hermes_skills.

### Metadata
- Source: skills_recursion_pr_2026-06-01
- Related Files: tools/sync_hermes_skills.py, docs/hermes_skills/self-improving-reference.md
- Tags: sync, skills, mirror_drift
- Pattern-Key: sync_reference_mirror_source
- Recurrence-Count: 1
- Skill-Path: self-improving

## [LRN-20260627-001] dem_is_current_ranker_not_baseline

**Logged**: 2026-06-27T16:28:00Z
**Priority**: high
**Status**: logged
**Area**: research

### Summary
"DEM" is the stored production top-30 output of the current pipeline, NOT a separate baseline algorithm. actionable_rank = final_score rank = A4 selector (sel_score) + clinical_50 ranker. Prior "DEM vs A4" framing compared stored production output against fresh recomputation variants (forward-filled inst_delta_z), not two independent algorithms.

### Details
Architecture confirmed across snapshots (DEM_CURRENT_RANKER_YTD_BACKTEST, 2026-06-27): pipeline is universe -> A4 selector -> clinical_50 ranker -> actionable_rank -> DEM top-30. A4 selector overlay FROZEN (anti-alpha vs DEM); inst_delta_z relaxation failed; catalyst-optionality (cat_opt, incl. solvency variant) failed and worsened the left tail, with cat_opt∩DEM overlap collapsing post-2024. Performance: 2025+ (n=14) DEM +7.898 pp/mo, t=2.167, hit 64.3% vs A4 +3.656, t=1.683 (DEM +4.2 pp/mo better); full history (n=69) DEM +3.048, t=3.309 vs A4 +0.242. Worst-month: DEM -8.87 vs A4 -14.32 vs cat_opt -21.24. Concentration: top-5 months (May/Jun/Jul 2025, Sep 2025, Feb 2026) = 100% of cumulative 2025+ alpha; DEM ex-best-5-months ≈ 0 (rally-concentrated, not yet persistent cross-sectional alpha).

### Suggested Action
Treat DEM as a forward-shadow candidate, NOT investable production. Gate any promotion on DEM_REGIME_CONDITIONAL_ALPHA (does the 2025+ edge survive outside the May–Jul 2025 rally cluster; is it alpha vs high-beta rally participation). Do not reopen A4 / inst-relax / cat_opt without a new spec and fundamentally new evidence.

### Metadata
- Source: dem_current_ranker_ytd_diagnostic_2026-06-27
- Tags: dem, selector_ranker, naming_collision, rally_concentration, forward_shadow
- Pattern-Key: dem_is_current_ranker_not_baseline
- Recurrence-Count: 1
- Skill-Path: decision-audit-trail (D-2026-008), failure-patterns (F-2026-010), selector-ranker, ic-evaluation

## [LRN-20260628-001] production_runner_is_run_daily_production

**Logged**: 2026-06-28T00:00:00Z
**Priority**: high
**Status**: promoted
**Area**: hermes_ops

### Summary
The daily production runner is `tools/run_daily_production.py`, NOT `scripts/run_batch.py`. The biotech-run-pipeline skill referenced the wrong file for the entire session history.

### Details
`scripts/run_batch.py` is old/deprecated. Production entrypoint is `tools/run_daily_production.py`. The pipeline has evolved to include Steps 1.45 (tastytrade options snapshot), 1.46 (options enrichment), 1.47 (options shadow IC update). Snapshot dir is `data/snapshots/`; `run_manifest.json` is the primary completion artifact.

### Suggested Action
Always verify runner entrypoint from crontab or `tools/` first, not from memory or old skill text. biotech-run-pipeline SKILL.md updated.

### Metadata
- Source: biotech_pipeline_session_2026-06-28
- Pattern-Key: production_runner_is_run_daily_production
- Recurrence-Count: 1
- Promotion-lane: skill
- Skill-Path: biotech-run-pipeline

## [LRN-20260628-002] run_screen_inputs_manifest_valid_choices

**Logged**: 2026-06-28T00:00:00Z
**Priority**: medium
**Status**: logged
**Area**: hermes_ops

### Summary
`run_screen.py --inputs-manifest` only accepts `off`, `write`, `verify`. Using `skip` causes argparse to print help and exit (1.6s silent failure).

### Details
During timing test, used `--inputs-manifest skip` → argparse error → help printed → 1.6s exit. Valid choices confirmed at `run_screen.py:12301`. Production uses `write`; `off` disables the sidecar.

### Suggested Action
When scripting run_screen.py in isolation (timing tests, debugging), use `--inputs-manifest off` to skip sidecar I/O.

### Metadata
- Source: run_screen_timing_test_2026-06-28
- Pattern-Key: run_screen_inputs_manifest_valid_choices
- Recurrence-Count: 1
- Promotion-lane: skill
- Skill-Path: biotech-run-pipeline

## [LRN-20260628-003] pipeline_perf_delisted_filter_ctgov_parallel

**Logged**: 2026-06-28T00:00:00Z
**Priority**: medium
**Status**: logged
**Area**: data_pipeline

### Summary
Three production bottlenecks cut pipeline from 22 min → ~6 min on 2026-06-28. Each fix was independent.

### Details
1. **Delisted tickers in yfinance refresh**: universe.json contains `status=delisted` entries; old code retried all tickers 3× including delisted ones → 9.3 min. Fix: filter `e.get("status") == "delisted"` before building ticker list.
2. **tastytrade batch_size 100→200**: API accepts 200 per batch; halved round-trips → 4 min → <2s.
3. **ctgov serial→parallel**: `ThreadPoolExecutor(max_workers=8)` with `threading.Lock()` for shared state; `time.sleep(0.2)` per-thread preserved; 6.4× speedup (3.3 min → 31s). Commit: `412d97f6`.

### Suggested Action
When diagnosing slow pipeline steps: (1) check for delisted tickers in universe, (2) check batch sizes against API limits, (3) look for serial loops over network calls.

### Metadata
- Source: pipeline_perf_session_2026-06-28
- Tags: yfinance, tastytrade, ctgov, parallelization, delisted
- Pattern-Key: pipeline_perf_delisted_filter_ctgov_parallel
- Recurrence-Count: 1
- Promotion-lane: none
- Skill-Path: none (implementation detail, not skill-worthy pattern)

## [LRN-20260628-004] options_ic_5bday_forward_window

**Logged**: 2026-06-28T00:00:00Z
**Priority**: low
**Status**: logged
**Area**: research

### Summary
Step 1.47 (options shadow IC) requires 5 business days of forward price data before any IC value is available. First snapshot was June 26; first IC data available July 3.

### Details
`options_shadow_analysis.run()` looks back 5 business days from `as_of_date` for a snapshot. Each daily run accumulates one more cross-section until enough forward price data exists. Logs "insufficient forward data (accumulating)" until data is ready. `fast_mode=True` skips the slow RV IC (~30s) and runs only market-implied XS + autopsy cross-reference.

### Metadata
- Source: options_ic_framework_2026-06-28
- Pattern-Key: options_ic_5bday_forward_window
- Recurrence-Count: 1
- Promotion-lane: none

## [LRN-20260628-005] fetch_pending_biotech_data_is_the_fix_tool

**Logged**: 2026-06-28T00:00:00Z
**Priority**: high
**Status**: logged
**Area**: universe_maintenance

### Summary
`tools/fetch_pending_biotech_data.py` is the purpose-built tool for fixing `pending_data_collection` stubs. `refresh_eligible_biotech_universe.py` only flags — it does not fetch or fix.

### Details
`refresh_eligible_biotech_universe.py` identifies gaps and marks statuses but cannot fetch data. `fetch_pending_biotech_data.py` is the counterpart that actually fetches: market data + company_name from yfinance, financial data from SEC/yfinance, and CTGov trials from the API. It processes all tickers with status `pending_data_collection` or `pending_coverage`, merges results, runs `refresh_universe(finalize_collection=True)` internally, and writes back to both `production_data/universe.json` and `production_data/trial_records.json` when `--apply` is passed.

ETF-sourced stubs (added via `audit_universe_against_xbi_ibb.py`) arrive with bare universe.json entries (no nested `market_data` dict, no `name`). `market_data.json` may have price/market_cap for these tickers already, but the coverage check reads the nested `market_data` dict inside each universe.json row — not `market_data.json`. Only `fetch_pending_biotech_data.py` populates that nested dict.

### Suggested Action
When `refresh_eligible_biotech_universe.py` reports `pending_data_collection` tickers, the fix is: `python3 tools/fetch_pending_biotech_data.py --as-of-date YYYY-MM-DD --apply`. Dry-run first (omit `--apply`) to preview market_success, trial_success, promoted_active counts. Typical run: ~2 min for 11 stubs + 23 pending_coverage retries.

### Metadata
- Source: universe_refresh_session_2026-06-28
- Pattern-Key: fetch_pending_biotech_data_is_the_fix_tool
- Recurrence-Count: 1
- Promotion-lane: skill
- Skill-Path: biotech-snapshot-qa

## [LRN-20260628-006] refresh_eligible_universe_finalize_collection_flag

**Logged**: 2026-06-28T00:00:00Z
**Priority**: high
**Status**: logged
**Area**: universe_maintenance

### Summary
Without `--finalize-collection`, `refresh_eligible_biotech_universe.py` re-downgrades `pending_coverage` tickers to `pending_data_collection` every run, creating a circular report of 23+ "new" problems each week.

### Details
`pending_coverage` = "company + market data covered, CTGov has no trials for this ticker." Without `--finalize-collection`, the script treats any ticker missing `clinical_trials` or `scientific_cartography` as needing data collection, regardless of prior coverage attempts. With `--finalize-collection`, tickers whose only gaps are `clinical_trials`/`scientific_cartography` and whose `company_name` + `market_data` are covered get correctly classified as `pending_coverage` (tried, unavailable) rather than `pending_data_collection` (not yet tried).

The weekly refresh script (`cron_universe_weekly_refresh.sh`) must use `--finalize-collection` in step 1, and `fetch_pending_biotech_data.py` always uses `finalize_collection=True` internally. Running step 1 without it will falsely inflate the pending count by ~27 tickers weekly.

### Suggested Action
Always pass `--finalize-collection` to `refresh_eligible_biotech_universe.py` in automated/weekly contexts. For one-off diagnostics only (pre-fetch, to see raw gap count), omitting it is fine.

### Metadata
- Source: universe_refresh_session_2026-06-28
- Pattern-Key: refresh_eligible_universe_finalize_collection_flag
- Recurrence-Count: 1
- Promotion-lane: skill
- Skill-Path: biotech-snapshot-qa

## [LRN-20260628-007] run_agent_direct_bypasses_hermes_gateway

**Logged**: 2026-06-28T00:00:00Z
**Priority**: medium
**Status**: logged
**Area**: infrastructure

### Summary
`tools/run_agent_direct.py` calls Together.ai and Anthropic APIs directly. It does NOT route through the Hermes gateway (:8642/:8644). `HERMES_JOB_PREFIX = "hermes-"` is a naming guard, not a gateway reference.

### Details
The file's docstring says "Workaround for OpenClaw gateway billing issue." It instantiates `openai.OpenAI(base_url="https://api.together.xyz/v1")` or `anthropic.Anthropic(api_key=...)` directly. No HTTP calls to :8642, :8644, or :19001. The constant `HERMES_JOB_PREFIX = "hermes-"` guards against dispatching agents whose job names start with "hermes-" (those belong to the OpenClaw/Hermes scheduler, not run_agent_direct). The Hermes gateway manages its own separate fleet via the Hermes scheduler cron system.

When investigating whether ops/sentinel run "through Hermes," grep for gateway URLs (8642|8644) in the file — the result is empty. The Together.ai key is what matters for these agents' availability.

### Metadata
- Source: agent_routing_investigation_2026-06-28
- Pattern-Key: run_agent_direct_bypasses_hermes_gateway
- Recurrence-Count: 1
- Promotion-lane: none

## [LRN-20260628-008] ic_health_artifact_location_and_interpretation

**Logged**: 2026-06-28T00:00:00Z
**Priority**: medium
**Status**: logged
**Area**: validation

### Summary
IC health lives at `artifacts/ic_dashboard/YYYY-MM-DD_dashboard.json`. Primary signal is `score_rank_pct`; thresholds: HEALTHY ≥ +0.030, ALERT < 0.00. Regime monitor is at `artifacts/forward_validation/dem_regime_monitor_YYYY-MM-DD.json`.

### Details
`ic_dashboard` files are generated daily and contain `signals.score_rank_pct.{mean_ic, hit_rate, n_dates, health, per_date[]}`. `attention` field summarizes overall health (LOW / MEDIUM / HIGH). The `per_date` array gives individual IC readings — useful for trend analysis.

The 2026-04-08 to 2026-04-22 inversion (IC −0.18 at worst) was caused by the tariff-shock macro sell-off. Crossover to positive happened Apr 23 as XBI found its floor. The signal then ran strongly to +0.276 peak (May 14) before normalizing. This pattern should inform interpretation of future inversions: a sharp policy-driven macro shock can invert cross-sectional IC without implying model breakdown — recovery occurred within 15 trading days.

Regime monitor gate threshold: 20 windows. As of 2026-06-28, gate met at 118/20.

### Suggested Action
When checking model health, read `ic_dashboard` first (`sort -r` for latest file). If `attention=HIGH` or `mean_ic < 0.00`, do not open new positions. IC inversions during macro shocks can be transient — check `per_date` trend before concluding breakdown.

### Metadata
- Source: ic_health_check_2026-06-28
- Pattern-Key: ic_health_artifact_location_and_interpretation
- Recurrence-Count: 1
- Promotion-lane: skill
- Skill-Path: biotech-ic-check

## [LRN-20260629-001] financial_data_json_vs_financial_records_json

**Logged**: 2026-06-29T00:00:00Z
**Priority**: high
**Status**: logged
**Area**: universe_maintenance

### Summary
`production_data/financial_data.json` and `production_data/financial_records.json` are two different files. `run_screen.py` reads `financial_records.json`. Stubs enriched by `fetch_pending_biotech_data.py` land in `financial_data.json` only and will show `financials_missing` ineligibility until manually merged.

### Details
`fetch_pending_biotech_data.py` writes enriched financial data into `universe.json[ticker].financial_data` (nested) and also to `production_data/financial_data.json` (flat list, 360 records as of 2026-06-29). `run_screen.py` (line 9824) reads `production_data/financial_records.json` (341 records pre-fix) for survivability scoring via `module_5_composite_v3.py → compute_survivability_score()`. The `financial_records.json` file is what `module_2_financial.py` also uses for runway/burn scoring.

When 11 stubs were enriched on 2026-06-28, they appeared in `financial_data.json` but not `financial_records.json`. The decision engine saw no `Cash` or `CFO`, computed `missing_cash + missing_burn_data` coverage flags, and marked all 11 as `financials_missing` ineligible — despite them having valid Cash (ranging from $105M to $9.4B) and CFO data.

Fix: merge missing ticker records from `financial_data.json` into `financial_records.json` using the canonical field set (`ticker, cik, Assets, Cash, CFO, ShortTermInvestments, R&D, NetIncome, OperatingExpenses, CashAndSecurities, collected_at` + date companions). The schemas are compatible; `financial_data.json` has extra fields (LongTermDebt, Revenue, InterestExpense) and `financial_records.json` has `R&D`/`R&D_date` not in the enrichment output.

### Suggested Action
After any run of `fetch_pending_biotech_data.py --apply`, check for newly promoted tickers absent from `financial_records.json` and merge them in. Integrate this into `cron_universe_weekly_refresh.sh` Step 2 or add a Step 2.5 merge. Command pattern:
```python
missing = set(fin_dat_map.keys()) - set(fin_rec_map.keys())
# For each: copy record conforming to RECORD_KEYS, append to financial_records.json
```
Until this is automated, the gap will silently re-appear for every new stub batch.

### Metadata
- Source: ineligible_validation_2026-06-29
- Pattern-Key: financial_data_json_vs_financial_records_json
- Recurrence-Count: 1
- Promotion-lane: skill
- Skill-Path: biotech-snapshot-qa

## [LRN-20261117-001] ic_backtest_roster_hindsight_quote_both

**Context:** 2026-11-17 IC council model review. The 13F roster v4 was chosen on trailing 2023–26 returns; at Top-30 it ties the old roster v3.2 before the selection window (+0.10 pt/month, t 0.27) and gains all of its lead inside it. Every regenerated history (v5–v9) uses v4, so their Top-N figures carry the hindsight.

**Rule:** Quote every Top-N backtest under both rosters, split at the selection window.

**Suggested-Action:** Checklist item in `skills/biotech-ic-council/references/biotech-domain-checks.md` (Backtest validity), added 2026-11-17.

### Metadata
- Source: ic_council_model_review_2026-11-17
- Pattern-Key: IC_BACKTEST_roster_hindsight_quote_both
- Recurrence-Count: 2
- Area: research
- Promotion-lane: skill
- Skill-Path: skills/biotech-ic-council/SKILL.md

## [LRN-20261117-002] ic_production_baseline_red_check_masks_regressions

**Context:** The `replay` CI job failed on every run, `main` included (no pass in 400+ runs since at least 2026-06-10), because its golden rankings were made by February code. Every PR carried a red check that meant nothing, so a real ranking regression would have been invisible.

**Rule:** Before reading a red check as evidence about a PR, check the same job on `main`; a permanently red check is a broken check, fix the baseline.

**Suggested-Action:** Production reliability checklist item, added 2026-11-17; golden rankings regenerated in PR #702.

### Metadata
- Source: ic_council_model_review_2026-11-17
- Pattern-Key: IC_PRODUCTION_baseline_red_check_masks_regressions
- Recurrence-Count: 1
- Area: hermes_ops
- Promotion-lane: skill
- Skill-Path: skills/biotech-ic-council/SKILL.md

## [LRN-20261117-003] ic_pit_leak_audit_each_input_class_and_reference_as_of

**Context:** The v9 rebuild was clean on 13F, market data and financials but per-date CT.gov caches carried trial records as captured months later (25–45% of records updated after the as-of date for 2020–2025 dates). Separately, the ETF audit's IBB holdings were pinned to 2026-06-26 by a hard-coded URL date for five months. The live 2026-11-16 snapshot was replaced; the forward-validation ledger was safe only because captures store baskets inline.

**Rule:** Audit regenerated histories per input class per date, read external reference data by its own as-of date, and before replacing a live snapshot confirm the ledger does not depend on the path.

**Suggested-Action:** PIT discipline checklist items, added 2026-11-17; IBB staleness guard in PR #700.

### Metadata
- Source: ic_council_model_review_2026-11-17
- Pattern-Key: IC_PIT_LEAK_audit_each_input_class_and_reference_as_of
- Recurrence-Count: 1
- Area: data_pipeline
- Promotion-lane: skill
- Skill-Path: skills/biotech-ic-council/SKILL.md
