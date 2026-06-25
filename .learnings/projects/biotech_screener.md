# Project: Biotech Screener

<!-- WARM tier — per-project learnings (≤200 lines) -->

## Skill recursion (meta)
- Stack map: `.learnings/README.md` · audit: `python3 tools/audit_learnings.py`
- Durable session lessons → `.learnings/LEARNINGS.md` → HOT `memory.md` → `skills/<dir>/SKILL.md` → `sync_hermes_skills.py` → `harvest_log.md`.
- Domain ops patterns: `.learnings/domains/agent_ops.md`
- Ops/hermes lessons: `screener_ops`. Tooling/preflight: `codegraph`. Context bloat: `openclaw-agent-optimize`.
- Loop spec: `skills/self-improving/SKILL.md`. Do not encode scoring/cron changes in skills without governance Spec.

## Tooling (codegraph)
- Installed: v0.9.9 (pinned in `.cursor/environment.json`). Re-verify with `codegraph status` after merges.
- MCP tools (`codegraph_search`, `codegraph_callers`, `codegraph_callees`, `codegraph_impact`, `codegraph_context`) for IDE/agent. CLI (`codegraph query`, etc.) for bash.
- Preflight (search → node → callers → callees → impact) is mandatory before any Tier 2+ edit per `skills/codegraph/SKILL.md`.
- `codegraph_impact` on each changed symbol is the mechanical verification step for the AGENT_ROUTING_POLICY Tier 2 review trigger.
- Hermes agents use `common/codegraph_guard.py` (CodegraphGuard). All 5 gates implemented. Registration no longer deferred.
- `@latest` caused non-deterministic installs — always pin version explicitly when upgrading.

## Agent registry / Hermes
- Repo agent fleet: 31 directories = 29 active + 2 deprecated (`bioshort_watch`, `shadow_watch`). Do not re-add absent overlap dirs (`policy_shadow_watch`, `biotech_news_digest`, `company_news_ingest`) to `AGENT_REGISTRY.json` unless their directories are restored.
- **Fleet migration (2026-06-24):** phases 2–13 code-complete on `main`. Deterministic cron, fleet ops artifacts, crontab verify, watchdog Herald recovery. Host gates: install crontab, run `run_fleet_operator_checklist.sh`, close F-2026-005/006 before `SELFIMPROVE_GATES_MET=1`. Index: `docs/AGENT_FLEET_ARCHITECTURE_INDEX.md`.

## Pipeline recovery (2026-06-24)

Classes M–P in `openclaw-data-pipeline-debug` (detail) · Town triage in `town-operator-bridge`:

- **M** — yfinance date: use `strftime("%Y-%m-%d")`, not `datetime.isoformat()`
- **N** — universe leak: delisted filter must cover refresh_prices, run_screen, run_screen_from_bundle, coverage denominators
- **O** — cache warm timeout: argparse CLI default for `--warm-sources` masked essential-only function default
- **P** — cron `sys.path`: `from tools.*` fails without `PROJECT_ROOT` insert (42× agents.log before fix)

## Enrichment
- indication_master --min-tickers 3 is the right cutoff (1,568 conditions, covers all shared by 3+ companies)
- Full enrichment build takes ~60min (API latency). Plan accordingly.
- PEV enrichment rate is the key metric: 19% → 66% was the enrichment session win.

## Graveyard / Survivorship
- 4,361 records, 219 tickers. Dataset is real but signal is not promotable (size confound).
- Terminated + Withdrawn = 2,401 (HIGH confidence). Completed-no-results = 1,960 (MEDIUM).
- Top severity tickers are large-cap (GILD, AMGN) — expected, they run most trials.
- Lead-failure-only variant (386 records) untested — potential next research step.

## Catalyst History
- 4,254 events, 308 tickers (90.3% coverage). SEC 74.8%, CTgov 22.7%, FDA 2.5%.
- 707 negative regulatory events. 695 event groups with date revisions.
- All raw signals show positive IC (size confound). Same pattern as graveyard.

## Policy Shadow (Spec 035)
- Current verdict: PROMISING (4/4 gates pass, +0.57pp, 60% win rate).
- C-tier drag: -0.78% P&L/weight-day (2x worse than A-tier).
- Headwind bleed: 2.3x non-headwind rate.
- Wired into run_screen.py — accumulating daily history.

## Shadow Portfolio
- KOD was 45% of all positive P&L (+$10,637 on GLOW2 readout).
- MAZE (-$5,669) and SLN (-$3,856) were top losers. MAZE exited; SLN still held as C-tier.
- PEPG: A-tier on smart_money despite deep drawdown — separate edge case.
