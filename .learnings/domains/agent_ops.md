# Domain: Agent Operations (WARM)

<!-- Cross-cutting Hermes / Cursor / Cloud patterns. ≤200 lines. -->

## Host authority

| Host | Authoritative for |
| --- | --- |
| Operator WSL | `crontab`, `output/hedge_report/`, `~/.hermes/config.yaml`, B1b producer |
| Cursor Cloud | Repo plumbing, CodeGraph index, skills sync, ledger **build** (read-only) |

Cloud knowledge-layer builds emit `UNKNOWN_CLOUD_ENV` for C1/C3 — not a pass/fail. First-fire `FAIL_ARTIFACT_MISSING_PAST_DEADLINE` is expected without hedge artifacts in checkout.

## Hermes surfaces (do not conflate)

| Surface | Model? | Role |
| --- | --- | --- |
| Hermes MCP | No | Fleet registry, SOUL, ledgers when built |
| Lane A `hermes-*` jobs | No | Deterministic governance |
| Gateway `~/.hermes/config.yaml` | Yes | OpenRouter / DeepSeek intent on WSL |
| `run_agent_direct.py` | Yes (bypass) | Defaults Together Llama — not gateway |

## CodeGraph (bounded)

- MCP in Cursor; CLI on shell; Hermes cron uses `common/codegraph_guard.py`
- Rule: **CodeGraph first, grep/read second, edit third**
- Not proof for cron, subprocess, file-path literals, or dynamic dispatch
- Pin: `@colbymchenry/codegraph@0.9.9` in `.cursor/environment.json`

## Agent registry profiles

- Current repo fleet: 31 directories = 29 active + 2 deprecated retained workspaces (`bioshort_watch`, `shadow_watch`).
- Registry is bidirectional: every entry must have a matching `agents/<name>/` directory; absent retired overlaps stay documented in Hermes docs, not in `AGENT_REGISTRY.json`.

## CI signals

- Actions budget exhaustion pre-start → infrastructure, not PR regression
- Track B governance tests in draft PR — expected skips, not gates to “fix” toward scoring

## Skill ↔ knowledge recursion

1. Lesson → `LEARNINGS.md` with `Pattern-Key`
2. Promote → `memory.md` or this file
3. Executable → `skills/*` + `harvest_log.md`
4. Audit → `audit_learnings.py` + `audit_hermes_skills.py`

## Fleet migration + operator sequence (2026-06-25)

Phases 2–15 code-complete on `main`. Canonical WSL sequence:

```bash
bash tools/run_operator_host_setup.sh
FREEZE_LIFT_ACK=1 bash tools/run_forward_evidence_package.sh --write
```

| Script | Role |
| --- | --- |
| `run_operator_host_setup.sh` | Fleet onboarding + optional research battery + Path A shadow (A1) |
| `run_research_host_battery.sh` | Checklist v2 + Spec 100 IC + Spec 105 |
| `run_path_a_shadow.sh` | Spec 106 timing gates shadow (`portfolio_policy_path_a_shadow.json`) |
| `run_forward_evidence_package.sh` | Path C close + forward IC (does not lift freeze) |

Close F-2026-005 / F-2026-006 before `SELFIMPROVE_GATES_MET=1`. Path A design: `docs/governance/PATH_A_PORTFOLIO_TIMING_GATES_SPEC_106_2026_06_25.md` (A0+A1 shadow on main; A2 blocked by freeze).

## Cron sys.path isolation (Class P)

Hermes cron shells lack virtualenv activation and `PYTHONPATH`. Repo-relative imports
(`from tools.*`, `from common.*`) fail with `ModuleNotFoundError` even when interactive
shell works. Fix: insert `PROJECT_ROOT` onto `sys.path` before imports in every cron
entry script. Town `cron_missed` alerts may be the first signal — triage via
`town-operator-bridge` operator table.

Confirmed 2026-06-24 (735ac3f7): `agents_direct` cron fired 42× before fix.

## Pipeline recovery patterns (2026-06-24)

| Class | Pattern-Key | Skill |
| --- | --- | --- |
| M | `yfinance_isoformat_date_parse` | `openclaw-data-pipeline-debug` |
| N | `multi_path_universe_leak` | `openclaw-data-pipeline-debug`, `screener-ops` |
| O | `argparse_cli_default_masks_function_default` | `openclaw-data-pipeline-debug`, `screener-ops` |
| P | `cron_sys_path_isolation` | `openclaw-cron-scheduler-debug` Class J, `town-operator-bridge` |
