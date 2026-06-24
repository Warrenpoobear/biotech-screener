# HOT Memory (≤100 lines)

<!-- Bootstrap block: lines 5–18 — load first every session. Promote after Pattern-Key ≥3×. -->

## Bootstrap (read first)

| Priority | Rule |
| --- | --- |
| Recursion | `LEARNINGS.md` → `memory.md` / `domains/` / `projects/` → `skills/` → sync → `harvest_log.md` |
| Audit | `python3 tools/audit_learnings.py` · map: `.learnings/README.md` |
| CodeGraph | **First → grep/read → edit.** v0.9.9 pinned. Bounded — not cron/literals/dispatch proof. |
| Host | **WSL** = cron, hedge, gateway. **Cloud** = repo/CI/skills; `UNKNOWN_CLOUD_ENV` expected. |
| Governance | No ranker/selector/sizing/`final_score` without Spec. Track B skips = expected. |
| CI | Actions budget pre-start ≠ code failure. |
| Hermes | MCP = read-only fleet; cron agents → `codegraph_guard.py`; gateway on WSL only. |
| Agents | Registry has 31 dirs: 29 active + 2 deprecated (`bioshort_watch`, `shadow_watch`). |
| Self-learn | `operational-health-baselines` skill live; weekly `tools/cron_weekly_skills_review.sh` on WSL. |

Detail: `domains/agent_ops.md` · skills: `self-improving`, `codegraph`, `screener_ops`, `operational-health-baselines`

---

## Code style

- Plain strings for static markdown table headers — not f-strings (flake8 F541). **5×**
- Remove unused imports before commit (flake8 F401). **4×**

## Research signals

- **raw_count_size_confound**: raw event/trial counts correlate with size — residualize before signal tests. **3×** (PI trials, graveyard, catalyst density). LRN-20260329-001 promoted.

## Portfolio construction

- Shadow drag = **construction policy** (flat 3% C-tier), not ranker defects. Tier weights A=4/B=2.5/C=1/D=0 → +1.60pp.
- Headwind + deep_drawdown bleeds **2.3×** non-headwind. Exit overlay +0.22pp.

## API patterns

- Open Targets: search returns generic `SearchResult` — no inline fragments. Two-step: search → ID → fetch.

## Ops (compact)

- `--snapshot-dir`: pass **parent** dir — run_screen appends date (no double nesting).
- Weekend: `run_daily_production.py` blocks; use `run_screen.py` for manual runs.
- Cloud: `pip install -r requirements.txt` before screen/pytest; **pytest-xdist not required** (LRN-20260528-002).
- **cron_sys_path_isolation**: cron entry scripts need `PROJECT_ROOT` on `sys.path` before `from tools.*` — interactive shell masks the bug. LRN-20260624-001.

## Stalled-loop verdicts (Rule 12 efficacy gate — operator fill-in)

Patch-efficacy tracking is **blocked** until both rows close. Cloud cannot confirm host recovery — operator verifies on WSL.

| ID | System | Status | Evidence (2026-06-24) | Operator close criterion | Target |
| --- | --- | --- | --- | --- | --- |
| F-2026-005 | Herald Digest | **OPEN** | Last repo classified JSONL: `2026-02-26`; no `artifacts/herald/` in cloud clone | Host: Herald cron produces `deduped` + `classified` JSONL for ≥1 trading day; zero recurrence 14d post-fix | **2026-07-01** (operator confirm) |
| F-2026-006 | GitHub CI | **OPEN** | `main` workflow failures complete in ~3–4s (budget pre-start pattern); merge gates unverified | Host: Actions budget restored; `tests` workflow green on `main` push | **2026-07-01** (operator confirm) |

When either closes: set `promotion_status: RESOLVED` in failure-patterns feed, append harvest_log efficacy block, unblock Rule 12 back-check for that outage's patches.
