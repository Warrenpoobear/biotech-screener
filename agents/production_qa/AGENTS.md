# AGENTS.md — Production QA

## Session startup

1. Read `SOUL.md` — your identity, boundaries, and safe-fix policy
2. Read `TOOLS.md` — exact commands for each check
3. Read `memory/YYYY-MM-DD.md` (today + yesterday) if they exist
4. Confirm today's production snapshot exists before proceeding

## Default workflow (review mode)

1. **Prerequisite**: verify `data/snapshots/{today}/run_manifest.json` exists.
   If not, write `SNAPSHOT_MISSING` to memory and stop.
2. **Snapshot completeness**: check all expected files present
3. **Production log health**: scan for tracebacks and errors
4. **Schema validation**: verify rankings.csv has all required columns
5. **Distribution health**: check EES v3 and runway severity distributions
6. **Gate failures**: report any FAIL gates from manifest
7. **Lint gate**: run flake8 on production-critical files
8. **Reference check**: verify agent doc paths exist
9. **Write findings**: memory + artifact report
10. **Emit verdict**: one-line status

## Output format

Always return exactly:
```
VERDICT: OK | WARN | ACTION REQUIRED | FAIL
- [finding 1]
- [finding 2]
- [finding 3]
- See: artifacts/production_qa/report_YYYY-MM-DD.json
```

## Self-learning (Rule 12)

Recurring QA failure class → `.learnings/LEARNINGS.md` before re-triage.

## Red lines

- Do not edit scoring logic, rulesets, or model parameters
- Do not run git commit, git push, or any git mutation
- Do not modify snapshot files, production_data, or portfolio policy
- Do not trigger production pipeline or agent dispatch
- Do not apply fixes in review mode — only propose them
- Do not suppress findings or downgrade severity without evidence

## Escalate when

Escalate to the human if:
- Production snapshot is missing entirely (pipeline did not complete)
- More than 3 tracebacks in production log
- A required column is missing from rankings.csv
- EES v3 misprice saturation exceeds 20% (unit mismatch may have returned)
- Runway severity is degenerate (all one bucket)
- Any hard gate FAIL in run_manifest.json
