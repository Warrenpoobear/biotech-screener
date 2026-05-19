# Errors Log

<!-- Self-improving agent error entries. Format: [ERR-YYYYMMDD-XXX] -->

## [ERR-20260329-001] open_targets_drug_query_400

**Logged**: 2026-03-29T17:00:00Z
**Priority**: high
**Status**: resolved
**Area**: data_pipeline

### Summary
Open Targets drug detail query returned HTTP 400 — field `maxPhaseForIndication` does not exist on `ClinicalIndicationFromDrug` type.

### Error
```
HTTP 400: {"errors":[{"message":"Cannot query field 'maxPhaseForIndication' on type 'ClinicalIndicationFromDrug'."}]}
```

### Context
The OT API schema changed the field name to `maxClinicalStage`. The `_graphql_post` helper was swallowing the 400 error silently (returning None), so the failure appeared as "0 results" rather than an explicit error.

### Suggested Fix
Applied: renamed field to `maxClinicalStage`. Also added two-step query pattern (search → detail) to avoid SearchResult type fragment issue.

### Metadata
- Reproducible: yes (prior to fix)
- Related Files: tools/enrich_open_targets.py
- See Also: LRN-20260329-003

## [ERR-20260329-002] snapshot_double_nesting

**Logged**: 2026-03-29T17:45:00Z
**Priority**: medium
**Status**: resolved
**Area**: ops

### Summary
run_screen.py with `--snapshot-dir data/snapshots/2026-03-28` nested output under `data/snapshots/2026-03-28/2026-03-28/` instead of directly in the target directory.

### Error
Screen output files were under a double-nested path. Had to manually `mv` files up one level.

### Context
The `--snapshot-dir` flag appends the date as a subdirectory internally. When the user provides a date-suffixed path, it creates double nesting.

### Suggested Fix
Either use `--snapshot-dir data/snapshots` (without date) or strip trailing date from the provided path. Documented as known behavior.

### Metadata
- Reproducible: yes
- Related Files: run_screen.py

## [ERR-20260519-001] telegram_status_reported_resolutions_as_latest_snapshot

**Logged**: 2026-05-19T03:30:00Z
**Priority**: high
**Status**: resolved
**Area**: ops

### Summary
Telegram `/status` and `/snap` always reported `Latest snapshot: resolutions` instead of the actual most recent dated snapshot.

### Error
```
> /status
<b>System Status</b>
<b>Latest snapshot:</b> resolutions
```

### Context
`tools/telegram_command_handler.py` `:251` and `:335` did `sorted([d.name for d in snap_dir.iterdir() if d.is_dir()])[-1]`. `data/snapshots/` contains a non-date subdir `resolutions/`. Because `'r'` (0x72) > `'2'` (0x32) in ASCII, `resolutions` lex-sorts after every `YYYY-MM-DD` name, so `[-1]` always picked it.

### Suggested Fix
Applied: shared helper `_list_snapshot_dates()` that filters by regex `^\d{4}-\d{2}-\d{2}$` before sorting.

### Metadata
- Reproducible: yes (prior to fix)
- Related Files: tools/telegram_command_handler.py
- Fix Commit: dde2ae9
- See Also: LRN-20260519-001

## [ERR-20260519-002] claude_settings_hook_silently_broken_after_deprecation

**Logged**: 2026-05-19T03:40:00Z
**Priority**: medium
**Status**: resolved
**Area**: ops

### Summary
`.claude/settings.json:9` wired a `UserPromptSubmit` hook to `./skills/self-improvement/scripts/activator.sh`, but the script had been renamed to `activator.sh.deprecated` (along with `error-detector.sh`). The hook failed silently on every prompt.

### Error
Hook command resolved to a non-existent file; Claude Code did not surface the failure. `.learnings/` nudges never triggered.

### Context
Discovered during dead-file sweep. The `.deprecated` rename disabled execution but did not update settings.json, leaving an inconsistent state. `.learnings/` directory itself was alive — only the activator was orphaned.

### Suggested Fix
Applied: removed the `UserPromptSubmit` hook block (`settings.json` → `{}`) and deleted both `.deprecated` scripts plus the now-empty `skills/self-improvement/{scripts,}` dirs.

### Metadata
- Reproducible: yes (prior to fix)
- Related Files: .claude/settings.json, skills/self-improvement/
- Fix Commit: ec08489
- Tags: hook, deprecation_drift, settings

## [ERR-20260519-003] first_fire_ledger_emitted_dual_status_and_eval_fields

**Logged**: 2026-05-19T03:45:00Z
**Priority**: medium
**Status**: resolved
**Area**: data_pipeline

### Summary
`artifacts/ops/first_fire_ledger/latest.json` carried both a stale `status: "PENDING"` (from `FIRST_FIRE_SEED`) and a live `eval: "PASS"|"FAIL_..."` (from `check_first_fire_status()`). Spec 089 documents a single `status` field — wire format violated the spec.

### Error
No runtime error. Latent. Three internal readers worked around it with `.get("eval", item["status"])` at `tools/build_hermes_knowledge_layer.py` `:541`, `:682`, `:831`. Any human (or future consumer) reading the JSON would see contradictory values.

### Context
`check_first_fire_status(result)` did `result = dict(ff_item)` then `result["eval"] = ...`, leaving the seed's `status` field untouched. Both keys reached the output ledger.

### Suggested Fix
Applied: overwrite `status` in place inside `check_first_fire_status()`; drop the `.get("eval", ...)` fallback at all three reader sites. No external consumers existed, so the fix is invisible downstream.

### Metadata
- Reproducible: yes (prior to fix)
- Related Files: tools/build_hermes_knowledge_layer.py, specs/changes/spec_089_hermes_knowledge_layer.md
- Fix Commit: 37d9997
- See Also: LRN-20260519-002
