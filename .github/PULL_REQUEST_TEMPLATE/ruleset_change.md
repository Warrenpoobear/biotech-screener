## Ruleset Change

**Candidate ID**: <!-- 8-char hex, e.g. 68729184 -->
**Action**: <!-- bump / promote / parameter tweak -->

### Checklist

- [ ] Candidate created via `scripts/bump_ruleset.py` (not manual manifest edit)
- [ ] `manifest.json` entry has `updated_by` and `updated_at` fields
- [ ] `RULESET_CHANGELOG.md` entry is finalized (no `[DRAFT]` marker)
- [ ] Local tests pass: `python -m pytest tests/test_decision_ruleset.py -v`
- [ ] Contract tests pass: `python -m pytest tests/test_decision_engine_contract.py -v`
- [ ] Golden fingerprint refreshed if decision logic changed (`scripts/refresh_goldens.py`)
- [ ] `ruleset-release` workflow run linked: <!-- paste Actions URL -->

### Parameter Changes

<!-- Output of bump_ruleset.py or diff summary -->

### Rationale

<!-- Why this change? Link to backtest results, policy matrix output, etc. -->
