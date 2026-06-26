# Coding Standards

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18

## Purpose

Unified coding conventions shared across Wake Robin's Python codebases. Prevents cross-repo drift on shared rules (Decimal arithmetic, PIT safety, deterministic output) and provides a single reference for coding assistants (CLAUDE.md conventions).

---

## Repos in Scope

| Repo | Phase | Stack | Tests |
| --- | --- | --- | --- |
| `biotech-screener` | Production (v1.14.0) | Python 3.12, pydantic v2 | CI pipeline (currently RED ~48 days as of Jun 25; Jun 1 recovery target unconfirmed) |
| `asset-allocation` | Phase 23 | Python 3.12, pydantic v2, numpy, pandas, pyarrow | 386 passing |
| `hermes-agent` | v0.14.0 | Python, Docker | See CONTRIBUTING.md |
| `biotech_alpha_system_v1` | Legacy (Jan 2026) | Python | Minimal |
| `performance-validation` | Shell | -- | None |

---

## Universal Rules (All Repos)

### Decimal Arithmetic Mandate

- All **scoring** arithmetic MUST use `Decimal` (never `float`). Initialize from strings: `Decimal("500000000")`.
- **Statistical analysis** (IC measurement, Spearman correlation, bootstrap resampling) may use `float`/numpy/scipy.
- The `exp()` function in sigmoid formulas is exempt: compute in float, then convert result to Decimal before re-entering scoring paths.
- Rounding: `ROUND_HALF_UP`. Scores to 2 dp (`0.01`), rates to 4 dp (`0.0001`).

### Point-in-Time (PIT) Enforcement

- All dates MUST be ISO 8601 (`YYYY-MM-DD`).
- Never call `datetime.now()`. All timestamps derived from `as_of_date`.
- Standard PIT: `source_date <= as_of_date - 1 day`.
- Strict PIT: `source_date < as_of_date - 2 days` (for intraday data).
- Lookahead (`age_days < 0`): **reject unconditionally**.

### Deterministic Output

- Same inputs MUST produce byte-identical outputs.
- All JSON serialization uses sorted keys.
- All list operations use deterministic sort keys.
- Content hashes (SHA256) included in every output for verification.
- No external API calls during scoring (stdlib only).
- Random seed: 42 (when randomization is needed).
- No overwriting existing run directories; reruns create a new `run_id`.

### Governance Metadata

Every pipeline output MUST include:
```json
{
  "_governance": {
    "run_id": "<deterministic-hash>",
    "score_version": "<version>",
    "schema_version": "<version>",
    "parameters_hash": "sha256:<hash>",
    "pit_cutoff": "<ISO-date>",
    "as_of_date": "<ISO-date>"
  }
}
```

---

## Repo-Specific Rules

### biotech-screener

```bash
# Test
pytest -p no:warnings
# Lint
ruff check src tests scripts tools
ruff format --check src tests scripts tools
```

- **CLAUDE.md:** 42KB comprehensive coding assistant instructions
- **Architecture freeze:** LIFTED 2026-05-26 (h20d checkpoint passed). No new enforcement logic or scoring changes without explicit operator approval.
- **CI pipeline:** GitHub Actions. Currently RED since ~May 8 (~48 days as of Jun 25). PR #285 open/unmerged. Jun 1 recovery target unconfirmed.
- **Key constraint:** Always warm 8-K cache BEFORE running screen.

### asset-allocation

```bash
# Test (omit cvxportfolio-gated tests)
.venv/bin/pytest -p no:warnings --ignore=tests/test_transaction_cost_summary.py
# Lint
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
# Run
.venv/bin/python scripts/run_sfo_study.py --config configs/base.yaml
```

- **Phase gates are real.** Each phase ships a `docs(model): lock Phase N` design commit BEFORE implementation.
- **MODEL_DOCUMENTATION.md** is doc-as-spec -- every behavior change updates it in the same series.
- **CMA baseline is immutable.** Scenarios are perturbations.
- **Quarterly ledger is the spine.** Every flow lands on it. New flow types require a Phase doc-lock.
- **What NOT to do:** Don't hard-code 60/40. Don't bypass design-lock. Don't push red main. Don't build parallel cash-flow forecasts that silently conflict with workbook.

### hermes-agent

- **AGENTS.md:** 46KB agent fleet documentation
- **CONTRIBUTING.md:** 28KB contributor guide
- **Prompting conventions:** IF/THEN chains, step numbering, schema-first output, no inferred data (originally Llama-optimized, carried forward to DeepSeek v4)
- **Model routing:** "deepseek" models -> Together API (OpenAI-compatible), "claude" -> Anthropic SDK
- **Primary model (as of 2026-05-20):** DeepSeek v4 flash via Together AI
- **Docker:** Compose file + multi-stage build

---

## Naming Conventions

### Signal Names (Cross-Repo)

| Current Name | Legacy Name | Notes |
| --- | --- | --- |
| `coinvest_score_z` | `sponsorship_score_z` | Renamed v1.14.0 |
| `inst_delta_z` | `momentum_delta_z` | Renamed v1.14.0 |

Always use current names in new code. When encountering legacy names in documentation or .docx files, treat as identical (see CON-1 in selector-ranker).

### File Naming

- Python files: `snake_case.py`
- Config files: `snake_case.yaml`
- Markdown docs: `UPPER_CASE.md` (repo root), `snake_case.md` (subdirectories)
- Production data: `production_data/{descriptor}.json`
- Artifacts: `artifacts/{category}/{YYYY-MM-DD}/`

---

## Git Workflow

### Branches

- `main`: Production. Must be green (currently violated -- CI red ~48 days as of Jun 25; Jun 1 recovery target unconfirmed).
- `feature/*`: Feature branches. PR required for merge.
- `hygiene/*`: Cleanup work. Can be deferred during freeze.

### Architecture Freeze Protocol

During freeze windows:
- No new enforcement logic or scoring changes
- Monitoring and documentation changes are allowed
- CI fixes and test-only changes are allowed
- Spec research continues but does not land in production
- Freeze lifts after explicit operator approval at checkpoint

### Signal / Schema Rename Protocol (doc-propagation checklist)

A rename only counts as "done" when it has propagated across every layer that references the old name. Renaming a signal, score field, or schema key in code MUST be accompanied by this checklist (origin: failure F-2026-001, NC, HIGH severity — the v1.14.0 `sponsorship_score_z` -> `coinvest_score_z` and `momentum_delta_z` -> `inst_delta_z` rename left 4+ documents stale):

1. **Code** — rename in production code and config (`production_data/`).
2. **Skills** — update every skill that names the field.
3. **GitHub model docs** — update `model_documentation_root.md` / `docs/MODEL_DOCUMENTATION.md` in the same PR or a follow-up within 48h.
4. **.docx exports** — flag for update at the next external-doc release.
5. **Agent prompts** — grep agent `SOUL.md` / `AGENTS.md` for the legacy name.
6. **Changelog** — add a `RULESET_CHANGELOG` entry naming BOTH the old and new identifiers.

Until all six are done, the old and new names must be documented as equivalent so no layer is read in isolation.

---

## Anti-Patterns (Do Not Do)

1. `float` in scoring paths (use `Decimal`)
2. `datetime.now()` anywhere (use explicit `as_of_date`)
3. Hard-coded allocation ratios (read from config)
4. Overwriting existing run directories
5. Raw EDGAR XML as source of truth (use canonical summary)
6. Pushing red main (WIP commits stay local)
7. Implementation before design-lock commit
8. Behavior changes without MODEL_DOCUMENTATION.md update
