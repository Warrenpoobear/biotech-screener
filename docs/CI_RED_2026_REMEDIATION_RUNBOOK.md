# CI_RED_2026 — Remediation Runbook (Hermes WSL2 host)

**Incident:** CI_RED_2026 (single record: issue #521)
**Executor:** Hermes WSL2 host (has the toolchain; GitHub-hosted runners are `ubuntu-latest`)
**Repo:** `Warrenpoobear/biotech-screener` @ `main`
**Purpose:** Reproduce each red CI job locally, record the *exact* root cause per job, and drive the incident to closure — **not** to make CI green by the fastest available shortcut.

---

## Containment in force (from #521 — read before acting)

- `NO_NEW_MODEL_OR_PRODUCTION_LOGIC_MERGES`
- `DATA_ONLY_AUTOMATED_OUTPUTS`: CONTINUE_ONLY_IF existing production health gates pass
- `PKOS_M4_AND_STAGE_1`: PAUSED
- `CI_REPAIR_PRS`: ALLOWED
- `DOCUMENTATION_CORRECTIONS`: ALLOWED

## Two hard prohibitions (non-negotiable)

1. **DO NOT refresh the replay golden baseline merely to make CI green.** First classify the diff (Step 4).
2. **DO NOT lower the `--cov-fail-under=55` coverage threshold** to make pytest green. First separate the failure type (Step 3).

## Preconditions

- Work in a **dedicated clone or `git worktree`** — never the shared cron/production checkout (concurrent sessions + cron will clobber or be clobbered).
- `git fetch && git checkout main && git rev-parse HEAD` — record the SHA under test.
- Both interpreters available: **Python 3.10 and 3.12** (pytest runs as a matrix on both).
- `gh auth status` OK (Step 4 downloads the golden release bundle).
- Set `PYTHONHASHSEED=0` for pytest and replay (CI does).

---

## Execution order

### 1. `pre-commit run --all-files`  → reproduces the `lint` job

CI runs (Python 3.12): `pre-commit run --all-files --show-diff-on-failure`.

```bash
export PYTHONHASHSEED=0
python -m pip install --upgrade pip && pip install pre-commit
pre-commit run --all-files --show-diff-on-failure 2>&1 | tee /tmp/cired_lint.log
```

Record **which hooks** failed (black / isort / flake8 / etc.) and the exact file:line violations.
`model-docs-sync`, `semgrep`, and `secret-scan` are confirmed passing — if lint is red it is a formatting/lint hook, not those.

→ Populate: `LINT: REAL_FAILURE — <hooks + exact violations>`

### 2. Targeted pytest collection + failing tests  → isolates `pytest (3.10)` / `pytest (3.12)`

First prove collection/import is clean, *then* run tests **without coverage** so an assertion failure can't be confused with a coverage-threshold failure.

```bash
for PY in 3.10 3.12; do
  # (use the matching interpreter / venv for $PY)
  python -m pip install -r requirements.txt && pip install pytest-xdist yfinance && pip install -e .
  # 2a. collection only — surfaces import/collection errors
  python -m pytest tests/ --collect-only -q --ignore=tests/integration 2>&1 | tee /tmp/cired_collect_$PY.log
  # 2b. run tests, NO coverage gate — surfaces genuine assertion failures
  PYTHONHASHSEED=0 python -m pytest tests/ -q --override-ini="addopts=" \
    -m "not slow" --ignore=tests/integration 2>&1 | tee /tmp/cired_pytest_$PY.log
done
```

Note the **3.10 vs 3.12 delta** explicitly (the incident already observed 3.10 reporting more failing annotations than 3.12 — capture the version-specific difference).
Watch specifically for **data-coupled guardrail tests** (e.g. `test_rollup_guardrails.py`) that read live rollup/production CSVs and assert against drifting values — those are a *data* failure, not a code failure.

→ Populate: `PYTEST_3_10` / `PYTEST_3_12` with the failure **type** (see Step 3 taxonomy) and, for 3.12, the version-specific delta.

### 3. Coverage-only diagnosis  → separates assertion failure from coverage-threshold failure

If Step 2b's tests all **pass** but CI still reports pytest red, the failure is the coverage gate. Reproduce it exactly:

```bash
PYTHONHASHSEED=0 python -m pytest tests/ -q --override-ini="addopts=" \
  -m "not slow" --ignore=tests/integration \
  --cov=common --cov=tools --cov=scripts \
  --cov-report=term-missing:skip-covered --cov-fail-under=55 2>&1 | tee /tmp/cired_cov.log
python -m coverage report | tail -1   # actual total %
```

**Classify the pytest failure into exactly one of** (do not conflate):
```
test assertions failed
coverage threshold failed
collection/import failed
Python-version-specific failed
data-coupled guardrail failed
```
If it is `coverage threshold failed`: the fix is to **add/repair tests to reach 55%**, not to lower the threshold.

### 4. Replay diagnosis against the exact golden bundle  → reproduces the `replay` job

CI pulls the golden bundle from the `golden-baseline` GitHub Release and diffs candidate vs `golden/baseline_<DATE>`. Use the **same bundle** — do not invent inputs.

```bash
mkdir -p /tmp/golden && gh release download golden-baseline --pattern "replay_bundle_*.tgz" --dir /tmp/golden
BUNDLE=$(ls /tmp/golden/replay_bundle_*.tgz | head -1)
DATE=$(python -c "import tarfile,json,io,sys;\
t=tarfile.open(sys.argv[1],'r:gz');print(json.load(io.TextIOWrapper(t.extractfile('bundle_index.json')))['as_of_date'])" "$BUNDLE")
test -f "golden/baseline_${DATE}/rankings.csv" || echo "MISSING BASELINE for ${DATE} (this alone fails replay)"

mkdir -p /tmp/candidate
PYTHONHASHSEED=0 python run_screen.py --replay-bundle "$BUNDLE" --as-of-date "$DATE" \
  --data-dir /tmp --output /tmp/candidate/${DATE}/screen_output.json \
  --decision-mode phase2 --snapshot-dir /tmp/candidate

python scripts/replay_diff.py --baseline "golden/baseline_${DATE}" \
  --candidate "/tmp/candidate/${DATE}/" --output-dir output/ \
  --thresholds production_data/diff_thresholds/v1.json
echo "replay_diff exit: $?"   # 0 OK · 2 WARN(passes) · 1 FAIL(exceeds thresholds)
```

**Before touching the baseline, classify the diff as exactly one of:**
- **intended model behavior** — a merged change legitimately moved rankings → baseline refresh justified *and documented*.
- **expected data evolution** — inputs legitimately changed → decide whether replay should pin the bundle date (usually should).
- **nondeterminism** — non-seeded ordering/floats → fix the determinism, do not refresh.
- **PIT contamination** — candidate used data not knowable as of `as_of_date` → real bug, fix the leak.
- **genuinely stale baseline** — only then refresh, **with the justification recorded on #521.**

First check whether the failure is simply a **missing/absent baseline** (`golden/baseline_<DATE>/rankings.csv` absent) vs a **ranking-drift** failure — they have different fixes.

→ Populate: `REPLAY: REAL_FAILURE — <missing-baseline | drift + which of the 5 classes>`

### 5. Full CI-equivalent rerun  → confirm green before any merge

Run all gating jobs the way CI does (lint + pytest 3.10 + pytest 3.12 + replay; type-check pins **mypy 1.8.0**, not the dependabot 2.3.0 bump). Only after this is green locally, open a **CI-repair PR** and require the workflow runs themselves to be green.

**Two consecutive CI-equivalent runs must be green**, then **one subsequent production cycle green**, before closing.

---

## Exit criteria (mirror of #521 — all required)

- [ ] Exact root cause recorded for **every** failing job
- [ ] lint green
- [ ] pytest 3.10 green
- [ ] pytest 3.12 green
- [ ] replay green **with baseline decision justified** (baseline NOT refreshed merely to go green)
- [ ] stale failure documentation corrected (`FAILURE_PATTERN_LIBRARY.md` F-2026-006) — DONE via #522
- [ ] required checks enforced on `main` — ⚠️ native branch protection is unavailable on this repo's plan (403); needs a merge-gating workflow or plan/visibility change, **not** a checkbox
- [ ] two consecutive CI-equivalent runs green
- [ ] one subsequent production cycle green
- [ ] incident closed with terminal state

## Reporting

Record each job's exact root cause as a comment on **#521** (the single incident record). Do not open parallel CI-red issues.
