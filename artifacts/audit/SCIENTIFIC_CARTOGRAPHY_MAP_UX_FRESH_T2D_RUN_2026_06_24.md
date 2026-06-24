# Scientific Cartography Map UX — Fresh T2D Run + Fork-Agent Allowlist
**Date:** 2026-06-24
**Status:** PASS
**Verdict:** `PASS_MAP_UX_FRESH_T2D_RUN_REPRODUCIBLE`

---

## 1. Fresh T2D Map Generation

Regenerated the T2D poster map on this VM using production `trial_records.json`
(20,057 trials) and Map UX v0.3 generator.

### Commands

```bash
# Step 1: diagnostics (cache-only, no rankings.csv)
mkdir -p artifacts/scientific_cartography/scratch-2026-06-24/{ctgov_cache,snapshot}
ln -sf "$(pwd)/production_data/trial_records.json" \
  artifacts/scientific_cartography/scratch-2026-06-24/ctgov_cache/trial_records.json

python3 tools/run_scientific_cartography_diagnostics.py \
  --as-of-date 2026-06-24 \
  --snapshot-dir artifacts/scientific_cartography/scratch-2026-06-24/snapshot \
  --ctgov-cache artifacts/scientific_cartography/scratch-2026-06-24/ctgov_cache \
  --output-dir artifacts/scientific_cartography/scratch-2026-06-24

# Step 2: map UX v0.3
python3 tools/generate_scientific_cartography_map.py \
  --input-dir artifacts/scientific_cartography/scratch-2026-06-24 \
  --disease "type 2 diabetes mellitus" \
  --output-dir artifacts/scientific_cartography/map_ux/type-2-diabetes-mellitus
```

### Output files (local, gitignored)

| File | Size | Path |
|------|------|------|
| `index.html` | 53,487 bytes | `artifacts/scientific_cartography/map_ux/type-2-diabetes-mellitus/index.html` |
| `map.svg` | 41,313 bytes | `artifacts/scientific_cartography/map_ux/type-2-diabetes-mellitus/map.svg` |
| `map.json` | 183,799 bytes | `artifacts/scientific_cartography/map_ux/type-2-diabetes-mellitus/map.json` |
| `README.md` | 1,043 bytes | `artifacts/scientific_cartography/map_ux/type-2-diabetes-mellitus/README.md` |

Open locally: `file://<repo>/artifacts/scientific_cartography/map_ux/type-2-diabetes-mellitus/index.html`

### Pipeline counts

| Stage | Count |
|-------|-------|
| Raw CT.gov trials loaded | 20,057 |
| Total program records built | 73,075 |
| T2D disease-filtered (raw) | 551 |
| Non-drug filtered (D3) | 37 |
| After D1 dedup (unique asset+company) | 335 |
| Canonicalization merge groups | 9 |

### Map summary (`map.json`)

| Field | Value |
|-------|-------|
| `metadata.generator_version` | `v0.3` |
| `metadata.as_of_date` | `2026-06-24` |
| `summary.total_programs` | 335 |
| `summary.mechanism_lane_count` | 17 |
| `summary.stage_coverage_pct` | 74.0% |
| `summary.mechanism_coverage_pct` | 49.3% |
| `summary.ticker_coverage_pct` | 0.0% (expected — no rankings.csv) |
| Stage distribution | unknown: 87, phase3: 82, phase2: 61, phase1: 57, approved: 48 |

### Named mechanism lanes (16 + Unknown)

Insulin, Biguanide, GLP-1 receptor agonist, PPAR agonist, Glucokinase activator,
DPP-4 inhibitor, SGLT2 inhibitor, Sulfonylurea, GCGR antisense oligonucleotide,
GLP-1/Insulin fixed-ratio combination, SGLT2/SGLT1 inhibitor, GLP-1/GCGR dual agonist,
Meglitinide, Alpha-glucosidase inhibitor, Amylin analog, Unknown Mechanism.

### Delta vs 2026-06-23 visual QA

| Metric | 2026-06-23 QA | 2026-06-24 run | Notes |
|--------|---------------|----------------|-------|
| Programs | 335 | 335 | Same after D1/D3 |
| Mechanism lanes | 11 | 17 | Built-in normalizer only (alias pack CSV absent) |
| Mechanism coverage | 23.0% | 49.3% | Built-in dict broader than alias pack v0.1 |
| Stage coverage | 74.0% | 74.0% | Stable |

**Note:** `scientific_cartography/data/mechanism_aliases_v0_1.csv` is referenced by
diagnostics but not present in this workspace checkout. The 2026-06-23 QA run loaded
alias pack v0.1; this run used built-in `MechanismNormalizer` only. Lane counts and
coverage percentages are not directly comparable until alias pack is restored.

### Warnings emitted

1. 37 non-pharmaceutical programs filtered (exercise, tobacco cessation, eating-window comparisons).
2. 9 asset-name variant groups merged (saxagliptin dose forms, dapagliflozin dose forms).

---

## 2. Governance

- READ_ONLY_DIAGNOSTIC ✓
- No `rankings.csv`, `portfolio_positions.csv`, or scoring files read ✓
- `ForbiddenSourceError` guard active in map generator ✓
- Freeze ACTIVE — no ranker/selector/sizing/final_score changes ✓
- Generated HTML contains required governance language ✓

---

## 3. Fork-Agent Allowlist Prompt (Map UX v0.4+)

Use this prompt verbatim when launching a fork agent for Map UX work.
Derived from scope hygiene audit `b8f4c66b` (2026-06-23 scope creep on `20da2dd0`).

```
TASK: Scientific Cartography Map UX — [describe specific version/change]

GOVERNANCE TIER: Tier 2 (diagnostic-only, no production model changes)

ALLOWED FILES — you may ONLY create or modify these paths:
  - tools/generate_scientific_cartography_map.py
  - tests/scientific_cartography/test_map_generator.py
  - artifacts/audit/SCIENTIFIC_CARTOGRAPHY_MAP_UX_*.md

FORBIDDEN — do NOT touch, stage, or commit:
  - run_screen.py, decision_engine.py, ranker_engine.py, selector_engine.py
  - Any file under src/ related to scoring, sizing, portfolio, snapshots
  - tools/run_agent_direct.py, tools/record_skill_feedback.py
  - Any selfimprove, Hermes, MCP, cron, or React frontend files
  - rankings.csv, portfolio_positions.csv, screen_output.json (must not be read)

COMMIT RULES:
  1. Before EVERY commit, run: git diff --name-only --cached
  2. If ANY file outside ALLOWED FILES appears, run: git reset HEAD <file>
  3. If unexpected modified files exist in git status, STOP and report — do not commit
  4. One commit per logical change; message must start with "sci-cart map UX"
  5. Do not commit generated map artifacts (artifacts/scientific_cartography/map_ux/)

PRE-FLIGHT:
  1. pytest tests/scientific_cartography/test_map_generator.py -p no:warnings
  2. Confirm HTML output still contains "DIAGNOSTIC ONLY" and "NOT AN INVESTMENT RECOMMENDATION"
  3. Confirm no external CDN references in generated HTML

SCOPE:
  - Layout/rendering and data QC only unless explicitly authorized otherwise
  - Do not add ticker linkage, MCP wiring, cron, or production integration
  - Do not modify mechanism alias packs unless that is the named task

DELIVERABLES:
  - Code + tests (if applicable)
  - Audit memo in artifacts/audit/SCIENTIFIC_CARTOGRAPHY_MAP_UX_<VERSION>_<DATE>.md
  - Regenerated T2D map path reported in memo (do not commit HTML/SVG/JSON)
```

---

## 4. Recommended Next Steps

1. Restore `scientific_cartography/data/mechanism_aliases_v0_1.csv` to workspace
   (referenced by tests and diagnostics; missing in current checkout).
2. Re-run diagnostics with alias pack loaded; compare lane counts to this run.
3. Use fork-agent allowlist above for any v0.4 layout or density refinement work.
4. Optional: add golden-file determinism test for `generate_map()` output hash.
