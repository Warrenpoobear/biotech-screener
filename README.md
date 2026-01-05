# Wake Robin Biotech Alpha System - AACT Integration

Point-in-time safe, deterministic biotech investment screening with clinical trials data from AACT.

## Quick Start

### 1. Run Stub Mode (Baseline)

```bash
python -m src.snapshot_generator --as-of 2024-01-31 \
  --universe data/universe/biotech_universe_v1.csv \
  --output snapshots/stub
```

### 2. Run AACT Mode (Real Clinical Trials Data)

```bash
python -m src.snapshot_generator --as-of 2024-01-31 \
  --universe data/universe/biotech_universe_v1.csv \
  --clinical-provider aact \
  --aact-snapshots data/aact_snapshots \
  --trial-map data/trial_mapping.csv \
  --output snapshots/aact
```

### 3. Enable Diff Computation (PCD Pushes / Status Flips)

```bash
python -m src.snapshot_generator --as-of 2024-01-31 \
  --universe data/universe/biotech_universe_v1.csv \
  --clinical-provider aact \
  --aact-snapshots data/aact_snapshots \
  --trial-map data/trial_mapping.csv \
  --aact-enable-diffs \
  --output snapshots/aact_with_diffs
```

### 4. Compare Snapshots

```bash
python scripts/compare_snapshots.py \
  --baseline snapshots/stub/2024-01-31.json \
  --compare snapshots/aact/2024-01-31.json \
  --output output/compare/2024-01-31_stub_vs_aact/
```

## Architecture

### Provider Pattern

All data providers implement a common protocol with PIT boundary enforcement:

```python
class ClinicalTrialsProvider(Protocol):
    def get_trials_as_of(
        self,
        as_of_date: date,
        pit_cutoff: date,
        tickers: list[str],
        trial_mapping: dict[str, list[str]],
    ) -> ProviderResult:
        """Return trials per ticker, filtered to PIT-safe snapshot <= pit_cutoff."""
```

The provider owns the PIT boundary, so downstream modules stay pure and deterministic.

### PIT Safety Rules

- `pit_cutoff = as_of_date - pit_lag_days` (default: 1 day)
- Strict mode: snapshot date must be **< pit_cutoff** (never same-day)
- Non-strict mode: snapshot date can be **<= pit_cutoff**

### Diff Computation Control

Diff computation (pcd_pushes_18m, status_flips_18m) is **disabled by default** to keep Phase 1 simple.

- `--aact-enable-diffs`: Enable diff computation (requires 2+ snapshots)
- Provenance tracks: `compute_diffs_enabled`, `compute_diffs_available`, `snapshots_available_count`
- Trial flags distinguish:
  - `diffs_disabled`: Diffs were not requested via config
  - `diffs_unavailable_insufficient_snapshots`: Diffs requested but <2 snapshots available

### AACT Snapshot Structure

```
data/aact_snapshots/
├── 2024-01-15/
│   ├── studies.csv
│   └── sponsors.csv
└── 2024-01-29/
    ├── studies.csv
    └── sponsors.csv
```

**Required columns in `studies.csv`:**
- `nct_id`, `phase`, `overall_status`
- `primary_completion_date`, `primary_completion_date_type`
- `last_update_posted_date`, `study_type`

**Required columns in `sponsors.csv`:**
- `nct_id`, `name`, `lead_or_collaborator`

**Schema validation**: Missing required columns will raise `ValueError` with clear error message.

### AACT Extract SQL (Recommended)

Use `\copy` for client-side export with proper quoting:

```sql
\copy (
  SELECT 
    nct_id, study_type, phase, overall_status,
    primary_completion_date, primary_completion_date_type,
    last_update_posted_date
  FROM ctgov.studies
  WHERE study_type = 'Interventional'
) TO 'studies.csv'
WITH (FORMAT csv, HEADER true, QUOTE '"', FORCE_QUOTE *);

\copy (
  SELECT nct_id, name, lead_or_collaborator
  FROM ctgov.sponsors
  WHERE lead_or_collaborator = 'lead'
) TO 'sponsors.csv'
WITH (FORMAT csv, HEADER true, QUOTE '"', FORCE_QUOTE *);
```

`FORCE_QUOTE *` ensures all fields are quoted, eliminating comma ambiguity.

### Trial Mapping

The `trial_mapping.csv` connects tickers to NCT IDs:

| ticker | nct_id | effective_start | effective_end | source | sponsor_name_at_map_time | mapping_confidence |
|--------|--------|-----------------|---------------|--------|--------------------------|-------------------|
| MRNA | NCT04470427 | 2020-07-01 | | company_ir | Moderna TX Inc | high |
| MRNA | NCT04860297 | 2021-01-01 | | clinicaltrials.gov | Moderna TX Inc | high |

This is the **Option A (cleanest)** approach—deterministic, auditable, no fuzzy matching.

## Output Structure

### Snapshot JSON

```json
{
  "snapshot_id": "sha256:abc123...",
  "as_of_date": "2024-01-31",
  "pit_cutoff": "2024-01-30",
  "pit_lag_days": 1,
  "provenance": {
    "pit_cutoff": "2024-01-30",
    "providers": {
      "clinical": {
        "name": "aact",
        "snapshot_date_used": "2024-01-29",
        "snapshots_root": "data/aact_snapshots",
        "compute_diffs_enabled": true,
        "compute_diffs_available": true,
        "snapshots_available_count": 2
      }
    }
  },
  "input_hashes": {
    "universe": "sha256:...",
    "trial_facts": "sha256:..."
  },
  "coverage": {
    "catalyst": {
      "tickers_total": 10,
      "tickers_with_trials": 8,
      "coverage_rate": 0.8
    }
  },
  "tickers": {
    "MRNA": {
      "trials": [
        {
          "nct_id": "NCT04470427",
          "phase": "P2",
          "overall_status": "recruiting",
          "pcd_pushes_18m": 1,
          "status_flips_18m": 0,
          "flags": []
        }
      ],
      "trial_count": 2
    }
  }
}
```

### Determinism Guarantee

Same inputs → identical `snapshot_id` → identical snapshot content.

### Trial Flags

Flags track data quality and processing state:
- `pcd_missing`: Primary completion date not present
- `pcd_type_missing`: PCD type not present or unknown
- `pcd_parse_error`: PCD could not be parsed as a date
- `last_update_missing`: Last update posted date not present
- `diffs_disabled`: Diff computation was disabled via config
- `diffs_unavailable_insufficient_snapshots`: Not enough snapshots for diffs

## Testing

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Test Categories (63 tests)

1. **Provider Unit Tests** (`tests/providers/test_aact_provider.py`)
   - Snapshot selection with PIT constraints
   - Canonical TrialRow serialization and hashing
   - Diff counts (PCD pushes, status flips)

2. **CSV Parsing Tests** (`tests/providers/test_csv_parsing.py`)
   - Commas in quoted fields
   - Quotes inside quoted fields ("evil" rows)
   - Phase/status/PCDType normalization (parametrized)

3. **Schema Validation Tests** (`tests/providers/test_schema_validation.py`)
   - Missing required columns raise clear errors
   - Duplicate NCT IDs keep row with latest update
   - Flag emission for missing data and diff unavailability

4. **Integration Tests** (`tests/integration/test_snapshot_integration.py`)
   - Snapshot contains coverage + provenance
   - A/B compare handles missing trials gracefully

## Data Integrity Hardening

### Duplicate Key Handling

If duplicate NCT IDs appear in a snapshot (shouldn't happen, but can with bad extracts):
- **Rule**: Keep row with latest `last_update_posted_date`
- **Logging**: Warning logged with count of duplicates found

### CSV Parsing Safety

- Uses Python `csv.DictReader` with `newline=""` for platform compatibility
- Properly handles RFC 4180 quoting (doubled quotes for escape)
- Schema validation fails fast with clear error messages

## Phase 2: AACT Diffs + Cache (Future)

Once Phase 1 is stable:

1. **Cache diffs as Parquet artifacts**:
   ```
   data/curated/aact_diffs/{prev_snapshot}_{snapshot}.parquet
   ```

2. **Provider behavior**:
   - If diff artifact exists → load it
   - Else compute diff once → write artifact → use it

3. **Expand to monthly as_of dates**

## Troubleshooting

### "Coverage is near zero"

Usually one of:
- Mapping file missing or wrong path
- Snapshot date selection picks older snapshot without relevant trials
- AACT snapshot extracts filtered incorrectly

### "Hashes change between runs"

Usually one of:
- Nondeterministic ordering (dict iteration, set ordering)
- Trial lists not sorted
- Flags not sorted
- Timestamps accidentally included in hashed payload

### "CSV parse errors"

Usually one of:
- Unquoted commas in string fields (use `FORCE_QUOTE *` in extract)
- Inconsistent encoding
- Mixed delimiters

### "ValueError: Missing required columns"

Your AACT extract is missing required columns. Check the extract SQL and ensure all columns are present.

## Changes Between Snapshots (Sample Data)

The sample data includes changes between 2024-01-15 and 2024-01-29:

| NCT ID | Change Type | Details |
|--------|-------------|---------|
| NCT04470427 | PCD Push | 2024-06-15 → 2024-09-15 (3 months) |
| NCT04368728 | PCD Push | 2024-07-01 → 2024-10-01 (3 months) |
| NCT03872479 | Status Flip | Active → Suspended |
| NCT04488081 | Status Flip | Suspended → Recruiting |
| NCT05105568 | PCD Push | 2025-03-01 → 2025-06-01 (3 months) |
