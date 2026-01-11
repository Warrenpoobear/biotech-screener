# Governance and Lineage Tracking

This document describes the governance, audit, and lineage tracking system for the Alpha Engine.

## Overview

The governance system ensures:
1. **Determinism**: Same inputs always produce byte-identical outputs
2. **Auditability**: Complete machine-readable audit trail for every run
3. **Reproducibility**: Any historical run can be reproduced exactly
4. **Schema Safety**: Schema changes are versioned and validated

## Key Concepts

### Run ID

A `run_id` is a deterministic identifier computed from:
- `as_of_date`: Analysis date
- `score_version`: Version of scoring parameters
- `parameters_hash`: SHA256 hash of parameters
- `input_hashes`: Sorted list of input file hashes
- `mapping_hashes`: Sorted list of adapter mapping hashes
- `pipeline_version`: Version of pipeline code

**Important**: Same inputs always produce the same `run_id`. There are no timestamps or UUIDs.

```python
from governance import compute_run_id

run_id = compute_run_id(
    as_of_date="2024-01-15",
    score_version="v1",
    parameters_hash="abc123...",
    input_hashes=[{"path": "data.json", "sha256": "..."}],
    pipeline_version="2.0.0",
)
```

### Parameters Archive

Scoring parameters are stored in versioned files under `params_archive/`:

```
params_archive/
  v1.json    # Initial release parameters
  v2.json    # Future version
```

Each file contains all tunable parameters. The `parameters_hash` is computed from the canonical JSON representation.

To use a specific version:
```bash
python run_pipeline.py --score-version v1 --as-of 2024-01-15
```

### Adapter Mappings

Field mappings between source schemas and canonical schemas are stored under `adapters/`:

```
adapters/
  screener/
    mapping_v1.json
  holdings/
    mapping_v1.json
```

Mappings define:
- Required fields (fail-closed if missing)
- Field name translations
- Source-specific validation rules

### Audit Log

Every run produces an `audit.jsonl` file with one record per stage:

```jsonl
{"run_id":"abc123","stage_name":"INIT","status":"OK",...}
{"run_id":"abc123","stage_name":"LOAD","status":"OK",...}
{"run_id":"abc123","stage_name":"SCORE","status":"OK",...}
{"run_id":"abc123","stage_name":"FINAL","status":"OK",...}
```

Each record includes:
- `run_id`: Deterministic run identifier
- `stage_name`: Pipeline stage (INIT, LOAD, ADAPT, FEATURES, RISK, SCORE, REPORT, FINAL)
- `status`: OK, FAIL, or SKIP
- `score_version`, `schema_version`, `parameters_hash`
- `stage_inputs`, `stage_outputs`: File references with hashes
- `error_code`, `error_message`: On failure only

## Reproducing Historical Runs

To reproduce a historical run:

1. **Identify the run**: Find the `run_id` in the audit log
2. **Check parameters**: Load the same `score_version` from params_archive
3. **Verify inputs**: Ensure input files match the recorded hashes
4. **Run pipeline**: Use the same `--as-of` date and `--score-version`

```bash
# Original run produced run_id: abc123...
# Audit log shows: score_version=v1, as_of=2024-01-15

# To reproduce:
python run_pipeline.py \
  --as-of 2024-01-15 \
  --score-version v1 \
  --data-dir ./data
```

If inputs and parameters match, you'll get:
- Identical `run_id`
- Byte-identical outputs
- Identical output hashes

## Schema Evolution

### Adding New Fields

1. Add field to output without changing existing fields
2. Bump minor version (e.g., 1.0.0 → 1.1.0)
3. Old consumers ignore new fields (forward compatible)

### Changing Existing Fields

1. Create new `score_version` (e.g., v2)
2. Add new params file to `params_archive/v2.json`
3. Keep old version working for reproducibility
4. Document migration in CHANGELOG

### Breaking Changes

1. Bump major version (e.g., 1.0.0 → 2.0.0)
2. Old data may not validate against new schema
3. Provide migration script if needed

## Error Handling

The system is **fail-closed**: any validation failure stops the run and records an error.

### Error Codes

| Code | Description |
|------|-------------|
| `MISSING_INPUT` | Required input file not found |
| `SCHEMA_MISMATCH` | Source data missing required fields |
| `HASH_ERROR` | File hash verification failed |
| `PARAMS_MISSING` | Parameters file not found |
| `MAPPING_MISSING` | Adapter mapping file not found |
| `VALIDATION_ERROR` | Data validation failed |

### Example Failure Record

```json
{
  "run_id": "abc123",
  "stage_name": "LOAD",
  "status": "FAIL",
  "error_code": "MISSING_INPUT",
  "error_message": "Input file not found: market_data.json"
}
```

## Output Format

All outputs include governance metadata:

```json
{
  "_governance": {
    "run_id": "abc123...",
    "score_version": "v1",
    "schema_version": "1.0.0",
    "parameters_hash": "def456...",
    "input_lineage": [
      {"path": "market.json", "sha256": "...", "as_of_date": "2024-01-15"}
    ],
    "generation_metadata": {
      "pipeline_version": "2.0.0",
      "tool_name": "biotech-screener"
    }
  },
  "results": { ... }
}
```

## Canonical JSON

All outputs use canonical JSON serialization:
- Dict keys sorted recursively
- Floats formatted without scientific notation
- NaN and Infinity forbidden (raise error)
- Lists preserve order (caller must sort if needed)
- Trailing newline

This ensures byte-identical outputs for identical inputs.

## CLI Reference

```bash
# Standard run
python run_pipeline.py \
  --as-of 2024-01-15 \
  --score-version v1 \
  --data-dir ./data \
  --output ./output

# Dry run (shadow output, still produces audit)
python run_pipeline.py \
  --as-of 2024-01-15 \
  --score-version v1 \
  --data-dir ./data \
  --output ./output \
  --dry-run

# Validate inputs only
python run_pipeline.py \
  --as-of 2024-01-15 \
  --data-dir ./data \
  --dry-run
```

## Module Reference

### governance.canonical_json
- `canonical_dumps(obj)`: Serialize to canonical JSON string
- `canonical_dump(obj, fp)`: Write canonical JSON to file

### governance.hashing
- `hash_file(path)`: SHA256 hash of file contents
- `hash_bytes(data)`: SHA256 hash of bytes
- `hash_canonical_json(obj)`: SHA256 hash of canonical JSON

### governance.run_id
- `compute_run_id(...)`: Generate deterministic run ID

### governance.audit_log
- `AuditLog`: Audit log writer class
- `AuditStage`: Pipeline stage enum
- `AuditStatus`: OK/FAIL/SKIP enum

### governance.params_loader
- `load_params(score_version)`: Load and hash parameters
- `compute_parameters_hash(params)`: Compute params hash

### governance.mapping_loader
- `load_mapping(source, version)`: Load adapter mapping
- `validate_source_schema(data, mapping)`: Validate required fields

## Testing

Run governance tests:
```bash
pytest tests/test_governance.py -v
```

Key test cases:
- Canonical JSON determinism
- Hash consistency across runs
- Run ID stability
- Audit log structure
- Parameter change detection
