# Deprecated Module Files

This directory contains deprecated module variants that have been superseded by the canonical versions.

**Do not use these files in new code.** They are preserved only for reference during the transition period.

## Canonical Modules to Use

| Module | Canonical File | Purpose |
|--------|---------------|---------|
| Module 1 | `module_1_universe.py` | Universe filtering |
| Module 2 | `module_2_financial_v2.py` | Financial health scoring |
| Module 3 | `module_3_catalyst.py` | Catalyst event detection |
| Module 4 | `module_4_clinical_dev_v2.py` | Clinical development scoring |
| Module 5 | `module_5_composite_v2.py` | Composite ranking |

## Why These Were Deprecated

These files were backup/variant versions created during iterative development:
- `*_ORIGINAL.py` - Pre-enhancement versions
- `*_FIXED.py` - Bug fix iterations
- `*_ENHANCED.py` - Feature enhancement iterations
- `*_BUCKETED.py` - Experimental bucketing approaches
- `*_CORRECTED.py` - Correction iterations
- `*_CUSTOM.py` - One-off customizations
- `*.backup.py` - Manual backups

The canonical `_v2.py` versions incorporate all necessary fixes and enhancements.

## Migration Guide

If your code imports any of these files:

```python
# OLD (deprecated)
from module_2_financial_ENHANCED import run_module_2

# NEW (canonical)
from module_2_financial_v2 import run_module_2_v2
```

## Deletion Schedule

These files will be permanently deleted in the next major version (v2.0).
