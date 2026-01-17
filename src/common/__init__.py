"""
Common utilities for Wake Robin Biotech Alpha System.

DEPRECATED: This module is deprecated. Import from 'common' instead.

    # OLD (deprecated)
    from src.common import compute_hash

    # NEW
    from common import compute_hash
"""
import warnings

warnings.warn(
    "Importing from 'src.common' is deprecated. "
    "Use 'from common import ...' instead. "
    "This module will be removed in v2.0.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from canonical location for backwards compatibility
from common.hash_utils import (
    compute_hash,
    compute_snapshot_id,
    compute_trial_facts_hash,
    stable_json_dumps,
)

__all__ = [
    "compute_hash",
    "compute_snapshot_id",
    "compute_trial_facts_hash",
    "stable_json_dumps",
]
