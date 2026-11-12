"""Registry-transition guard must neutralize inst_delta_z, not only the diagnostics.

On 2026-11-03 the current institutional summary had 30 roster managers and the
prior (2026-08-13) had 49. The guard blanked the inst_flow_* diagnostic columns
but left inst_delta_z live: 189/298 tickers showed net exits (mean net -1.34)
that were roster removals, not selling, and inst_delta_z feeds the decision
engine's institutional sort contribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_screen import _apply_inst_registry_transition_guard  # noqa: E402


def _rows():
    return [
        {"ticker": "AAA", "inst_delta_z": -1.9161, "inst_delta_net": -3},
        {"ticker": "BBB", "inst_delta_z": 3.1579, "inst_delta_net": 2},
    ]


def test_transition_zeroes_inst_delta_z():
    rows = _rows()
    transition = _apply_inst_registry_transition_guard(
        rows, {"elite_managers_with_filing": 30}, {"elite_managers_with_filing": 49}
    )
    assert transition is True
    assert [r["inst_delta_z"] for r in rows] == [0.0, 0.0]
    # Raw counts are kept for audit; only the model-facing z is neutralized.
    assert [r["inst_delta_net"] for r in rows] == [-3, 2]


def test_same_roster_size_leaves_values():
    rows = _rows()
    transition = _apply_inst_registry_transition_guard(
        rows, {"elite_managers_with_filing": 30}, {"elite_managers_with_filing": 30}
    )
    assert transition is False
    assert [r["inst_delta_z"] for r in rows] == [-1.9161, 3.1579]


def test_missing_prior_is_not_a_transition():
    rows = _rows()
    assert _apply_inst_registry_transition_guard(rows, {"elite_managers_with_filing": 30}, None) is False
    assert _apply_inst_registry_transition_guard(rows, None, {"elite_managers_with_filing": 49}) is False
    assert [r["inst_delta_z"] for r in rows] == [-1.9161, 3.1579]
