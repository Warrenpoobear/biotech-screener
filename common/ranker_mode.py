"""Which ordering the production screen applies inside the eligible set.

Operator decision 2026-11-13: the default moved from ``pairwise_minimal`` to
``selector_only``. The pairwise ranker's negative ``financial_score`` weight was fitted
on history that read today's market cap on every past date; with point-in-time market
cap ``financial_score`` shows no relation to next-period returns and the selector order
alone did better. See docs/governance/2026-11-13-selector-only-production-switch.md.

Rollback: ``--ranker-mode pairwise_minimal``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

RANKER_MODES: Tuple[str, ...] = ("selector_only", "pairwise_minimal", "clinical_50")
DEFAULT_RANKER_MODE = "selector_only"


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if out == out else 0.0  # NaN -> 0.0


def apply_selector_only_ranking(
    eligible_rows: List[Dict[str, Any]],
    shadow_v2_by_ticker: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> int:
    """Order the eligible set by ``selector_score`` alone.

    Sets ``final_score = selector_score`` on every eligible row, so the sort anchor is
    the selector for the whole eligible set and nothing reorders the top-60 cohort.
    ``ranker_active`` is "0" on every row. No ranker output, and no options field,
    reaches ``final_score`` in this mode.

    The pairwise ranker's score and rank are kept in ``ranker_v2_score`` /
    ``ranker_v2_rank`` as a shadow record when they were computed; they do not affect
    the order. Returns the number of rows that carry a shadow ranker score.
    """
    shadow = shadow_v2_by_ticker or {}
    n_shadow = 0
    for row in eligible_rows:
        row["ranker_active"] = "0"
        row["ranker_adjustment"] = 0.0
        row["final_score"] = _to_float(row.get("selector_score"))
        v2 = shadow.get(str(row.get("ticker", "")))
        if v2 and v2.get("ranker_v2_score") is not None:
            row["ranker_v2_score"] = v2["ranker_v2_score"]
            row["ranker_v2_rank"] = v2.get("ranker_v2_rank", "")
            n_shadow += 1
        else:
            row["ranker_v2_score"] = ""
            row["ranker_v2_rank"] = ""
        row["ranker_options_block"] = ""
        row["ranker_inst_block"] = ""
        row["ranker_aact_block"] = ""
    return n_shadow
