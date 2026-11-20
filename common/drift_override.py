"""Operator-recorded, dated overrides for the blocking drift gate.

The drift gate compares today's rankings with the last promoted snapshot and blocks
promotion on extreme churn, because extreme churn usually means a broken feed. An
authorized change to the production order (for example the 2026-11-13 switch to
``selector_only``) produces the same churn on purpose, and the gate then blocks every
run until one snapshot in the new order has been promoted.

An override entry in ``config/drift_gate_overrides.json`` lets that first run through.
It is deliberately narrow:

* it is pinned to the prior snapshot date the run is compared against, so it lapses by
  itself as soon as a snapshot in the new order is promoted;
* it only applies inside a short ``valid_from`` .. ``valid_through`` window;
* it only covers the metrics it names. Any other FAIL reason still blocks.

A matched override downgrades the gate from FAIL to WARN and is written into the gate
detail and ``drift_report.json``. It never hides the metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent.parent / "config" / "drift_gate_overrides.json"

# Longest window an entry may cover (calendar days, inclusive of both ends).
MAX_WINDOW_DAYS = 7

# Metrics an entry is allowed to cover. Top-60 overlap, rank correlation and the
# eligibility count are left out on purpose: an order change inside the eligible set
# does not move them, so a FAIL there is a data problem and must keep blocking.
#
# ``name_turnover_pct`` belongs to the separate ``turnover`` gate, which compares the Top-30
# with its own baseline (the newest earlier snapshot that is not cache-degraded). The same
# order change trips it, so an entry may cover it too.
OVERRIDABLE_METRICS = frozenset(
    {"top20_overlap_pct", "mean_abs_rank_delta_top60", "tier_migration_count", "name_turnover_pct"}
)

_REQUIRED_TEXT_FIELDS = ("id", "prior_date", "valid_from", "valid_through", "reason", "authorized_by", "memo")


@dataclass(frozen=True)
class DriftOverride:
    """One validated override entry."""

    id: str
    prior_date: str
    valid_from: str
    valid_through: str
    metrics: Tuple[str, ...]
    reason: str
    authorized_by: str
    memo: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prior_date": self.prior_date,
            "valid_from": self.valid_from,
            "valid_through": self.valid_through,
            "metrics": list(self.metrics),
            "reason": self.reason,
            "authorized_by": self.authorized_by,
            "memo": self.memo,
        }


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 10:
        return False
    try:
        from datetime import date

        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _days_between(start: str, end: str) -> int:
    from datetime import date

    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _parse_entry(raw: Any) -> Optional[DriftOverride]:
    """Return a validated entry, or None when the entry is malformed or too broad."""
    if not isinstance(raw, dict):
        return None
    for name in _REQUIRED_TEXT_FIELDS:
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            return None
    if not all(_is_iso_date(raw[name]) for name in ("prior_date", "valid_from", "valid_through")):
        return None
    span = _days_between(raw["valid_from"], raw["valid_through"])
    if span < 0 or span + 1 > MAX_WINDOW_DAYS:
        return None
    if raw["prior_date"] >= raw["valid_from"]:
        return None
    metrics = raw.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        return None
    if not all(isinstance(m, str) and m in OVERRIDABLE_METRICS for m in metrics):
        return None
    return DriftOverride(
        id=raw["id"].strip(),
        prior_date=raw["prior_date"],
        valid_from=raw["valid_from"],
        valid_through=raw["valid_through"],
        metrics=tuple(metrics),
        reason=raw["reason"].strip(),
        authorized_by=raw["authorized_by"].strip(),
        memo=raw["memo"].strip(),
    )


def load_drift_overrides(path: Optional[Path] = None) -> List[DriftOverride]:
    """Load valid override entries. A missing or unreadable file yields no overrides.

    Malformed and over-broad entries are dropped, so a bad edit can only make the gate
    stricter, never looser.
    """
    target = path if path is not None else DEFAULT_OVERRIDES_PATH
    try:
        with open(target, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return []
    entries = payload.get("overrides") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    parsed = [_parse_entry(raw) for raw in entries]
    return [entry for entry in parsed if entry is not None]


def fail_reason_metric(reason: str) -> str:
    """Metric name from a drift FAIL reason such as ``"FAIL top20_overlap_pct: 14.3% < 30.0%"``."""
    parts = reason.split()
    if len(parts) >= 2 and parts[0] == "FAIL":
        return parts[1].rstrip(":")
    return ""


def find_applicable_override(
    overrides: Sequence[DriftOverride],
    as_of_date: str,
    prior_date: str,
    fail_reasons: Sequence[str],
) -> Optional[DriftOverride]:
    """Return the override that covers every FAIL reason of this run, if there is one."""
    if not fail_reasons:
        return None
    failed = {fail_reason_metric(reason) for reason in fail_reasons}
    if "" in failed:
        return None
    for entry in overrides:
        if entry.prior_date != prior_date:
            continue
        if not entry.valid_from <= as_of_date <= entry.valid_through:
            continue
        if failed <= set(entry.metrics):
            return entry
    return None
