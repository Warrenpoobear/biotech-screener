"""Corporate action registry — authoritative source for splits, renames, acquisitions, delistings.

Loads ``production_data/corporate_actions.json`` and provides lookup functions
for use across the pipeline: price adjustment, universe filtering, 13F mapping,
event continuity, and backtest panel construction.

PIT safety:
  - All lookups accept an ``as_of`` parameter.
  - Actions with ``effective_date > as_of`` are invisible to the caller.
  - This prevents retroactive application of present-day knowledge.

Usage::

    from common.corporate_actions import load_actions, get_splits, is_dead, resolve_ticker

    actions = load_actions()
    splits = get_splits("AKTX", actions)
    dead = is_dead("CNTA", "2026-04-01", actions)
    current = resolve_ticker("BGNE", "2025-06-01", actions)  # → "ONC"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "production_data" / "corporate_actions.json"

# Valid action types
ACTION_TYPES = frozenset(
    {
        "reverse_split",
        "forward_split",
        "acquisition",
        "pending_acquisition",
        "delisted",
        "ticker_change",
        "spinoff",
        "bankruptcy",
    }
)

DEAD_ACTION_TYPES = frozenset({"acquisition", "delisted", "bankruptcy"})
"""Action types that mean the security no longer trades.

``pending_acquisition`` is deliberately NOT here: an announced-but-unclosed deal
still trades, so the ticker must stay in the universe.  It is a recognised action
type only so the registry stops silently dropping it — see the loader warning
that hid APGE's AbbVie deal until 2026-08-05.
"""


@dataclass(frozen=True)
class CorporateAction:
    """Single corporate action event."""

    ticker: str
    action: str
    effective_date: str  # ISO date. For pending_acquisition (no close yet) the
    # loader substitutes announced_date so PIT filtering still works — never None.
    # Split fields
    ratio: Optional[str] = None
    factor: Optional[float] = None
    # Acquisition fields
    acquirer: Optional[str] = None
    deal_price: Optional[float] = None
    announced_date: Optional[str] = None
    # Rename fields
    old_ticker: Optional[str] = None
    new_ticker: Optional[str] = None
    # General
    notes: str = ""


@dataclass
class CorporateActionRegistry:
    """Loaded registry of all corporate actions."""

    actions: List[CorporateAction] = field(default_factory=list)
    _by_ticker: Dict[str, List[CorporateAction]] = field(default_factory=dict, repr=False)
    _rename_map: Dict[str, List[CorporateAction]] = field(default_factory=dict, repr=False)

    def _build_indices(self) -> None:
        from collections import defaultdict

        by_ticker: Dict[str, List[CorporateAction]] = defaultdict(list)
        rename_map: Dict[str, List[CorporateAction]] = defaultdict(list)
        for a in self.actions:
            by_ticker[a.ticker].append(a)
            if a.action == "ticker_change" and a.old_ticker:
                rename_map[a.old_ticker].append(a)
        self._by_ticker = dict(by_ticker)
        self._rename_map = dict(rename_map)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_actions(path: Optional[Path] = None) -> CorporateActionRegistry:
    """Load the corporate actions registry from JSON.

    Returns an empty registry if the file is missing (fail-open for
    environments without the data file).
    """
    p = path or _DEFAULT_PATH
    if not p.exists():
        logger.warning("Corporate actions file not found: %s", p)
        return CorporateActionRegistry()

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    raw_actions = data.get("actions", [])
    actions = []
    for raw in raw_actions:
        action_type = raw.get("action", "")
        if action_type not in ACTION_TYPES:
            logger.warning(
                "Unknown action type %r for %s — skipping",
                action_type,
                raw.get("ticker"),
            )
            continue
        # effective_date may be absent OR explicitly null (a pending deal has no
        # close date yet), and `.get(k, "")` does NOT protect against an explicit
        # null. A None here propagates into `a.effective_date > as_of` and
        # `sorted(key=effective_date)`, both of which raise TypeError against str.
        # Fall back to announced_date so PIT filtering stays correct: an announced
        # deal is knowable from its announcement, not from the beginning of time.
        announced = raw.get("announced_date") or None
        effective = raw.get("effective_date") or announced or ""
        actions.append(
            CorporateAction(
                ticker=raw.get("ticker", ""),
                action=action_type,
                effective_date=effective,
                ratio=raw.get("ratio"),
                factor=raw.get("factor"),
                acquirer=raw.get("acquirer"),
                deal_price=raw.get("deal_price"),
                announced_date=announced,
                old_ticker=raw.get("old_ticker"),
                new_ticker=raw.get("new_ticker"),
                notes=raw.get("notes", ""),
            )
        )

    registry = CorporateActionRegistry(actions=actions)
    registry._build_indices()
    logger.info(
        "Loaded %d corporate actions (%d tickers)",
        len(actions),
        len(registry._by_ticker),
    )
    return registry


# ---------------------------------------------------------------------------
# Query functions — all PIT-safe via as_of parameter
# ---------------------------------------------------------------------------


def get_actions(
    ticker: str,
    registry: CorporateActionRegistry,
    *,
    as_of: Optional[str] = None,
    action_type: Optional[str] = None,
) -> List[CorporateAction]:
    """Return actions for a ticker, optionally filtered by date and type.

    Only returns actions with ``effective_date <= as_of`` when as_of is given.
    """
    candidates = registry._by_ticker.get(ticker, [])
    results = []
    for a in candidates:
        # effective_date is normalised to a str by the loader, but stay defensive:
        # comparing None to a str raises TypeError and would take down the caller.
        eff = a.effective_date or ""
        if as_of and eff > as_of:
            continue
        if action_type and a.action != action_type:
            continue
        results.append(a)
    return sorted(results, key=lambda a: a.effective_date or "")


def get_splits(
    ticker: str,
    registry: CorporateActionRegistry,
    *,
    as_of: Optional[str] = None,
) -> List[CorporateAction]:
    """Return all split events (forward + reverse) for a ticker."""
    return get_actions(
        ticker,
        registry,
        as_of=as_of,
        action_type=None,
    )


def get_splits_only(
    ticker: str,
    registry: CorporateActionRegistry,
    *,
    as_of: Optional[str] = None,
) -> List[CorporateAction]:
    """Return only split events (forward_split or reverse_split)."""
    results = []
    for a in get_actions(ticker, registry, as_of=as_of):
        if a.action in ("forward_split", "reverse_split"):
            results.append(a)
    return results


def is_dead(
    ticker: str,
    as_of: str,
    registry: CorporateActionRegistry,
) -> bool:
    """Return True if the ticker is acquired, delisted, or bankrupt as of the given date."""
    for a in get_actions(ticker, registry, as_of=as_of):
        if a.action in DEAD_ACTION_TYPES:
            return True
    return False


def is_pending_deal(
    ticker: str,
    as_of: str,
    registry: CorporateActionRegistry,
) -> bool:
    """True if the ticker has an announced-but-unclosed acquisition as of ``as_of``.

    Such a ticker still trades, so :func:`is_dead` is False and it stays in the
    universe — but its price is pinned near the deal price and carries essentially
    no remaining alpha, so ranking it against live names is misleading.

    NOT wired into any gate yet.  The registry's own APGE note says "Gate from
    ranker/selector until acquisition closes", but acting on that changes the
    scored cohort and is an operator decision under the NO_MODEL_CHANGE window.
    Exposed here so the gate can be wired deliberately rather than by accident.
    """
    for a in get_actions(ticker, registry, as_of=as_of):
        if a.action == "pending_acquisition":
            return True
    return False


def death_date(
    ticker: str,
    registry: CorporateActionRegistry,
) -> Optional[str]:
    """Return the effective date of the first death event, or None."""
    for a in registry._by_ticker.get(ticker, []):
        if a.action in DEAD_ACTION_TYPES:
            return a.effective_date
    return None


def resolve_ticker(
    ticker: str,
    as_of: str,
    registry: CorporateActionRegistry,
) -> str:
    """Resolve a possibly-old ticker to the current ticker as of a date.

    Follows the rename chain forward: if BGNE renamed to ONC on 2025-01-02,
    then ``resolve_ticker("BGNE", "2025-06-01", reg)`` returns ``"ONC"``.

    Returns the input ticker if no rename is found.
    """
    renames = registry._rename_map.get(ticker, [])
    for r in sorted(renames, key=lambda a: a.effective_date):
        if r.effective_date <= as_of and r.new_ticker:
            # Follow the chain (new_ticker might also have been renamed)
            return resolve_ticker(r.new_ticker, as_of, registry)
    return ticker


def resolve_ticker_reverse(
    ticker: str,
    as_of: str,
    registry: CorporateActionRegistry,
) -> List[str]:
    """Return all predecessor tickers that map to this ticker as of a date.

    Useful for 13F holdings mapping: if ONC was BGNE before 2025-01-02,
    we need to also check 13F filings referencing BGNE CUSIPs.
    """
    predecessors = []
    for a in registry.actions:
        if a.action == "ticker_change" and a.new_ticker == ticker and a.effective_date <= as_of:
            predecessors.append(a.old_ticker)
            # Recursively check for deeper chains
            predecessors.extend(resolve_ticker_reverse(a.old_ticker, as_of, registry))
    return predecessors


def cumulative_split_factor(
    ticker: str,
    from_date: str,
    to_date: str,
    registry: CorporateActionRegistry,
) -> float:
    """Compute the cumulative split adjustment factor between two dates.

    Returns the multiplier to apply to prices on ``from_date`` to make them
    comparable to prices on ``to_date``.

    Example: if a 1:5 reverse split occurred between from_date and to_date,
    returns 5.0 (multiply old prices by 5 to match new prices).
    """
    factor = 1.0
    for a in get_splits_only(ticker, registry):
        if from_date < a.effective_date <= to_date and a.factor:
            factor *= a.factor
    return factor


def list_dead_tickers(
    as_of: str,
    registry: CorporateActionRegistry,
) -> List[str]:
    """Return all tickers that are dead (acquired/delisted/bankrupt) as of a date."""
    dead_types = {"acquisition", "delisted", "bankruptcy"}
    result = set()
    for a in registry.actions:
        if a.action in dead_types and a.effective_date <= as_of:
            result.add(a.ticker)
    return sorted(result)


def list_renames(
    registry: CorporateActionRegistry,
) -> List[CorporateAction]:
    """Return all ticker rename actions."""
    return [a for a in registry.actions if a.action == "ticker_change"]
