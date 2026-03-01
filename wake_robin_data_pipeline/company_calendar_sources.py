"""
company_calendar_sources.py - Curated URL mapping for IR events + press release collectors.

Loads production_data/company_calendar_sources.json and merges ir_url / pr_rss_url
into universe rows so collectors can fetch real pages without modifying universe.json.

Design:
  - Universe values win (never override existing ir_url / pr_rss_url).
  - Merge is deterministic: output order matches input universe order.
  - Stats track how many URLs were added.
"""
from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

_SCHEMA = "company_calendar_sources.v1"
_UNMAPPED_CAP = 50


def load_company_calendar_sources(path: Path) -> Dict[str, Dict[str, str]]:
    """Load curated source mapping from JSON.

    Returns:
        Dict mapping ticker (uppercase) -> {"ir_url": ..., "pr_rss_url": ...}.
        Empty dict if file missing or malformed.
    """
    if not path.exists():
        logger.debug("Company calendar sources file not found: %s", path)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load company calendar sources: %s", e)
        raise

    schema = data.get("_schema", "")
    if schema != _SCHEMA:
        logger.warning(
            "Company calendar sources schema mismatch: expected %s, got %s",
            _SCHEMA, schema,
        )

    raw_sources = data.get("sources", {})
    if not isinstance(raw_sources, dict):
        logger.error("Company calendar sources 'sources' field is not a dict")
        return {}

    # Normalize tickers to uppercase
    normalized: Dict[str, Dict[str, str]] = {}
    for ticker, urls in raw_sources.items():
        tk = ticker.strip().upper()
        if not tk or not isinstance(urls, dict):
            continue
        normalized[tk] = {k: v for k, v in urls.items() if isinstance(v, str)}

    return normalized


def validate_company_calendar_sources(
    sources: Dict[str, Dict[str, str]],
) -> List[str]:
    """Validate source URLs. Returns list of warning strings (non-fatal)."""
    warnings: List[str] = []

    for ticker, urls in sorted(sources.items()):
        for field in ("ir_url", "pr_rss_url"):
            url = urls.get(field)
            if url is None:
                continue
            if not url.startswith(("http://", "https://")):
                warnings.append(
                    f"{ticker}.{field}: URL does not start with http(s): {url[:80]}"
                )
            if field == "pr_rss_url":
                url_lower = url.lower()
                if not any(hint in url_lower for hint in ("rss", "feed", ".xml")):
                    warnings.append(
                        f"{ticker}.pr_rss_url: URL lacks rss/feed/.xml hint: {url[:80]}"
                    )

    return warnings


def merge_universe_with_company_calendar_sources(
    universe: List[Dict[str, Any]],
    sources: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge curated source URLs into universe rows.

    For each universe row with ticker T:
      - If sources has T and row lacks ir_url, set from sources.
      - If sources has T and row lacks pr_rss_url, set from sources.
      - Existing universe values are NEVER overridden.

    Args:
        universe: List of universe dicts (not mutated).
        sources: Dict from load_company_calendar_sources().

    Returns:
        (merged_universe, stats) where merged_universe is a new list and
        stats tracks what changed.
    """
    ir_added = 0
    pr_added = 0
    ir_before = 0
    pr_before = 0
    tickers_seen: set = set()

    merged: List[Dict[str, Any]] = []

    for row in universe:
        ticker = (row.get("ticker") or "").strip().upper()
        new_row = deepcopy(row)

        if ticker:
            tickers_seen.add(ticker)

        # Count pre-existing URLs
        if row.get("ir_url"):
            ir_before += 1
        if row.get("pr_rss_url"):
            pr_before += 1

        # Merge from sources
        src = sources.get(ticker) if ticker else None
        if src:
            if not new_row.get("ir_url") and src.get("ir_url"):
                new_row["ir_url"] = src["ir_url"]
                ir_added += 1
            if not new_row.get("pr_rss_url") and src.get("pr_rss_url"):
                new_row["pr_rss_url"] = src["pr_rss_url"]
                pr_added += 1

        merged.append(new_row)

    # Unmapped sources (in mapping but not in universe)
    unmapped = sorted(set(sources.keys()) - tickers_seen)

    stats = {
        "tickers_total": len(tickers_seen),
        "ir_url_before": ir_before,
        "pr_rss_url_before": pr_before,
        "ir_url_added": ir_added,
        "pr_rss_url_added": pr_added,
        "ir_url_after": ir_before + ir_added,
        "pr_rss_url_after": pr_before + pr_added,
        "unmapped_sources_count": len(unmapped),
        "unmapped_sources": unmapped[:_UNMAPPED_CAP],
    }

    return merged, stats
