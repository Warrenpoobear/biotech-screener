"""Cell Atlas REST API.

Endpoints
----------
``GET /api/atlas/cells``          all companies with cell metadata
``GET /api/atlas/cell/{ticker}``  detailed single cell
``GET /api/atlas/tissue/{cat}`    cells grouped by therapeutic category ("tissue")
``GET /api/atlas/stats``          summary statistics
``GET /api/atlas/snapshots``      available snapshot dates (for the timeline slider)
``GET /api/atlas/meta``           tissue/flag vocabulary for UI filters
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from .. import data_loader as dl

router = APIRouter(prefix="/api/atlas", tags=["atlas"])


# ---------------------------------------------------------------------------
# /cells
# ---------------------------------------------------------------------------


@router.get("/cells")
def get_cells(
    snapshot: Optional[str] = Query(
        default=None,
        description="Snapshot date YYYY-MM-DD; defaults to latest",
    ),
    tissue: Optional[str] = Query(default=None, description="Filter by therapeutic category"),
    min_composite: Optional[float] = Query(default=None, description="Minimum composite score"),
    has_catalyst: Optional[bool] = Query(default=None, description="Only cells with >=1 upcoming catalyst"),
):
    """Return every cell (company) as an array of cell-atlas records."""
    cells = dl.build_all_cells(snapshot)
    out: List[Dict[str, Any]] = []
    for c in cells:
        if tissue and c["tissue"].lower() != tissue.lower():
            continue
        if min_composite is not None:
            cs = c.get("composite_score")
            if cs is None or cs < min_composite:
                continue
        if has_catalyst is True and c["receptors"]["catalyst_count"] < 1:
            continue
        if has_catalyst is False and c["receptors"]["catalyst_count"] >= 1:
            continue
        out.append(c)
    dates = dl.load_snapshot_dates()
    resolved_snapshot = snapshot or (dates[-1] if dates else None)
    return {
        "count": len(out),
        "snapshot": resolved_snapshot,
        "cells": out,
    }


# ---------------------------------------------------------------------------
# /cell/{ticker}
# ---------------------------------------------------------------------------


@router.get("/cell/{ticker}")
def get_cell(ticker: str, snapshot: Optional[str] = Query(default=None)):
    """Detailed cell record for a single ticker."""
    cell = dl.get_cell(ticker, snapshot)
    if cell is None:
        raise HTTPException(status_code=404, detail=f"Cell not found: {ticker}")
    # Enrich with full trial rows for the detail panel.
    trials = dl.load_trial_mapping().get(ticker.upper(), [])
    aact = dl.load_aact_studies()
    sponsors = dl.load_aact_sponsors()
    cell["trials"] = [
        {
            **t,
            "phase": aact.get(t.get("nct_id", ""), {}).get("phase", "Unknown"),
            "overall_status": aact.get(t.get("nct_id", ""), {}).get("overall_status", "Unknown"),
            "primary_completion_date": aact.get(t.get("nct_id", ""), {}).get("primary_completion_date", ""),
            "all_sponsors": sponsors.get(t.get("nct_id", ""), []),
        }
        for t in trials
    ]
    return cell


# ---------------------------------------------------------------------------
# /tissues  (list all categories) — MUST precede /tissue/{category} so the
# static path is matched before the {category} path parameter would grab it.
# ---------------------------------------------------------------------------


@router.get("/tissues")
def list_tissues(snapshot: Optional[str] = Query(default=None)):
    """List every therapeutic category present, with member counts."""
    cells = dl.build_all_cells(snapshot)
    counts: Dict[str, int] = {}
    for c in cells:
        counts[c["tissue"]] = counts.get(c["tissue"], 0) + 1
    items = [{"tissue": t, "cell_count": n} for t, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {"tissues": items, "total": len(items)}


# ---------------------------------------------------------------------------
# /tissue/{category}
# ---------------------------------------------------------------------------


@router.get("/tissue/{category}")
def get_tissue(category: str, snapshot: Optional[str] = Query(default=None)):
    """All cells belonging to a therapeutic category ("tissue")."""
    cells = dl.build_all_cells(snapshot)
    members = [c for c in cells if c["tissue"].lower() == category.lower()]
    scores = [c["composite_score"] for c in members if c.get("composite_score") is not None]
    return {
        "tissue": category,
        "cell_count": len(members),
        "tickers": [c["ticker"] for c in members],
        "avg_composite_score": statistics.fmean(scores) if scores else None,
        "total_catalysts": sum(c["receptors"]["catalyst_count"] for c in members),
        "total_trials": sum(c["trial_count"] for c in members),
        "cells": members,
    }


# ---------------------------------------------------------------------------
# /stats
# ---------------------------------------------------------------------------


@router.get("/stats")
def get_stats(snapshot: Optional[str] = Query(default=None)):
    """Summary statistics for the current atlas."""
    cells = dl.build_all_cells(snapshot)
    scores = [c["composite_score"] for c in cells if c.get("composite_score") is not None]
    prices = [c["latest_price"] for c in cells if c.get("latest_price") is not None]
    catalyst_counts = [c["receptors"]["catalyst_count"] for c in cells]
    trial_counts = [c["trial_count"] for c in cells]

    # Tissue distribution
    tissue_dist: Dict[str, int] = {}
    for c in cells:
        tissue_dist[c["tissue"]] = tissue_dist.get(c["tissue"], 0) + 1

    # Severity distribution
    sev_dist: Dict[str, int] = {}
    for c in cells:
        sev = c.get("severity") or "unknown"
        sev_dist[sev] = sev_dist.get(sev, 0) + 1

    # Score histogram (10 buckets, 0-10, 10-20, ...)
    hist = [0] * 10
    for s in scores:
        idx = min(int(s // 10), 9)
        hist[idx] += 1

    return {
        "snapshot": snapshot or (dl.load_snapshot_dates()[-1] if dl.load_snapshot_dates() else None),
        "total_cells": len(cells),
        "cells_with_score": len(scores),
        "avg_composite_score": statistics.fmean(scores) if scores else None,
        "median_composite_score": statistics.median(scores) if scores else None,
        "min_composite_score": min(scores) if scores else None,
        "max_composite_score": max(scores) if scores else None,
        "avg_price": statistics.fmean(prices) if prices else None,
        "total_catalysts": sum(catalyst_counts),
        "avg_catalyst_count": statistics.fmean(catalyst_counts) if catalyst_counts else 0.0,
        "total_trials": sum(trial_counts),
        "tissue_distribution": tissue_dist,
        "severity_distribution": sev_dist,
        "score_histogram": hist,
        "histogram_bins": [f"{i*10}-{i*10+10}" for i in range(10)],
        "available_snapshots": dl.load_snapshot_dates(),
    }


# ---------------------------------------------------------------------------
# /snapshots  (timeline list)
# ---------------------------------------------------------------------------


@router.get("/snapshots")
def list_snapshots():
    """Available snapshot dates for the timeline slider."""
    dates = dl.load_snapshot_dates()
    return {"snapshots": dates, "count": len(dates)}


# ---------------------------------------------------------------------------
# /meta (UI vocabulary)
# ---------------------------------------------------------------------------


@router.get("/meta")
def get_meta():
    """Static-ish vocabularies used by the dashboard filters."""
    cells = dl.build_all_cells()
    tissues = sorted({c["tissue"] for c in cells})
    flags = sorted({f for c in cells for f in c["flags"]})
    severities = sorted({c["severity"] for c in cells})
    cap_buckets = sorted({c["market_cap_bucket"] for c in cells})
    stages = sorted({c["stage_bucket"] for c in cells})
    return {
        "tissues": tissues,
        "flags": flags,
        "severities": severities,
        "cap_buckets": cap_buckets,
        "stages": stages,
    }
