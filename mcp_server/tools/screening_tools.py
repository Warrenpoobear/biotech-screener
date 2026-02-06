"""Screening result tools — read from screen output JSON files."""

import json
import logging
from pathlib import Path

from mcp_server.app import mcp
from mcp_server.config import DATA_DIR, SCREEN_OUTPUT_FILE

logger = logging.getLogger(__name__)


def _load_screen_data() -> dict | None:
    """Load the most recent screen output, preferring dated files."""
    # Try dated files first (most recent)
    candidates = sorted(DATA_DIR.glob("screen_2*.json"), reverse=True)
    for path in candidates:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            continue

    # Fall back to screen_output.json
    if SCREEN_OUTPUT_FILE.exists():
        try:
            with open(SCREEN_OUTPUT_FILE) as f:
                return json.load(f)
        except Exception:
            pass

    return None


def _extract_ranked(data: dict) -> list[dict]:
    """Extract ranked_securities from the screen output structure."""
    # module_5_composite.ranked_securities is the canonical location
    m5 = data.get("module_5_composite", {})
    ranked = m5.get("ranked_securities", [])
    if ranked:
        return ranked

    # Fallback: check summary.final_ranked if it's a list
    summary = data.get("summary", {})
    fr = summary.get("final_ranked")
    if isinstance(fr, list):
        return fr

    return []


def _slim_record(rec: dict) -> dict:
    """Return a compact representation of a ranked security."""
    return {
        "ticker": rec.get("ticker"),
        "composite_rank": rec.get("composite_rank"),
        "composite_score": rec.get("composite_score"),
        "rank_score": rec.get("rank_score"),
        "confidence_overall": rec.get("confidence_overall"),
        "volatility": rec.get("volatility"),
        "market_cap_bucket": rec.get("market_cap_bucket"),
        "stage_bucket": rec.get("stage_bucket"),
        "severity": rec.get("severity"),
        "rank_driver": rec.get("rank_driver"),
        "component_scores": rec.get("component_scores"),
    }


@mcp.tool()
def get_latest_screen_results(top_n: int = 20) -> str:
    """Return the top-N ranked securities from the latest screen run.

    Args:
        top_n: Number of top-ranked securities to return. Defaults to 20.

    Returns JSON with run metadata and a ranked list.
    """
    try:
        data = _load_screen_data()
        if data is None:
            return json.dumps({"error": "No screen output files found"})

        ranked = _extract_ranked(data)
        if not ranked:
            return json.dumps({"error": "No ranked_securities in screen output"})

        # Sort by composite_rank
        ranked.sort(key=lambda r: r.get("composite_rank", 9999))
        top = [_slim_record(r) for r in ranked[:top_n]]

        meta = data.get("run_metadata", {})
        summary = data.get("summary", {})

        return json.dumps({
            "as_of_date": meta.get("as_of_date", "unknown"),
            "total_ranked": len(ranked),
            "total_evaluated": summary.get("total_evaluated"),
            "regime": summary.get("regime"),
            "top_n": top_n,
            "results": top,
        })
    except Exception as e:
        logger.exception("get_latest_screen_results failed")
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_ticker_detail(ticker: str) -> str:
    """Return the full score breakdown for a single ticker from the latest screen.

    Args:
        ticker: Equity ticker symbol (e.g. "VRTX").

    Returns JSON with all scoring fields for that ticker.
    """
    try:
        data = _load_screen_data()
        if data is None:
            return json.dumps({"error": "No screen output files found"})

        ranked = _extract_ranked(data)
        match = [r for r in ranked if r.get("ticker", "").upper() == ticker.upper()]

        if not match:
            return json.dumps({
                "ticker": ticker,
                "error": f"Ticker {ticker} not found in screen results",
            })

        rec = match[0]
        # Serialise Decimal-like objects
        return json.dumps(rec, default=str)
    except Exception as e:
        logger.exception("get_ticker_detail failed for %s", ticker)
        return json.dumps({"ticker": ticker, "error": str(e)})


@mcp.tool()
def compare_tickers(tickers: list[str]) -> str:
    """Side-by-side comparison of multiple tickers from the latest screen.

    Args:
        tickers: List of ticker symbols to compare.

    Returns JSON with each ticker's rank, score, components, and
    confidence metrics for easy comparison.
    """
    try:
        data = _load_screen_data()
        if data is None:
            return json.dumps({"error": "No screen output files found"})

        ranked = _extract_ranked(data)
        lookup = {r.get("ticker", "").upper(): r for r in ranked}

        comparison = []
        missing = []
        for tk in tickers:
            tk_upper = tk.upper()
            if tk_upper in lookup:
                comparison.append(_slim_record(lookup[tk_upper]))
            else:
                missing.append(tk)

        result: dict = {"compared": comparison}
        if missing:
            result["not_found"] = missing

        return json.dumps(result, default=str)
    except Exception as e:
        logger.exception("compare_tickers failed")
        return json.dumps({"error": str(e)})
