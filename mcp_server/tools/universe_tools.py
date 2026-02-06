"""Universe and data availability tools."""

import json
import logging
from pathlib import Path

from mcp_server.app import mcp
from mcp_server.config import UNIVERSE_FILE, UNIVERSE_MAPPED_FILE, DATA_DIR

logger = logging.getLogger(__name__)


def _find_latest_screen_file() -> Path | None:
    """Find the most recent screen_YYYY-MM-DD.json file."""
    candidates = sorted(DATA_DIR.glob("screen_2*.json"), reverse=True)
    return candidates[0] if candidates else None


@mcp.tool()
def get_universe() -> str:
    """Return the current biotech screening universe.

    Lists all tickers with basic metadata (name, exchange, sector,
    market cap) from the universe definition file.
    """
    try:
        if UNIVERSE_FILE.exists():
            with open(UNIVERSE_FILE) as f:
                universe = json.load(f)
        elif UNIVERSE_MAPPED_FILE.exists():
            with open(UNIVERSE_MAPPED_FILE) as f:
                universe = json.load(f)
        else:
            return json.dumps({"error": "No universe file found"})

        summary = []
        for entry in universe:
            ticker = entry.get("ticker", "?")
            rec = {
                "ticker": ticker,
                "name": entry.get("name", entry.get("market_data", {}).get("company_name", "")),
                "exchange": entry.get("exchange", ""),
                "sector": entry.get("sector", entry.get("market_data", {}).get("sector", "")),
                "market_cap": entry.get("market_cap", entry.get("market_data", {}).get("market_cap")),
            }
            summary.append(rec)

        return json.dumps({
            "total_tickers": len(summary),
            "tickers": summary,
        })
    except Exception as e:
        logger.exception("get_universe failed")
        return json.dumps({"error": str(e)})


@mcp.tool()
def check_data_availability() -> str:
    """Check availability of Morningstar, yfinance, and local data files.

    Reports which data sources are importable and which production
    data files exist on disk.
    """
    result: dict = {}

    # Check Morningstar SDK
    try:
        from wake_robin_data_pipeline.morningstar_data_provider import (
            check_morningstar_availability,
        )
        result["providers"] = check_morningstar_availability()
    except ImportError:
        result["providers"] = {
            "morningstar_available": False,
            "yfinance_available": False,
            "note": "wake_robin_data_pipeline not importable",
        }

    # Check local data files
    files_status = {}
    for label, path in [
        ("universe.json", UNIVERSE_FILE),
        ("universe_mapped.json", UNIVERSE_MAPPED_FILE),
        ("financial_records.json", DATA_DIR / "financial_records.json"),
        ("screen_output.json", DATA_DIR / "screen_output.json"),
    ]:
        files_status[label] = path.exists()

    latest_screen = _find_latest_screen_file()
    if latest_screen:
        files_status["latest_screen"] = latest_screen.name

    result["files"] = files_status

    return json.dumps(result)
