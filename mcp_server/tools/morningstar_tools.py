"""Morningstar MCP quantitative datapoint tools."""

import json
import logging

from mcp_server.app import mcp
from mcp_server.config import MORNINGSTAR_MCP_DATA_FILE

logger = logging.getLogger(__name__)


def _load_morningstar_data() -> dict:
    """Load morningstar_mcp_data.json."""
    if MORNINGSTAR_MCP_DATA_FILE.exists():
        with open(MORNINGSTAR_MCP_DATA_FILE) as f:
            return json.load(f)
    return {}


@mcp.tool()
def get_morningstar_fundamentals(ticker: str) -> str:
    """Return Morningstar quantitative datapoints for a ticker.

    Args:
        ticker: Equity ticker symbol (e.g. "VRTX").

    Returns all available Morningstar datapoints including Quantitative Fair
    Value (QV009), ROIC (STA4Z), ROE (HS08F), Debt/Equity (ST389),
    Debt/Capital (HS06U), Sales Growth (HS035), Net Margin (HS08D),
    P/E (HS05X), P/S (HS05U), P/B (ST408), and more.
    """
    try:
        data = _load_morningstar_data()
        if not data:
            return json.dumps({"ticker": ticker, "error": "Morningstar MCP data file not found"})

        metadata = data.get("metadata", {})
        records = data.get("records", {})
        catalog = metadata.get("datapoint_catalog", {})

        match = records.get(ticker.upper())
        if not match:
            return json.dumps({"ticker": ticker, "error": "Ticker not found in Morningstar data"})

        # Build human-readable output with both IDs and names
        result = {
            "ticker": ticker.upper(),
            "source": metadata.get("source", "Morningstar MCP"),
            "pull_date": metadata.get("pull_date"),
            "datapoints": {},
            "raw": match,
        }

        for dp_id, value in match.items():
            name = catalog.get(dp_id, dp_id)
            result["datapoints"][name] = value

        return json.dumps(result, default=str)
    except Exception as e:
        logger.exception("get_morningstar_fundamentals failed for %s", ticker)
        return json.dumps({"ticker": ticker, "error": str(e)})
