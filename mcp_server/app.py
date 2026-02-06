"""FastMCP application instance.

All tool modules import `mcp` from here to avoid circular imports.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "biotech-screener",
    instructions=(
        "Biotech screening tools backed by Morningstar Direct and yfinance. "
        "Provides daily returns, prices, volumes, screening results, "
        "financial health metrics, and universe management for biotech equities."
    ),
)
