"""MCP server entry point.

Run with:  python -m mcp_server.server
"""

import logging
import sys

# Configure logging to stderr (stdout is reserved for MCP JSON-RPC)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

import mcp_server.tools.cartography_tools  # noqa: F401, E402
import mcp_server.tools.fundamentals_tools  # noqa: F401, E402
import mcp_server.tools.morningstar_tools  # noqa: F401, E402
import mcp_server.tools.price_tools  # noqa: F401, E402
import mcp_server.tools.screening_tools  # noqa: F401, E402

# Import tool modules so their @mcp.tool() decorators register
import mcp_server.tools.universe_tools  # noqa: F401, E402

# Import the FastMCP instance
from mcp_server.app import mcp  # noqa: E402


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
