"""Path constants and configuration for the MCP server."""

from pathlib import Path

# Project root is one level up from mcp_server/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "production_data"
CACHE_DIR = PROJECT_ROOT / "cache"

# Key data files
UNIVERSE_FILE = DATA_DIR / "universe.json"
UNIVERSE_MAPPED_FILE = DATA_DIR / "universe_mapped.json"
FINANCIAL_RECORDS_FILE = DATA_DIR / "financial_records.json"
SCREEN_OUTPUT_FILE = DATA_DIR / "screen_output.json"
MORNINGSTAR_MCP_DATA_FILE = DATA_DIR / "morningstar_mcp_data.json"
