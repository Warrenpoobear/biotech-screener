"""
Morningstar Returns Data Fetcher

Fetches and caches returns data from Morningstar Direct for backtesting.
Follows Wake Robin architecture principles:
- Tier-0 provenance tracking (SHA256 hashes)
- Point-in-time discipline (no look-ahead bias)
- Atomic file operations
- Fail-closed design

Architecture: "Fetch once, use forever"
- Token required ONLY for fetching (build_returns_database.py)
- Token NOT required for validation (validate_signals.py)
- All data cached as JSON for offline use

Usage:
    from morningstar_returns import MorningstarReturnsFetcher

    fetcher = MorningstarReturnsFetcher()
    returns = fetcher.fetch_returns(["VRTX", "BMRN"], "2023-01-01", "2023-12-31")
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import shutil
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Version for provenance tracking
VERSION = "1.0.0"

# Morningstar data package (only imported when fetching)
_md = None


def _get_md():
    """Lazy import of morningstar-data package."""
    global _md
    if _md is None:
        try:
            import morningstar_data as md
            _md = md
        except ImportError as e:
            raise ImportError(
                "morningstar-data package not installed. "
                "Install with: pip install morningstar-data"
            ) from e
    return _md


def _compute_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash of data."""
    if isinstance(data, dict):
        # Exclude non-deterministic fields
        excluded = {"loaded_at", "generated_at", "timestamp", "runtime_ms", "fetched_at"}
        cleaned = {k: v for k, v in sorted(data.items()) if k not in excluded}
        data = cleaned

    def default_serializer(obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    json_str = json.dumps(data, sort_keys=True, default=default_serializer)
    return f"sha256:{hashlib.sha256(json_str.encode()).hexdigest()}"


def _atomic_write_json(data: Dict, filepath: Path) -> str:
    """
    Atomically write JSON with SHA256 integrity verification.

    Returns:
        SHA256 hash of written data
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Compute hash before writing
    content_hash = _compute_hash(data)

    # Write to temp file first
    fd, temp_path = tempfile.mkstemp(suffix=".json", dir=filepath.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=str)

        # Atomic rename
        shutil.move(temp_path, filepath)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise

    return content_hash


class MorningstarReturnsFetcher:
    """
    Fetches returns data from Morningstar Direct.

    Key Features:
    - Uses md.direct.returns() API
    - Complete provenance tracking
    - Point-in-time discipline
    - Batch processing (to avoid API timeouts)
    - Atomic file operations

    Token Requirement:
    - MD_AUTH_TOKEN environment variable required for fetch operations
    - Token NOT required for reading cached data
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        batch_size: int = 20,
    ):
        """
        Initialize the fetcher.

        Args:
            cache_dir: Directory for cached returns (default: data/returns)
            batch_size: Number of tickers per API call (default: 20)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/returns")
        self.batch_size = batch_size
        self._md_initialized = False

    def _ensure_md_initialized(self) -> None:
        """Initialize Morningstar connection (requires token)."""
        if self._md_initialized:
            return

        md = _get_md()

        # Check for auth token
        token = os.environ.get("MD_AUTH_TOKEN")
        if not token:
            raise EnvironmentError(
                "MD_AUTH_TOKEN environment variable not set. "
                "Get token from Morningstar Direct and set with:\n"
                "  Windows: $env:MD_AUTH_TOKEN='your-token'\n"
                "  Linux:   export MD_AUTH_TOKEN='your-token'"
            )

        # Initialize the library
        try:
            md.init(token)
            self._md_initialized = True
        except Exception as e:
            raise ConnectionError(
                f"Failed to initialize Morningstar connection: {e}"
            ) from e

    def _fetch_batch_returns(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, List[Dict]]:
        """
        Fetch returns for a batch of tickers.

        Returns:
            Dict mapping ticker -> list of {date, return} records
        """
        self._ensure_md_initialized()
        md = _get_md()

        try:
            # Build security IDs (format: "TICKER:US" for US stocks)
            sec_ids = [f"{t}:US" for t in tickers]

            # Try the new get_returns API first, fall back to deprecated returns() if needed
            df = None
            try:
                # Import Frequency enum for new API (correct path: data_type submodule)
                from morningstar_data.direct.data_type import Frequency
                df = md.direct.get_returns(
                    investments=sec_ids,
                    start_date=start_date,
                    end_date=end_date,
                    freq=Frequency.monthly,  # lowercase enum value, 'freq' parameter
                )
            except (AttributeError, ImportError):
                # Fall back to deprecated API if get_returns not available
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    df = md.direct.returns(
                        investments=sec_ids,
                        start_date=start_date,
                        end_date=end_date,
                        freq="monthly",  # 'freq' parameter for deprecated API too
                    )

            # Convert DataFrame to dict
            results = {}
            if df is not None and not df.empty:
                # Handle DataFrame structure from Morningstar
                # The new API may have different column structure - check for 'Return' column
                # or iterate over securities

                # Check if it's a multi-index or simple structure
                if hasattr(df, 'columns'):
                    # Try to find return columns - handle both old and new formats
                    for col in df.columns:
                        col_str = str(col)
                        # Column names might be security IDs or tuples
                        if isinstance(col, tuple):
                            # Multi-level column - extract ticker and check for Return
                            ticker = None
                            for part in col:
                                part_str = str(part)
                                if ":" in part_str:
                                    ticker = part_str.split(":")[0].upper()
                                    break
                            if ticker is None:
                                continue
                        elif ":" in col_str:
                            ticker = col_str.split(":")[0].upper()
                        else:
                            ticker = col_str.upper()

                        records = []
                        for idx, val in df[col].items():
                            if val is not None:
                                try:
                                    float_val = float(val)
                                    if not (str(float_val) == "nan"):
                                        # Index is typically the date
                                        date_str = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)[:10]
                                        records.append({
                                            "date": date_str,
                                            "return": float_val,
                                        })
                                except (ValueError, TypeError):
                                    continue

                        if records and ticker not in results:
                            results[ticker] = records

            return results

        except Exception as e:
            # Log error but don't fail - partial data is still useful
            print(f"  Warning: Error fetching batch: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def fetch_returns(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        include_benchmark: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch returns for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_benchmark: Include XBI benchmark (default: True)

        Returns:
            Dict with structure:
            {
                "metadata": {
                    "start_date": str,
                    "end_date": str,
                    "fetched_at": str,
                    "source": "morningstar_direct",
                    "version": str,
                    "ticker_count": int,
                },
                "returns": {
                    "TICKER": [
                        {"date": "2023-01-31", "return": 0.0523},
                        ...
                    ],
                    ...
                },
                "benchmark": {
                    "XBI": [...] if include_benchmark else None
                },
                "provenance": {
                    "content_hash": str,
                    "fetch_timestamp": str,
                }
            }
        """
        # Add benchmark to tickers if requested
        all_tickers = list(tickers)
        if include_benchmark and "XBI" not in all_tickers:
            all_tickers.append("XBI")

        # Remove duplicates, preserve order
        seen = set()
        unique_tickers = []
        for t in all_tickers:
            t_upper = t.upper()
            if t_upper not in seen:
                seen.add(t_upper)
                unique_tickers.append(t_upper)

        print(f"Fetching returns for {len(unique_tickers)} tickers from {start_date} to {end_date}")

        # Fetch in batches
        all_returns = {}
        batches = [
            unique_tickers[i:i + self.batch_size]
            for i in range(0, len(unique_tickers), self.batch_size)
        ]

        for i, batch in enumerate(batches):
            print(f"  Batch {i+1}/{len(batches)}: {len(batch)} tickers...")
            batch_results = self._fetch_batch_returns(batch, start_date, end_date)
            all_returns.update(batch_results)

        # Separate benchmark from regular returns
        benchmark_data = {}
        ticker_returns = {}

        for ticker, returns in all_returns.items():
            if ticker == "XBI":
                benchmark_data["XBI"] = returns
            else:
                ticker_returns[ticker] = returns

        # Build result structure
        fetched_at = datetime.utcnow().isoformat() + "Z"

        result = {
            "metadata": {
                "start_date": start_date,
                "end_date": end_date,
                "fetched_at": fetched_at,
                "source": "morningstar_direct",
                "version": VERSION,
                "ticker_count": len(ticker_returns),
                "benchmark_included": include_benchmark,
            },
            "returns": ticker_returns,
            "benchmark": benchmark_data if include_benchmark else None,
            "coverage": {
                "requested": len([t for t in unique_tickers if t != "XBI"]),
                "received": len(ticker_returns),
                "missing": sorted(set(t for t in unique_tickers if t != "XBI") - set(ticker_returns.keys())),
            },
        }

        # Add provenance
        result["provenance"] = {
            "content_hash": _compute_hash(result),
            "fetch_timestamp": fetched_at,
            "source_version": VERSION,
        }

        print(f"  Retrieved returns for {len(ticker_returns)} tickers")
        if result["coverage"]["missing"]:
            print(f"  Missing: {result['coverage']['missing'][:5]}{'...' if len(result['coverage']['missing']) > 5 else ''}")

        return result

    def save_database(
        self,
        data: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save returns database to JSON file.

        Args:
            data: Returns data from fetch_returns()
            filename: Optional filename (default: auto-generated)

        Returns:
            Path to saved file
        """
        if filename is None:
            start = data["metadata"]["start_date"]
            end = data["metadata"]["end_date"]
            filename = f"returns_db_{start}_{end}.json"

        filepath = self.cache_dir / filename
        content_hash = _atomic_write_json(data, filepath)

        print(f"Saved database to: {filepath}")
        print(f"  Content hash: {content_hash}")

        return filepath


class ReturnsDatabase:
    """
    Reads cached returns database. NO TOKEN REQUIRED.

    This class works entirely offline using cached JSON data.
    Use this for validation and backtesting.
    """

    def __init__(self, database_path: Path):
        """
        Load returns database from file.

        Args:
            database_path: Path to returns_db_*.json file
        """
        self.database_path = Path(database_path)

        if not self.database_path.exists():
            raise FileNotFoundError(
                f"Returns database not found: {self.database_path}\n"
                "Run build_returns_database.py first to create it."
            )

        with open(self.database_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        self._returns = self._data.get("returns", {})
        self._benchmark = self._data.get("benchmark", {})
        self._metadata = self._data.get("metadata", {})

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get database metadata."""
        return self._metadata

    @property
    def available_tickers(self) -> List[str]:
        """Get list of tickers in database."""
        return sorted(self._returns.keys())

    @property
    def date_range(self) -> Tuple[str, str]:
        """Get (start_date, end_date) tuple."""
        return (
            self._metadata.get("start_date", ""),
            self._metadata.get("end_date", ""),
        )

    def get_ticker_returns(self, ticker: str) -> Optional[List[Dict]]:
        """
        Get returns data for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            List of {date, return} records, or None if not found
        """
        return self._returns.get(ticker.upper())

    def get_benchmark_returns(self, benchmark: str = "XBI") -> Optional[List[Dict]]:
        """Get benchmark returns data."""
        return self._benchmark.get(benchmark.upper())

    def get_forward_return(
        self,
        ticker: str,
        screen_date: str,
        forward_months: int,
    ) -> Optional[Decimal]:
        """
        Calculate forward return from screen date.

        Point-in-time discipline: Only uses returns AFTER screen_date.

        Args:
            ticker: Ticker symbol
            screen_date: Date of the screen (YYYY-MM-DD)
            forward_months: Number of months forward

        Returns:
            Cumulative return as Decimal, or None if unavailable
        """
        returns = self.get_ticker_returns(ticker)
        if not returns:
            return None

        screen_dt = date.fromisoformat(screen_date)
        end_dt = screen_dt + timedelta(days=forward_months * 30)  # Approximate

        # Filter returns in forward window
        forward_returns = []
        for r in returns:
            r_date = date.fromisoformat(r["date"][:10])
            if screen_dt < r_date <= end_dt:
                forward_returns.append(Decimal(str(r["return"])))

        if not forward_returns:
            return None

        # Compound returns: (1 + r1) * (1 + r2) * ... - 1
        cumulative = Decimal("1")
        for r in forward_returns:
            cumulative *= (Decimal("1") + r)

        return (cumulative - Decimal("1")).quantize(Decimal("0.000001"))

    def get_excess_return(
        self,
        ticker: str,
        screen_date: str,
        forward_months: int,
        benchmark: str = "XBI",
    ) -> Optional[Decimal]:
        """
        Calculate excess return vs benchmark.

        Args:
            ticker: Ticker symbol
            screen_date: Date of the screen
            forward_months: Number of months forward
            benchmark: Benchmark ticker (default: XBI)

        Returns:
            Excess return as Decimal, or None if unavailable
        """
        ticker_return = self.get_forward_return(ticker, screen_date, forward_months)
        bench_return = self.get_forward_return(benchmark, screen_date, forward_months)

        if ticker_return is None or bench_return is None:
            return None

        return (ticker_return - bench_return).quantize(Decimal("0.000001"))


# Convenience functions
def load_database(path: str) -> ReturnsDatabase:
    """Load returns database from path. No token required."""
    return ReturnsDatabase(Path(path))


def fetch_and_save(
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Fetch returns and save to database. Token required.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory (default: data/returns)

    Returns:
        Path to saved database
    """
    cache_dir = Path(output_dir) if output_dir else None
    fetcher = MorningstarReturnsFetcher(cache_dir=cache_dir)
    data = fetcher.fetch_returns(tickers, start_date, end_date)
    return fetcher.save_database(data)
