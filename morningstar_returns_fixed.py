"""
Morningstar Direct Returns Data Integration
Wake Robin Biotech Screener - Phase 1

Fetches historical returns data for backtesting with Tier-0 provenance.
Uses md.direct.get_returns() with Frequency enum (updated API).

Key principles:
- Point-in-time discipline (never refetch historical data)
- Tier-0 provenance (every fact traces to source)
- Atomic file operations (SHA256 integrity)
- Fail-closed behavior (degrade visibly, not silently)
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import TypedDict
from dataclasses import dataclass, asdict


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class AbsoluteReturnRecord(TypedDict):
    """Absolute return record from fetch."""
    sec_id: str
    date: str
    return_pct: str


class ExcessReturnRecord(TypedDict):
    """Excess return record from fetch."""
    sec_id: str
    date: str
    excess_return_pct: str


class FetchReturnsProvenance(TypedDict):
    """Provenance metadata from fetch_returns."""
    source: str
    fetch_timestamp: str
    data_start_date: str
    data_end_date: str
    frequency: str
    benchmark_sec_id: Optional[str]
    api_method: str
    package_version: str
    num_observations: int
    sha256: str


class FetchReturnsResult(TypedDict):
    """Result structure from fetch_returns."""
    absolute: List[AbsoluteReturnRecord]
    excess: List[ExcessReturnRecord]
    provenance: FetchReturnsProvenance

# Morningstar package (optional dependency)
try:
    import morningstar_data as md
    from morningstar_data.direct.data_type import Frequency
    MORNINGSTAR_AVAILABLE = True
except ImportError:
    MORNINGSTAR_AVAILABLE = False
    md = None
    Frequency = None


@dataclass
class ReturnsProvenance:
    """Tier-0 provenance for returns data."""
    source: str  # 'morningstar_direct'
    fetch_timestamp: str  # ISO format
    data_start_date: str
    data_end_date: str
    frequency: str
    benchmark_sec_id: Optional[str]
    api_method: str  # 'md.direct.get_returns'
    package_version: str
    num_observations: int
    sha256: str  # Hash of raw data


@dataclass
class ReturnsRecord:
    """Single returns observation with provenance."""
    ticker: str
    sec_id: str
    date: str  # YYYY-MM-DD
    return_pct: Decimal
    excess_return_pct: Optional[Decimal]  # vs benchmark if available
    provenance: ReturnsProvenance


class MorningstarReturnsError(Exception):
    """Raised when returns data cannot be fetched."""
    pass


def check_availability() -> Tuple[bool, str]:
    """
    Check if Morningstar Direct is available.

    Returns:
        (available: bool, message: str)
    """
    if not MORNINGSTAR_AVAILABLE:
        return False, "morningstar-data package not installed"

    if not os.environ.get('MD_AUTH_TOKEN'):
        return False, "MD_AUTH_TOKEN environment variable not set"

    # The morningstar-data package reads MD_AUTH_TOKEN automatically
    # No explicit init() call needed
    return True, "Morningstar Direct available"


def fetch_returns(
    sec_ids: List[str],
    start_date: str,
    end_date: str,
    frequency: str = 'monthly',
    benchmark_sec_id: Optional[str] = None
) -> FetchReturnsResult:
    """
    Fetch historical returns from Morningstar Direct.

    Args:
        sec_ids: List of Morningstar security IDs
        start_date: YYYY-MM-DD format
        end_date: YYYY-MM-DD format
        frequency: 'monthly', 'quarterly', 'yearly'
        benchmark_sec_id: Optional benchmark for excess returns

    Returns:
        {
            'absolute': List of absolute return records,
            'excess': List of excess return records (if benchmark provided),
            'provenance': Provenance metadata
        }

    Raises:
        MorningstarReturnsError: If data cannot be fetched
    """
    available, msg = check_availability()
    if not available:
        raise MorningstarReturnsError(f"Morningstar not available: {msg}")

    fetch_timestamp = datetime.utcnow().isoformat() + 'Z'

    try:
        # Map frequency string to enum (lowercase values)
        freq_map = {
            'monthly': Frequency.monthly if Frequency else None,
            'quarterly': Frequency.quarterly if Frequency else None,
            'yearly': Frequency.yearly if Frequency else None,
            'daily': Frequency.daily if Frequency else None,
            'weekly': Frequency.weekly if Frequency else None,
        }
        freq_enum = freq_map.get(frequency.lower())

        # Fetch returns using get_returns (preferred) or returns (deprecated)
        absolute_returns = None
        api_method = 'md.direct.get_returns'

        try:
            if freq_enum is not None:
                absolute_returns = md.direct.get_returns(
                    investments=sec_ids,
                    start_date=start_date,
                    end_date=end_date,
                    freq=freq_enum  # Note: parameter is 'freq', not 'frequency'
                )
            else:
                raise AttributeError("Frequency enum not available")
        except (AttributeError, TypeError) as e:
            # Fall back to deprecated API
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                absolute_returns = md.direct.returns(
                    investments=sec_ids,
                    start_date=start_date,
                    end_date=end_date,
                    freq=frequency  # String frequency for deprecated API
                )
            api_method = 'md.direct.returns (deprecated)'

        if absolute_returns is None or absolute_returns.empty:
            raise MorningstarReturnsError("No returns data returned")

        # Convert DataFrame to records
        # DataFrame has columns: ['Id', 'Date', 'Monthly Return'] (or similar)
        absolute_records = []

        # Find the return column (could be 'Monthly Return', 'Return', etc.)
        return_col = None
        for col in absolute_returns.columns:
            col_str = str(col).lower()
            if 'return' in col_str and col_str not in ['id', 'date']:
                return_col = col
                break

        if return_col and 'Id' in absolute_returns.columns:
            # Standard format: rows with Id, Date, [Monthly Return] columns
            for idx, row in absolute_returns.iterrows():
                ret_val = row[return_col]
                # Skip NA/null values
                if ret_val is None or (hasattr(ret_val, '__class__') and str(ret_val) in ['<NA>', 'nan', 'NaN']):
                    continue
                try:
                    date_val = row.get('Date', idx)
                    date_str = str(date_val)[:10] if date_val is not None else str(idx)[:10]
                    absolute_records.append({
                        'sec_id': str(row['Id']),
                        'date': date_str,
                        'return_pct': str(Decimal(str(float(ret_val)))),
                    })
                except (ValueError, TypeError):
                    continue
        else:
            # Alternative format: security IDs as columns, dates as index
            for col in absolute_returns.columns:
                if str(col).lower() in ['id', 'date']:
                    continue
                sec_id = str(col)
                for idx, val in absolute_returns[col].items():
                    if val is not None and str(val) not in ['<NA>', 'nan', 'NaN']:
                        try:
                            float_val = float(val)
                            date_str = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)[:10]
                            absolute_records.append({
                                'sec_id': sec_id,
                                'date': date_str,
                                'return_pct': str(Decimal(str(float_val))),
                            })
                        except (ValueError, TypeError):
                            continue

        # Fetch excess returns if benchmark provided
        excess_records = []
        if benchmark_sec_id:
            try:
                excess_returns = md.direct.get_excess_returns(
                    investments=sec_ids,
                    benchmark_sec_id=benchmark_sec_id,
                    start_date=start_date,
                    end_date=end_date,
                    freq=freq_enum if freq_enum else frequency
                )

                if excess_returns is not None and not excess_returns.empty:
                    if 'Excess Return' in excess_returns.columns and 'Id' in excess_returns.columns:
                        for idx, row in excess_returns.iterrows():
                            date_val = row.get('Date', idx)
                            date_str = str(date_val)[:10] if date_val is not None else str(idx)[:10]
                            excess_records.append({
                                'sec_id': str(row['Id']),
                                'date': date_str,
                                'excess_return_pct': str(Decimal(str(row['Excess Return']))),
                            })
                    else:
                        for col in excess_returns.columns:
                            for idx, val in excess_returns[col].items():
                                if val is not None:
                                    try:
                                        float_val = float(val)
                                        if str(float_val) != 'nan':
                                            date_str = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)[:10]
                                            excess_records.append({
                                                'sec_id': str(col),
                                                'date': date_str,
                                                'excess_return_pct': str(Decimal(str(float_val))),
                                            })
                                    except (ValueError, TypeError):
                                        continue
            except Exception as e:
                print(f"Warning: Could not fetch excess returns: {e}")

        # Create provenance
        raw_data = json.dumps({
            'absolute': absolute_records,
            'excess': excess_records
        }, sort_keys=True)

        sha256 = hashlib.sha256(raw_data.encode()).hexdigest()

        provenance = ReturnsProvenance(
            source='morningstar_direct',
            fetch_timestamp=fetch_timestamp,
            data_start_date=start_date,
            data_end_date=end_date,
            frequency=frequency,
            benchmark_sec_id=benchmark_sec_id,
            api_method=api_method,
            package_version=getattr(md, '__version__', 'unknown'),
            num_observations=len(absolute_records),
            sha256=sha256
        )

        return {
            'absolute': absolute_records,
            'excess': excess_records,
            'provenance': asdict(provenance)
        }

    except MorningstarReturnsError:
        raise
    except Exception as e:
        raise MorningstarReturnsError(f"Failed to fetch returns: {e}") from e


def save_returns_cache(
    data: FetchReturnsResult,
    output_dir: str,
    filename: str,
    atomic: bool = True
) -> str:
    """
    Save returns data to disk with atomic write and SHA256 verification.

    Args:
        data: Returns data dict from fetch_returns()
        output_dir: Directory to save to
        filename: Filename (without path)
        atomic: Use atomic write with backup

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Serialize data
    json_data = json.dumps(data, indent=2, sort_keys=True)

    if atomic and os.path.exists(filepath):
        # Backup existing file
        backup_path = filepath + '.backup'
        os.replace(filepath, backup_path)

        try:
            # Write new file
            with open(filepath, 'w') as f:
                f.write(json_data)

            # Verify integrity
            with open(filepath, 'r') as f:
                verify_data = f.read()

            if verify_data != json_data:
                # Restore backup on corruption
                os.replace(backup_path, filepath)
                raise IOError("File corruption detected during write")

            # Remove backup on success
            if os.path.exists(backup_path):
                os.remove(backup_path)

        except Exception as e:
            # Restore backup on any error
            if os.path.exists(backup_path):
                os.replace(backup_path, filepath)
            raise
    else:
        # Simple write for new files
        with open(filepath, 'w') as f:
            f.write(json_data)

    return filepath


def load_returns_cache(filepath: str) -> FetchReturnsResult:
    """
    Load returns data from cache with integrity verification.

    Args:
        filepath: Path to cached returns file

    Returns:
        Returns data dict

    Raises:
        IOError: If file cannot be loaded or fails integrity check
    """
    if not os.path.exists(filepath):
        raise IOError(f"Cache file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Verify provenance exists
    if 'provenance' not in data:
        raise IOError("Cache file missing provenance")

    # Verify SHA256 if present
    if 'sha256' in data['provenance']:
        raw_data = json.dumps({
            'absolute': data.get('absolute', []),
            'excess': data.get('excess', [])
        }, sort_keys=True)

        computed_sha256 = hashlib.sha256(raw_data.encode()).hexdigest()
        stored_sha256 = data['provenance']['sha256']

        if computed_sha256 != stored_sha256:
            raise IOError(f"SHA256 mismatch: cache corrupted")

    return data


def get_cached_date_range(cache_dir: str, sec_id: str) -> Optional[Tuple[str, str]]:
    """
    Find the date range of cached returns for a security.

    Args:
        cache_dir: Directory containing cache files
        sec_id: Security ID

    Returns:
        (start_date, end_date) or None if not cached
    """
    if not os.path.exists(cache_dir):
        return None

    # Look for cache files matching this sec_id
    for filename in os.listdir(cache_dir):
        if not filename.endswith('.json'):
            continue

        try:
            filepath = os.path.join(cache_dir, filename)
            data = load_returns_cache(filepath)

            # Check if this sec_id is in the data
            for record in data.get('absolute', []):
                if record['sec_id'] == sec_id:
                    prov = data['provenance']
                    return (prov['data_start_date'], prov['data_end_date'])
        except Exception:
            continue

    return None


def build_ticker_mapping(
    sec_ids: List[str]
) -> Dict[str, str]:
    """
    Build mapping from SecId to ticker symbol.

    Args:
        sec_ids: List of security IDs

    Returns:
        {sec_id: ticker} mapping
    """
    available, msg = check_availability()
    if not available:
        raise MorningstarReturnsError(f"Morningstar not available: {msg}")

    mapping = {}

    try:
        # Use get_investment_data to get tickers
        data = md.direct.get_investment_data(
            investments=sec_ids,
            data_points=['Ticker', 'Name']
        )

        if data is not None and not data.empty:
            for idx, row in data.iterrows():
                sec_id = row.get('SecId') or row.get('Id')
                ticker = row.get('Ticker')
                if sec_id and ticker:
                    mapping[sec_id] = ticker
    except Exception as e:
        # If this fails, we'll just use sec_ids as tickers
        print(f"Warning: Could not build ticker mapping: {e}")
        for sec_id in sec_ids:
            mapping[sec_id] = sec_id

    return mapping


# Example usage
if __name__ == '__main__':
    # Test the module
    available, msg = check_availability()
    print(f"Morningstar availability: {available} - {msg}")

    if available:
        # Test with VRTX
        test_sec_ids = ['0P000005R7']  # VRTX

        print("\nFetching test returns data...")
        try:
            data = fetch_returns(
                sec_ids=test_sec_ids,
                start_date='2024-01-01',
                end_date='2024-12-31',
                frequency='monthly',
                benchmark_sec_id='FEUSA04AER'  # XBI
            )

            print(f"Retrieved {data['provenance']['num_observations']} observations")
            print(f"   SHA256: {data['provenance']['sha256'][:16]}...")
            print(f"\nSample records (first 3):")
            for record in data['absolute'][:3]:
                print(f"   {record}")

            # Test caching
            print("\nTesting cache operations...")
            cache_path = save_returns_cache(
                data,
                output_dir='/tmp/test_cache',
                filename='vrtx_2024.json'
            )
            print(f"Saved to: {cache_path}")

            loaded = load_returns_cache(cache_path)
            print(f"Loaded and verified SHA256")

        except MorningstarReturnsError as e:
            print(f"Error: {e}")
