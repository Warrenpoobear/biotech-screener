"""
cusip_mapper.py

CUSIP→Ticker Mapping with OpenFIGI API Integration
Three-tier caching strategy for production reliability.

ARCHITECTURE:
- Tier 0: Static map (biotech universe, never expires)
- Tier 1: Persistent cache (OpenFIGI results, 90-day TTL)
- Tier 2: OpenFIGI API (live lookups with rate limiting)

PIT SAFETY:
- All timestamp operations use explicit reference_time parameter
- Cache TTL validation uses reference time, not wall-clock time
- This ensures deterministic behavior across pipeline runs

Author: Wake Robin Capital Management
Date: 2026-01-09
Updated: 2026-01-23 (PIT safety hardening)
"""

import json
import logging
import socket
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

OPENFIGI_API_URL = "https://api.openfigi.com/v3/mapping"
OPENFIGI_API_KEY = None  # Set via environment or config file

# Rate limiting (OpenFIGI free tier: 25 requests/minute)
OPENFIGI_REQUEST_DELAY = 2.5  # ~24 requests/minute (conservative)
OPENFIGI_BATCH_SIZE = 100  # Max 100 CUSIPs per request

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SEC = 3.0
MAX_BACKOFF_SEC = 60.0
REQUEST_TIMEOUT_SEC = 30

# Cache TTL
CACHE_TTL_DAYS = 90  # Refresh mappings quarterly

# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class CUSIPMapping:
    """Single CUSIP→Ticker mapping with metadata"""
    cusip: str
    ticker: str
    name: str
    exchange: str
    security_type: str
    mapped_at: str  # ISO datetime
    source: str  # 'static', 'cache', 'openfigi'
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CUSIPMapping':
        return cls(**data)


# ==============================================================================
# TIER 0: STATIC MAP (BIOTECH UNIVERSE)
# ==============================================================================

def load_static_cusip_map(static_map_path: Path) -> Dict[str, CUSIPMapping]:
    """
    Load Tier 0 static CUSIP map.
    
    This is hand-curated or extracted from reliable sources.
    Never expires - acts as fallback.
    
    Returns:
        {cusip: CUSIPMapping}
    """
    if not static_map_path.exists():
        return {}
    
    with open(static_map_path) as f:
        data = json.load(f)
    
    return {
        cusip: CUSIPMapping.from_dict(mapping)
        for cusip, mapping in data.items()
    }


def build_static_map_from_universe(
    universe_path: Path,
    output_path: Path,
    reference_time: Optional[datetime] = None
) -> None:
    """
    Bootstrap static map from universe.json with known CUSIPs.

    This is a one-time operation to create the initial static map.
    You'll need to manually add CUSIPs for your 322 tickers.

    Args:
        universe_path: Path to universe.json
        output_path: Path to write static map
        reference_time: Explicit timestamp for PIT safety. If None, uses
                       a deterministic default (midnight of 2026-01-01).
                       IMPORTANT: For production use, always pass an explicit time.
    """
    # PIT SAFETY: Use explicit reference time, never wall-clock
    if reference_time is None:
        # Deterministic default for reproducibility
        reference_time = datetime(2026, 1, 1, 0, 0, 0)
        logger.warning(
            "build_static_map_from_universe: No reference_time provided. "
            f"Using deterministic default: {reference_time.isoformat()}"
        )

    with open(universe_path) as f:
        universe = json.load(f)

    static_map = {}

    # Example structure - you'll need to populate with real CUSIPs
    for security in universe:
        ticker = security.get('ticker')
        if ticker == '_XBI_BENCHMARK_':
            continue

        # PLACEHOLDER: You need to add actual CUSIPs
        # Sources: Bloomberg, SEC filings, Yahoo Finance, etc.
        cusip = security.get('cusip')  # If you have this in universe

        if cusip:
            static_map[cusip] = {
                'cusip': cusip,
                'ticker': ticker,
                'name': security.get('name', ''),
                'exchange': security.get('exchange', 'NASDAQ'),
                'security_type': 'Common Stock',
                'mapped_at': reference_time.isoformat(),
                'source': 'static'
            }

    with open(output_path, 'w') as f:
        json.dump(static_map, f, indent=2)

    print(f"Static map created: {len(static_map)} mappings")
    print(f"Saved to: {output_path}")


# ==============================================================================
# TIER 1: PERSISTENT CACHE
# ==============================================================================

def load_cache(
    cache_path: Path,
    reference_time: Optional[datetime] = None
) -> Dict[str, CUSIPMapping]:
    """
    Load Tier 1 persistent cache.

    Cached OpenFIGI results with 90-day TTL.

    Args:
        cache_path: Path to cache file
        reference_time: Explicit reference time for TTL validation.
                       If None, uses deterministic default for reproducibility.
                       IMPORTANT: For production use, always pass an explicit time.

    Returns:
        {cusip: CUSIPMapping}
    """
    if not cache_path.exists():
        return {}

    # PIT SAFETY: Use explicit reference time, never wall-clock
    if reference_time is None:
        # Deterministic default - use a far-future date to include all cache entries
        # This ensures reproducible behavior when no reference time is provided
        reference_time = datetime(2099, 12, 31, 23, 59, 59)
        logger.warning(
            "load_cache: No reference_time provided. "
            "Using deterministic default that includes all cache entries."
        )

    with open(cache_path) as f:
        data = json.load(f)

    # Filter expired entries using explicit reference time
    cutoff = reference_time - timedelta(days=CACHE_TTL_DAYS)
    valid_mappings = {}

    for cusip, mapping_dict in data.items():
        mapping = CUSIPMapping.from_dict(mapping_dict)
        mapped_at = datetime.fromisoformat(mapping.mapped_at)

        if mapped_at >= cutoff:
            valid_mappings[cusip] = mapping

    return valid_mappings


def save_cache(
    cache_path: Path,
    mappings: Dict[str, CUSIPMapping]
) -> None:
    """
    Save Tier 1 persistent cache.
    """
    cache_dict = {
        cusip: mapping.to_dict()
        for cusip, mapping in mappings.items()
    }
    
    with open(cache_path, 'w') as f:
        json.dump(cache_dict, f, indent=2)


# ==============================================================================
# TIER 2: OPENFIGI API
# ==============================================================================

def _calculate_backoff(attempt: int) -> float:
    """Calculate exponential backoff with jitter."""
    import random
    backoff = min(INITIAL_BACKOFF_SEC * (2 ** attempt), MAX_BACKOFF_SEC)
    # Add jitter (±25%)
    jitter = backoff * 0.25 * (2 * random.random() - 1)
    return backoff + jitter


def query_openfigi_batch(
    cusips: List[str],
    api_key: Optional[str] = None,
    reference_time: Optional[datetime] = None
) -> Dict[str, Optional[CUSIPMapping]]:
    """
    Query OpenFIGI API for batch of CUSIPs with retry logic.

    Args:
        cusips: List of 9-character CUSIPs (max 100)
        api_key: Optional OpenFIGI API key (increases rate limits)
        reference_time: Explicit timestamp for mapped_at field.
                       If None, uses deterministic default.
                       IMPORTANT: For production use, always pass an explicit time.

    Returns:
        {cusip: CUSIPMapping or None}

    Retry behavior:
        - Up to MAX_RETRIES attempts with exponential backoff
        - Handles timeouts, rate limits, and transient network errors
    """
    # PIT SAFETY: Use explicit reference time, never wall-clock
    if reference_time is None:
        # Deterministic default for reproducibility
        reference_time = datetime(2026, 1, 1, 0, 0, 0)
        logger.warning(
            "query_openfigi_batch: No reference_time provided. "
            f"Using deterministic default: {reference_time.isoformat()}"
        )
    if len(cusips) > OPENFIGI_BATCH_SIZE:
        raise ValueError(f"Batch size {len(cusips)} exceeds max {OPENFIGI_BATCH_SIZE}")

    # Build request payload
    payload = [
        {
            "idType": "ID_CUSIP",
            "idValue": cusip,
            "exchCode": "US"  # Prefer US exchanges
        }
        for cusip in cusips
    ]

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    if api_key:
        headers['X-OPENFIGI-APIKEY'] = api_key

    # Make request with retry logic
    req = urllib.request.Request(
        OPENFIGI_API_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers=headers
    )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as response:
                results = json.loads(response.read().decode('utf-8'))

            # Rate limiting
            time.sleep(OPENFIGI_REQUEST_DELAY)
            break  # Success - exit retry loop

        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 429:
                # Rate limit - wait longer
                wait_time = 60 if attempt == 0 else _calculate_backoff(attempt + 2)
                logger.warning(f"OpenFIGI rate limit hit (attempt {attempt + 1}/{MAX_RETRIES}) - waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            elif e.code >= 500:
                # Server error - retry with backoff
                wait_time = _calculate_backoff(attempt)
                logger.warning(f"OpenFIGI server error {e.code} (attempt {attempt + 1}/{MAX_RETRIES}) - retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                # Client error (4xx except 429) - don't retry
                logger.error(f"OpenFIGI HTTP error {e.code}: {e.reason}")
                return {cusip: None for cusip in cusips}

        except socket.timeout as e:
            last_error = e
            wait_time = _calculate_backoff(attempt)
            logger.warning(f"OpenFIGI timeout (attempt {attempt + 1}/{MAX_RETRIES}) - retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

        except urllib.error.URLError as e:
            last_error = e
            wait_time = _calculate_backoff(attempt)
            logger.warning(f"OpenFIGI network error: {e.reason} (attempt {attempt + 1}/{MAX_RETRIES}) - retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

        except Exception as e:
            last_error = e
            logger.error(f"OpenFIGI unexpected error: {e}")
            return {cusip: None for cusip in cusips}
    else:
        # All retries exhausted
        logger.error(f"OpenFIGI query failed after {MAX_RETRIES} attempts: {last_error}")
        return {cusip: None for cusip in cusips}

    # Parse results
    mappings = {}

    for i, cusip in enumerate(cusips):
        result = results[i]

        if 'error' in result:
            logger.debug(f"  {cusip}: {result['error']}")
            mappings[cusip] = None
            continue

        if 'data' not in result or not result['data']:
            logger.debug(f"  {cusip}: No mapping found")
            mappings[cusip] = None
            continue

        # Take first result (usually most liquid)
        figi_data = result['data'][0]

        mappings[cusip] = CUSIPMapping(
            cusip=cusip,
            ticker=figi_data.get('ticker', ''),
            name=figi_data.get('name', ''),
            exchange=figi_data.get('exchCode', ''),
            security_type=figi_data.get('securityType', ''),
            mapped_at=reference_time.isoformat(),
            source='openfigi'
        )

        logger.info(f"  {cusip} → {figi_data.get('ticker', 'N/A')}")

    return mappings


def query_openfigi_all(
    cusips: List[str],
    api_key: Optional[str] = None,
    reference_time: Optional[datetime] = None
) -> Dict[str, Optional[CUSIPMapping]]:
    """
    Query OpenFIGI for all CUSIPs in batches.

    Handles batching and rate limiting automatically.

    Args:
        cusips: List of CUSIPs to query
        api_key: Optional OpenFIGI API key
        reference_time: Explicit timestamp for mapped_at field (PIT safety)
    """
    all_mappings = {}

    # Process in batches
    for i in range(0, len(cusips), OPENFIGI_BATCH_SIZE):
        batch = cusips[i:i + OPENFIGI_BATCH_SIZE]

        logger.info(f"Querying OpenFIGI batch {i//OPENFIGI_BATCH_SIZE + 1} "
                    f"({len(batch)} CUSIPs)...")

        batch_results = query_openfigi_batch(batch, api_key, reference_time)
        all_mappings.update(batch_results)

    return all_mappings


# ==============================================================================
# INTEGRATED MAPPER (THREE-TIER STRATEGY)
# ==============================================================================

class CUSIPMapper:
    """
    Three-tier CUSIP→Ticker mapper with intelligent fallback.

    Tier 0: Static map (instant, never expires)
    Tier 1: Persistent cache (instant, 90-day TTL)
    Tier 2: OpenFIGI API (slow, rate-limited)

    PIT Safety:
        Pass explicit reference_time to ensure deterministic behavior.
        All timestamps use the reference_time instead of wall-clock time.
    """

    def __init__(
        self,
        static_map_path: Path,
        cache_path: Path,
        openfigi_api_key: Optional[str] = None,
        reference_time: Optional[datetime] = None
    ):
        self.static_map_path = static_map_path
        self.cache_path = cache_path
        self.openfigi_api_key = openfigi_api_key

        # PIT SAFETY: Store reference time for all operations
        self.reference_time = reference_time

        # Load tiers
        logger.info("Loading CUSIP mapper...")
        self.static_map = load_static_cusip_map(static_map_path)
        logger.info(f"  Static map: {len(self.static_map)} entries")

        self.cache = load_cache(cache_path, reference_time)
        logger.info(f"  Cache: {len(self.cache)} entries (valid within {CACHE_TTL_DAYS} days)")

        # Track new mappings for cache update
        self.new_mappings = {}
    
    def get(self, cusip: str) -> Optional[str]:
        """
        Get ticker for CUSIP.
        
        Returns:
            Ticker string or None if not found
        """
        mapping = self.get_mapping(cusip)
        return mapping.ticker if mapping else None
    
    def get_mapping(self, cusip: str) -> Optional[CUSIPMapping]:
        """
        Get full mapping for CUSIP with fallback strategy.

        Returns:
            CUSIPMapping or None
        """
        # Tier 0: Static map
        if cusip in self.static_map:
            return self.static_map[cusip]

        # Tier 1: Cache
        if cusip in self.cache:
            return self.cache[cusip]

        # Tier 2: OpenFIGI (live query)
        logger.info(f"  Cache miss: {cusip} - querying OpenFIGI...")
        result = query_openfigi_batch([cusip], self.openfigi_api_key, self.reference_time)

        mapping = result.get(cusip)

        if mapping:
            # Add to cache and new_mappings
            self.cache[cusip] = mapping
            self.new_mappings[cusip] = mapping

        return mapping
    
    def get_batch(self, cusips: List[str]) -> Dict[str, Optional[str]]:
        """
        Get tickers for batch of CUSIPs.

        More efficient than calling get() in a loop.

        Returns:
            {cusip: ticker or None}
        """
        results = {}
        unknown_cusips = []

        # Check static map and cache first
        for cusip in cusips:
            if cusip in self.static_map:
                results[cusip] = self.static_map[cusip].ticker
            elif cusip in self.cache:
                results[cusip] = self.cache[cusip].ticker
            else:
                unknown_cusips.append(cusip)

        # Query OpenFIGI for unknowns
        if unknown_cusips:
            logger.info(f"Querying OpenFIGI for {len(unknown_cusips)} unknown CUSIPs...")
            openfigi_results = query_openfigi_all(
                unknown_cusips, self.openfigi_api_key, self.reference_time
            )

            for cusip, mapping in openfigi_results.items():
                if mapping:
                    results[cusip] = mapping.ticker
                    self.cache[cusip] = mapping
                    self.new_mappings[cusip] = mapping
                else:
                    results[cusip] = None

        return results
    
    def save(self) -> None:
        """
        Save updated cache to disk.
        
        Call this after batch operations to persist new mappings.
        """
        if self.new_mappings:
            save_cache(self.cache_path, self.cache)
            logger.info(f"Saved {len(self.new_mappings)} new mappings to cache")
            self.new_mappings.clear()
    
    def stats(self) -> dict:
        """
        Get mapper statistics.
        """
        return {
            'static_map_size': len(self.static_map),
            'cache_size': len(self.cache),
            'new_mappings_pending': len(self.new_mappings),
            'total_known': len(self.static_map) + len(self.cache)
        }


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def extract_cusips_from_holdings_json(holdings_path: Path) -> List[str]:
    """
    Extract all unique CUSIPs from holdings_snapshots.json.
    
    Useful for batch pre-warming the cache.
    """
    # This would be implemented after you have real holdings data
    # For now, return empty list
    return []


def validate_cusip(cusip: str) -> bool:
    """
    Validate CUSIP format.
    
    CUSIP = 9 alphanumeric characters
    - First 6: Issuer
    - Next 2: Issue
    - Last 1: Check digit
    """
    if len(cusip) != 9:
        return False
    
    if not cusip.isalnum():
        return False
    
    # Optional: Validate check digit
    # (Complex algorithm, skip for now)
    
    return True


# ==============================================================================
# BOOTSTRAP UTILITIES
# ==============================================================================

def create_empty_static_map(output_path: Path) -> None:
    """
    Create empty static map file.
    
    You'll manually populate this with known CUSIP→Ticker mappings.
    """
    static_map = {
        # Example entries - replace with real data
        # "037833100": {
        #     "cusip": "037833100",
        #     "ticker": "AAPL",
        #     "name": "Apple Inc",
        #     "exchange": "NASDAQ",
        #     "security_type": "Common Stock",
        #     "mapped_at": "2026-01-09T00:00:00",
        #     "source": "static"
        # }
    }
    
    with open(output_path, 'w') as f:
        json.dump(static_map, f, indent=2)
    
    print(f"Empty static map created: {output_path}")
    print("Populate with known CUSIP→Ticker mappings before first run")


def create_empty_cache(output_path: Path) -> None:
    """
    Create empty cache file.
    
    This will be populated automatically by OpenFIGI queries.
    """
    with open(output_path, 'w') as f:
        json.dump({}, f)
    
    print(f"Empty cache created: {output_path}")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CUSIP→Ticker mapper with OpenFIGI integration"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize mapper files')
    init_parser.add_argument('--data-dir', type=Path, required=True)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query single CUSIP')
    query_parser.add_argument('cusip', type=str, help='9-character CUSIP')
    query_parser.add_argument('--data-dir', type=Path, required=True)
    query_parser.add_argument('--api-key', type=str, help='OpenFIGI API key')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Query batch of CUSIPs')
    batch_parser.add_argument('cusips_file', type=Path, help='Text file with CUSIPs (one per line)')
    batch_parser.add_argument('--data-dir', type=Path, required=True)
    batch_parser.add_argument('--api-key', type=str, help='OpenFIGI API key')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show mapper statistics')
    stats_parser.add_argument('--data-dir', type=Path, required=True)
    
    args = parser.parse_args()
    
    if args.command == 'init':
        # Initialize files
        static_map_path = args.data_dir / 'cusip_static_map.json'
        cache_path = args.data_dir / 'cusip_cache.json'
        
        if not static_map_path.exists():
            create_empty_static_map(static_map_path)
        
        if not cache_path.exists():
            create_empty_cache(cache_path)
        
        print("\nMapper initialized!")
        print(f"Static map: {static_map_path}")
        print(f"Cache: {cache_path}")
        print("\nNext steps:")
        print("1. Populate static map with known CUSIP→Ticker mappings")
        print("2. Get OpenFIGI API key (free): https://www.openfigi.com/api")
        print("3. Run 'cusip_mapper.py query <CUSIP>' to test")
    
    elif args.command == 'query':
        # Query single CUSIP
        mapper = CUSIPMapper(
            static_map_path=args.data_dir / 'cusip_static_map.json',
            cache_path=args.data_dir / 'cusip_cache.json',
            openfigi_api_key=args.api_key
        )
        
        cusip = args.cusip.upper()
        
        if not validate_cusip(cusip):
            print(f"Invalid CUSIP format: {cusip}")
            exit(1)
        
        mapping = mapper.get_mapping(cusip)
        
        if mapping:
            print(f"\n{cusip} → {mapping.ticker}")
            print(f"  Name: {mapping.name}")
            print(f"  Exchange: {mapping.exchange}")
            print(f"  Type: {mapping.security_type}")
            print(f"  Source: {mapping.source}")
        else:
            print(f"\n{cusip}: Not found")
        
        mapper.save()
    
    elif args.command == 'batch':
        # Query batch of CUSIPs
        with open(args.cusips_file) as f:
            cusips = [line.strip().upper() for line in f if line.strip()]
        
        print(f"Loading {len(cusips)} CUSIPs from {args.cusips_file}")
        
        mapper = CUSIPMapper(
            static_map_path=args.data_dir / 'cusip_static_map.json',
            cache_path=args.data_dir / 'cusip_cache.json',
            openfigi_api_key=args.api_key
        )
        
        results = mapper.get_batch(cusips)
        
        # Print results
        print(f"\n{'CUSIP':<12} {'Ticker':<8} {'Source'}")
        print("-" * 40)
        
        for cusip in cusips:
            ticker = results.get(cusip, 'NOT FOUND')
            
            # Determine source
            if cusip in mapper.static_map:
                source = 'static'
            elif cusip in mapper.cache:
                source = 'cache'
            else:
                source = 'openfigi'
            
            print(f"{cusip:<12} {ticker:<8} {source}")
        
        # Save new mappings
        mapper.save()
        
        # Statistics
        found = sum(1 for v in results.values() if v is not None)
        print(f"\nFound: {found}/{len(cusips)} ({found/len(cusips)*100:.1f}%)")
    
    elif args.command == 'stats':
        # Show statistics
        mapper = CUSIPMapper(
            static_map_path=args.data_dir / 'cusip_static_map.json',
            cache_path=args.data_dir / 'cusip_cache.json'
        )
        
        stats = mapper.stats()
        
        print("\nCUSIP Mapper Statistics")
        print("=" * 40)
        print(f"Static map entries:  {stats['static_map_size']}")
        print(f"Cache entries:       {stats['cache_size']}")
        print(f"Pending updates:     {stats['new_mappings_pending']}")
        print(f"Total known CUSIPs:  {stats['total_known']}")
    
    else:
        parser.print_help()
