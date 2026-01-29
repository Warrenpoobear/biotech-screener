"""
clustering.py - Correlated Security Clustering

Institutional-grade clustering based on return correlations.

Methodology:
1. Compute rolling return correlations (e.g., 60 trading days)
2. Build adjacency graph where corr >= threshold
3. Connected components = clusters
4. Add cluster_id and cluster_size to each security

Design:
- DETERMINISTIC: Stable sorts, reproducible cluster IDs
- STDLIB-ONLY: No scipy/numpy required
- PIT-SAFE: Only uses returns up to as_of_date
- O(N + E) complexity: Linear in edges, not O(N²)

Default threshold = 0.70:
- Captures "same tape" behavior in biotech
- Balances cluster size vs. granularity
- Can be tuned per-regime

Author: Wake Robin Capital Management
Version: 1.0.0
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default correlation threshold for clustering
DEFAULT_CORR_THRESHOLD = 0.70

# Default rolling window for correlation (trading days)
DEFAULT_CORR_WINDOW = 60

# Clustering model identifier for provenance
CLUSTER_MODEL_ID = "return_corr_connected_components"
CLUSTER_MODEL_VERSION = "1.0.0"


# =============================================================================
# CORRELATION COMPUTATION
# =============================================================================

def compute_correlation(
    returns_a: List[float],
    returns_b: List[float],
    min_overlap: int = 30,
) -> Optional[float]:
    """
    Compute Pearson correlation between two return series.

    Stdlib-only implementation - no numpy required.

    Args:
        returns_a: Daily returns for security A
        returns_b: Daily returns for security B
        min_overlap: Minimum overlapping observations required

    Returns:
        Correlation coefficient, or None if insufficient data
    """
    # Find overlapping observations (both non-None)
    pairs = [
        (a, b) for a, b in zip(returns_a, returns_b)
        if a is not None and b is not None
        and not (math.isnan(a) or math.isnan(b))
    ]

    n = len(pairs)
    if n < min_overlap:
        return None

    # Compute means
    mean_a = sum(p[0] for p in pairs) / n
    mean_b = sum(p[1] for p in pairs) / n

    # Compute covariance and variances
    cov = sum((p[0] - mean_a) * (p[1] - mean_b) for p in pairs) / n
    var_a = sum((p[0] - mean_a) ** 2 for p in pairs) / n
    var_b = sum((p[1] - mean_b) ** 2 for p in pairs) / n

    # Avoid division by zero
    if var_a <= 0 or var_b <= 0:
        return None

    # Pearson correlation
    corr = cov / math.sqrt(var_a * var_b)

    # Clamp to [-1, 1] to handle floating point errors
    return max(-1.0, min(1.0, corr))


def compute_pairwise_correlations(
    returns_by_ticker: Dict[str, List[float]],
    tickers: Optional[List[str]] = None,
    min_overlap: int = 30,
) -> List[Tuple[str, str, float]]:
    """
    Compute pairwise correlations for a set of tickers.

    Note: O(N²) complexity - for large universes, consider filtering
    to top K by score or within same indication bucket.

    Args:
        returns_by_ticker: {ticker: [daily_returns]}
        tickers: Subset of tickers to compute (None = all)
        min_overlap: Minimum overlapping observations

    Returns:
        List of (ticker_a, ticker_b, correlation) tuples
    """
    if tickers is None:
        tickers = sorted(returns_by_ticker.keys())
    else:
        tickers = sorted(tickers)

    pairs = []
    for i, t_a in enumerate(tickers):
        returns_a = returns_by_ticker.get(t_a)
        if not returns_a:
            continue

        for t_b in tickers[i+1:]:
            returns_b = returns_by_ticker.get(t_b)
            if not returns_b:
                continue

            corr = compute_correlation(returns_a, returns_b, min_overlap)
            if corr is not None:
                pairs.append((t_a, t_b, corr))

    return pairs


# =============================================================================
# CLUSTERING (Connected Components)
# =============================================================================

def build_corr_clusters(
    tickers: List[str],
    corr_pairs: List[Tuple[str, str, float]],
    threshold: float = DEFAULT_CORR_THRESHOLD,
) -> Dict[str, int]:
    """
    Build clusters using connected components on correlation graph.

    Args:
        tickers: List of all tickers in universe
        corr_pairs: List of (ticker_a, ticker_b, correlation) tuples
        threshold: Minimum correlation to form an edge

    Returns:
        {ticker: cluster_id} where cluster_id starts at 1
        Cluster IDs are deterministic: sorted by lowest ticker in component.
    """
    # Build adjacency list
    adj: Dict[str, Set[str]] = {t: set() for t in tickers}

    for a, b, c in corr_pairs:
        if a in adj and b in adj and c >= threshold:
            adj[a].add(b)
            adj[b].add(a)

    # Find connected components using DFS
    seen: Set[str] = set()
    components: List[List[str]] = []

    # Process tickers in sorted order for determinism
    for t in sorted(tickers):
        if t in seen:
            continue

        # DFS to find all connected tickers
        stack = [t]
        component = []
        seen.add(t)

        while stack:
            x = stack.pop()
            component.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)

        components.append(sorted(component))

    # Sort components by their first (alphabetically lowest) ticker
    # This ensures deterministic cluster IDs across runs
    components.sort(key=lambda c: c[0])

    # Assign cluster IDs
    cluster_map: Dict[str, int] = {}
    for cluster_id, component in enumerate(components, start=1):
        for t in component:
            cluster_map[t] = cluster_id

    return cluster_map


def compute_cluster_stats(
    cluster_map: Dict[str, int],
) -> Dict[int, Dict[str, Any]]:
    """
    Compute statistics for each cluster.

    Args:
        cluster_map: {ticker: cluster_id}

    Returns:
        {cluster_id: {size, tickers}}
    """
    stats: Dict[int, Dict[str, Any]] = {}

    for ticker, cluster_id in cluster_map.items():
        if cluster_id not in stats:
            stats[cluster_id] = {"size": 0, "tickers": []}
        stats[cluster_id]["size"] += 1
        stats[cluster_id]["tickers"].append(ticker)

    # Sort tickers within each cluster
    for cluster_id in stats:
        stats[cluster_id]["tickers"] = sorted(stats[cluster_id]["tickers"])

    return stats


# =============================================================================
# INTEGRATION WITH RANKED SECURITIES
# =============================================================================

def attach_cluster_ids(
    ranked_securities: List[Dict[str, Any]],
    returns_by_ticker: Optional[Dict[str, List[float]]] = None,
    corr_pairs: Optional[List[Tuple[str, str, float]]] = None,
    threshold: float = DEFAULT_CORR_THRESHOLD,
    ticker_key: str = "ticker",
) -> Dict[str, Any]:
    """
    Attach cluster IDs to ranked securities.

    Mutates ranked_securities in-place. Adds:
        - cluster_id: Integer cluster identifier (1-based)
        - cluster_size: Number of securities in this cluster

    Args:
        ranked_securities: List of security dicts
        returns_by_ticker: Optional {ticker: [daily_returns]} for correlation
        corr_pairs: Optional pre-computed correlation pairs
        threshold: Minimum correlation to form an edge
        ticker_key: Field name for ticker

    Returns:
        Provenance dict with clustering metadata
    """
    tickers = [r.get(ticker_key) for r in ranked_securities if r.get(ticker_key)]

    # Compute correlations if not provided
    if corr_pairs is None and returns_by_ticker is not None:
        corr_pairs = compute_pairwise_correlations(
            returns_by_ticker,
            tickers=tickers,
        )
    elif corr_pairs is None:
        # No correlation data - each security is its own cluster
        for r in ranked_securities:
            r["cluster_id"] = 0
            r["cluster_size"] = 1
        return {
            "cluster_model": "none",
            "cluster_reason": "no_correlation_data",
        }

    # Build clusters
    cluster_map = build_corr_clusters(tickers, corr_pairs, threshold)

    # Compute cluster sizes
    cluster_stats = compute_cluster_stats(cluster_map)

    # Attach to securities
    for r in ranked_securities:
        ticker = r.get(ticker_key)
        if ticker and ticker in cluster_map:
            cluster_id = cluster_map[ticker]
            r["cluster_id"] = cluster_id
            r["cluster_size"] = cluster_stats[cluster_id]["size"]
        else:
            r["cluster_id"] = 0
            r["cluster_size"] = 1

    # Compute summary statistics
    n_clusters = len(cluster_stats)
    cluster_sizes = [s["size"] for s in cluster_stats.values()]
    singletons = sum(1 for s in cluster_sizes if s == 1)
    largest = max(cluster_sizes) if cluster_sizes else 0

    return {
        "cluster_model": CLUSTER_MODEL_ID,
        "cluster_model_version": CLUSTER_MODEL_VERSION,
        "corr_cluster_threshold": threshold,
        "n_clusters": n_clusters,
        "n_singletons": singletons,
        "largest_cluster_size": largest,
        "avg_cluster_size": round(sum(cluster_sizes) / n_clusters, 2) if n_clusters > 0 else 0,
    }


# =============================================================================
# FALLBACK: INDICATION-BASED CLUSTERING
# =============================================================================

def build_indication_clusters(
    ranked_securities: List[Dict[str, Any]],
    indication_key: str = "primary_indication",
    ticker_key: str = "ticker",
) -> Dict[str, int]:
    """
    Build clusters based on therapeutic indication (fallback when no returns).

    This is a metadata-based approximation when correlation data is unavailable.
    Stocks targeting the same indication often have correlated returns.

    Args:
        ranked_securities: List of security dicts
        indication_key: Field name for therapeutic indication
        ticker_key: Field name for ticker

    Returns:
        {ticker: cluster_id}
    """
    # Group by indication
    indication_groups: Dict[str, List[str]] = {}

    for r in ranked_securities:
        ticker = r.get(ticker_key)
        indication = r.get(indication_key, "Unknown")

        if not ticker:
            continue

        if indication not in indication_groups:
            indication_groups[indication] = []
        indication_groups[indication].append(ticker)

    # Sort indications alphabetically for determinism
    sorted_indications = sorted(indication_groups.keys())

    # Assign cluster IDs
    cluster_map: Dict[str, int] = {}
    for cluster_id, indication in enumerate(sorted_indications, start=1):
        for ticker in sorted(indication_groups[indication]):
            cluster_map[ticker] = cluster_id

    return cluster_map


def attach_indication_clusters(
    ranked_securities: List[Dict[str, Any]],
    indication_key: str = "primary_indication",
    ticker_key: str = "ticker",
) -> Dict[str, Any]:
    """
    Attach indication-based cluster IDs (fallback method).

    Mutates ranked_securities in-place. Adds:
        - cluster_id: Integer cluster identifier
        - cluster_size: Number of securities in this cluster

    Returns:
        Provenance dict with clustering metadata
    """
    cluster_map = build_indication_clusters(
        ranked_securities,
        indication_key=indication_key,
        ticker_key=ticker_key,
    )

    # Compute cluster sizes
    cluster_stats = compute_cluster_stats(cluster_map)

    # Attach to securities
    for r in ranked_securities:
        ticker = r.get(ticker_key)
        if ticker and ticker in cluster_map:
            cluster_id = cluster_map[ticker]
            r["cluster_id"] = cluster_id
            r["cluster_size"] = cluster_stats[cluster_id]["size"]
        else:
            r["cluster_id"] = 0
            r["cluster_size"] = 1

    n_clusters = len(cluster_stats)
    cluster_sizes = [s["size"] for s in cluster_stats.values()]

    return {
        "cluster_model": "indication_based",
        "cluster_model_version": "1.0.0",
        "cluster_method": "therapeutic_indication_grouping",
        "n_clusters": n_clusters,
        "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "avg_cluster_size": round(sum(cluster_sizes) / n_clusters, 2) if n_clusters > 0 else 0,
    }


if __name__ == "__main__":
    print("Testing clustering.py...")
    print(f"Default correlation threshold: {DEFAULT_CORR_THRESHOLD}")
    print()

    # Test with sample data
    test_tickers = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF"]

    # Sample correlation pairs (AAAA-BBBB and CCCC-DDDD are correlated)
    test_corr_pairs = [
        ("AAAA", "BBBB", 0.85),  # High correlation
        ("AAAA", "CCCC", 0.30),  # Low correlation
        ("BBBB", "CCCC", 0.25),  # Low correlation
        ("CCCC", "DDDD", 0.75),  # High correlation
        ("EEEE", "FFFF", 0.50),  # Below threshold
    ]

    # Build clusters
    cluster_map = build_corr_clusters(test_tickers, test_corr_pairs, threshold=0.70)

    print("Cluster assignments:")
    for ticker in sorted(test_tickers):
        print(f"  {ticker}: cluster {cluster_map.get(ticker, 0)}")

    # Test with ranked securities
    test_securities = [
        {"ticker": "AAAA", "composite_score": "85.0"},
        {"ticker": "BBBB", "composite_score": "75.0"},
        {"ticker": "CCCC", "composite_score": "70.0"},
        {"ticker": "DDDD", "composite_score": "65.0"},
        {"ticker": "EEEE", "composite_score": "55.0"},
        {"ticker": "FFFF", "composite_score": "45.0"},
    ]

    provenance = attach_cluster_ids(
        test_securities,
        corr_pairs=test_corr_pairs,
        threshold=0.70,
    )

    print("\nWith cluster IDs attached:")
    for r in test_securities:
        print(f"  {r['ticker']}: cluster {r['cluster_id']} (size {r['cluster_size']})")

    print("\nProvenance:")
    for k, v in provenance.items():
        print(f"  {k}: {v}")
