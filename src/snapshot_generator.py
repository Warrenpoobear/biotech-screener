"""
Snapshot Generator for Wake Robin Biotech Alpha System.

Generates deterministic, PIT-safe snapshots by orchestrating data providers
and scoring modules.

Usage:
    # Stub mode (baseline)
    python -m src.snapshot_generator --as-of 2024-01-31 \\
        --universe data/universe/biotech_universe_v1.csv \\
        --prices data/stooq_prices.csv \\
        --output snapshots/stub

    # AACT mode for Modules 3-4
    python -m src.snapshot_generator --as-of 2024-01-31 \\
        --universe data/universe/biotech_universe_v1.csv \\
        --prices data/stooq_prices.csv \\
        --clinical-provider aact \\
        --aact-snapshots data/aact_snapshots \\
        --trial-map data/trial_mapping.csv \\
        --output snapshots/aact
"""

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from .common.hash_utils import (
    compute_hash,
    compute_snapshot_id,
    compute_trial_facts_hash,
    stable_json_dumps,
)
from .providers import (
    AACTClinicalTrialsProvider,
    ClinicalTrialsProvider,
    ProviderResult,
    StubClinicalTrialsProvider,
    TrialRow,
)
from .providers.aact_provider import load_trial_mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SnapshotConfig:
    """Configuration for snapshot generation."""
    as_of_date: date
    pit_lag_days: int = 1  # Default: 1 day lag for PIT safety
    
    # Paths
    universe_file: Path = field(default_factory=lambda: Path("data/universe/biotech_universe_v1.csv"))
    prices_file: Path = field(default_factory=lambda: Path("data/stooq_prices.csv"))
    output_dir: Path = field(default_factory=lambda: Path("snapshots"))
    
    # Clinical provider config
    clinical_provider: str = "stub"  # "stub" or "aact"
    aact_snapshots_dir: Optional[Path] = None
    trial_mapping_file: Optional[Path] = None
    aact_cache_dir: Optional[Path] = None
    aact_enable_diffs: bool = False  # Enable PCD push / status flip computation
    
    @property
    def pit_cutoff(self) -> date:
        """Compute PIT cutoff date."""
        return self.as_of_date - timedelta(days=self.pit_lag_days)


@dataclass
class Snapshot:
    """
    Complete snapshot output structure.
    
    Designed for JSON serialization with full provenance and
    deterministic reproducibility.
    """
    # Core metadata
    snapshot_id: str
    as_of_date: date
    generated_at: str  # ISO timestamp
    
    # PIT tracking
    pit_cutoff: date
    pit_lag_days: int
    
    # Provider provenance
    provenance: dict[str, Any]
    
    # Input hashes for reproducibility
    input_hashes: dict[str, str]
    
    # Coverage statistics
    coverage: dict[str, dict[str, Any]]
    
    # Per-ticker data
    tickers: dict[str, dict[str, Any]]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "snapshot_id": self.snapshot_id,
            "as_of_date": self.as_of_date.isoformat(),
            "generated_at": self.generated_at,
            "pit_cutoff": self.pit_cutoff.isoformat(),
            "pit_lag_days": self.pit_lag_days,
            "provenance": self.provenance,
            "input_hashes": self.input_hashes,
            "coverage": self.coverage,
            "tickers": self.tickers,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def load_universe(universe_file: Path) -> list[str]:
    """
    Load ticker universe from CSV.
    
    Expected columns: ticker (required), other columns optional
    
    Returns:
        Sorted list of tickers
    """
    tickers = []
    
    if not universe_file.exists():
        logger.warning(f"Universe file not found: {universe_file}")
        return tickers
    
    with open(universe_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker", "").strip().upper()
            if ticker:
                tickers.append(ticker)
    
    # Sort for determinism
    tickers = sorted(set(tickers))
    logger.info(f"Loaded universe: {len(tickers)} tickers")
    return tickers


def create_clinical_provider(config: SnapshotConfig) -> ClinicalTrialsProvider:
    """
    Factory function to create clinical trials provider.
    
    Args:
        config: Snapshot configuration
    
    Returns:
        Configured provider instance
    """
    if config.clinical_provider == "stub":
        return StubClinicalTrialsProvider()
    
    elif config.clinical_provider == "aact":
        if config.aact_snapshots_dir is None:
            raise ValueError("--aact-snapshots required when --clinical-provider=aact")
        
        return AACTClinicalTrialsProvider(
            snapshots_root=config.aact_snapshots_dir,
            strict_pit=True,
            compute_diffs=config.aact_enable_diffs,
        )
    
    else:
        raise ValueError(f"Unknown clinical provider: {config.clinical_provider}")


def generate_snapshot(config: SnapshotConfig) -> Snapshot:
    """
    Generate a complete snapshot.
    
    Args:
        config: Snapshot configuration
    
    Returns:
        Generated Snapshot object
    """
    from datetime import datetime
    
    logger.info(f"Generating snapshot for {config.as_of_date} (pit_cutoff={config.pit_cutoff})")
    
    # Load universe
    tickers = load_universe(config.universe_file)
    
    if not tickers:
        raise ValueError("No tickers in universe")
    
    # Load trial mapping if using AACT
    trial_mapping: dict[str, list[str]] = {}
    if config.clinical_provider == "aact" and config.trial_mapping_file:
        trial_mapping = load_trial_mapping(config.trial_mapping_file)
    
    # Create clinical provider
    clinical_provider = create_clinical_provider(config)
    
    # Get clinical trials data
    clinical_result = clinical_provider.get_trials_as_of(
        as_of_date=config.as_of_date,
        pit_cutoff=config.pit_cutoff,
        tickers=tickers,
        trial_mapping=trial_mapping,
    )
    
    # Compute input hashes
    trial_facts_hash = compute_trial_facts_hash(clinical_result.trials_by_ticker)
    universe_hash = compute_hash(tickers)
    
    input_hashes = {
        "universe": universe_hash,
        "trial_facts": trial_facts_hash,
    }
    
    # Build provenance
    provenance = {
        "pit_cutoff": config.pit_cutoff.isoformat(),
        "providers": {
            "clinical": {
                "name": clinical_result.provider_name,
                "snapshot_date_used": clinical_result.snapshot_date_used.isoformat(),
                "snapshots_root": clinical_result.snapshots_root,
                "compute_diffs_enabled": clinical_result.compute_diffs_enabled,
                "compute_diffs_available": clinical_result.compute_diffs_available,
                "snapshots_available_count": clinical_result.snapshots_available_count,
            }
        }
    }
    
    # Build coverage statistics
    coverage = {
        "catalyst": {
            "tickers_total": clinical_result.tickers_total,
            "tickers_with_trials": clinical_result.tickers_with_trials,
            "coverage_rate": float(clinical_result.coverage_rate),
        },
        "clinical_dev": {
            "tickers_total": clinical_result.tickers_total,
            "tickers_with_trials": clinical_result.tickers_with_trials,
            "coverage_rate": float(clinical_result.coverage_rate),
        }
    }
    
    # Build per-ticker data
    tickers_data: dict[str, dict[str, Any]] = {}
    
    for ticker in tickers:
        trials = clinical_result.trials_by_ticker.get(ticker, [])
        
        tickers_data[ticker] = {
            "trials": [t.to_dict() for t in trials],
            "trial_count": len(trials),
            # Module scores will be added here when modules are wired
            "module3_catalyst": None,
            "module4_clinical": None,
        }
    
    # Compute snapshot ID
    snapshot_id = compute_snapshot_id(
        as_of_date=config.as_of_date,
        pit_cutoff=config.pit_cutoff,
        input_hashes=input_hashes,
        provider_metadata=provenance["providers"],
    )
    
    snapshot = Snapshot(
        snapshot_id=snapshot_id,
        as_of_date=config.as_of_date,
        generated_at=datetime.utcnow().isoformat() + "Z",
        pit_cutoff=config.pit_cutoff,
        pit_lag_days=config.pit_lag_days,
        provenance=provenance,
        input_hashes=input_hashes,
        coverage=coverage,
        tickers=tickers_data,
    )
    
    return snapshot


def save_snapshot(snapshot: Snapshot, output_dir: Path) -> Path:
    """
    Save snapshot to JSON file.
    
    Args:
        snapshot: Snapshot to save
        output_dir: Output directory
    
    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{snapshot.as_of_date.isoformat()}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(snapshot.to_json())
    
    logger.info(f"Saved snapshot to {output_file}")
    return output_file


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate biotech alpha snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--as-of",
        type=lambda s: date.fromisoformat(s),
        required=True,
        help="Snapshot date (YYYY-MM-DD)",
    )
    
    # Input paths
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("data/universe/biotech_universe_v1.csv"),
        help="Path to universe CSV",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("data/stooq_prices.csv"),
        help="Path to prices CSV",
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("snapshots"),
        help="Output directory for snapshots",
    )
    
    # PIT configuration
    parser.add_argument(
        "--pit-lag-days",
        type=int,
        default=1,
        help="Days of lag for PIT safety (default: 1)",
    )
    
    # Clinical provider configuration
    parser.add_argument(
        "--clinical-provider",
        choices=["stub", "aact"],
        default="stub",
        help="Clinical trials data provider",
    )
    parser.add_argument(
        "--aact-snapshots",
        type=Path,
        help="Path to AACT snapshots directory",
    )
    parser.add_argument(
        "--trial-map",
        type=Path,
        help="Path to trial mapping CSV",
    )
    parser.add_argument(
        "--aact-cache",
        type=Path,
        help="Path to AACT cache directory (optional)",
    )
    parser.add_argument(
        "--aact-enable-diffs",
        action="store_true",
        default=False,
        help="Enable PCD push / status flip computation (requires multiple snapshots)",
    )
    
    args = parser.parse_args()
    
    # Build config
    config = SnapshotConfig(
        as_of_date=args.as_of,
        pit_lag_days=args.pit_lag_days,
        universe_file=args.universe,
        prices_file=args.prices,
        output_dir=args.output,
        clinical_provider=args.clinical_provider,
        aact_snapshots_dir=args.aact_snapshots,
        trial_mapping_file=args.trial_map,
        aact_cache_dir=args.aact_cache,
        aact_enable_diffs=args.aact_enable_diffs,
    )
    
    # Generate and save
    snapshot = generate_snapshot(config)
    output_file = save_snapshot(snapshot, config.output_dir)
    
    # Summary
    print(f"\nSnapshot generated successfully!")
    print(f"  Snapshot ID: {snapshot.snapshot_id}")
    print(f"  As-of date: {snapshot.as_of_date}")
    print(f"  PIT cutoff: {snapshot.pit_cutoff}")
    print(f"  Clinical provider: {config.clinical_provider}")
    print(f"  Coverage (catalyst): {snapshot.coverage['catalyst']['coverage_rate']:.1%}")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
