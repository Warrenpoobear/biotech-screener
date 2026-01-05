"""
AACT Clinical Trials Provider.

Reads from local CSV snapshot folders with strict PIT boundary enforcement.
Each snapshot folder contains studies.csv and sponsors.csv extracted from AACT.

Folder layout:
    data/aact_snapshots/2024-01-15/studies.csv
    data/aact_snapshots/2024-01-15/sponsors.csv
    data/aact_snapshots/2024-01-29/studies.csv
    data/aact_snapshots/2024-01-29/sponsors.csv
"""

import csv
import logging
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from .protocols import (
    ClinicalTrialsProvider,
    PCDType,
    Phase,
    ProviderResult,
    TrialDiff,
    TrialRow,
    TrialStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class AACTSnapshot:
    """Represents a loaded AACT snapshot."""
    snapshot_date: date
    studies: dict[str, dict]  # nct_id -> study row
    sponsors: dict[str, str]  # nct_id -> lead_sponsor name


class AACTClinicalTrialsProvider:
    """
    AACT provider that reads from local CSV snapshot folders.
    
    PIT Safety:
        - Only loads snapshots with date < pit_cutoff (strict inequality)
        - Selects the latest available snapshot that satisfies PIT constraint
        - Never uses same-day or future data
    
    Usage:
        provider = AACTClinicalTrialsProvider(Path("data/aact_snapshots"))
        result = provider.get_trials_as_of(
            as_of_date=date(2024, 1, 31),
            pit_cutoff=date(2024, 1, 30),
            tickers=["MRNA", "BNTX"],
            trial_mapping={"MRNA": ["NCT04470427", "NCT04860297"], ...}
        )
    """
    
    # Required columns for schema validation
    REQUIRED_STUDIES_COLUMNS = {
        "nct_id", "phase", "overall_status", 
        "primary_completion_date", "primary_completion_date_type",
        "last_update_posted_date", "study_type"
    }
    REQUIRED_SPONSORS_COLUMNS = {"nct_id", "name", "lead_or_collaborator"}
    
    def __init__(
        self,
        snapshots_root: Path,
        strict_pit: bool = True,  # If True, use < pit_cutoff; if False, use <= pit_cutoff
        compute_diffs: bool = False,  # If True, compute pcd_pushes/status_flips from snapshots
    ):
        """
        Initialize AACT provider.
        
        Args:
            snapshots_root: Path to directory containing dated snapshot folders
            strict_pit: If True, snapshot must be < pit_cutoff; if False, <= pit_cutoff
            compute_diffs: If True, compute PCD pushes and status flips from snapshot history
        """
        self.snapshots_root = Path(snapshots_root)
        self.strict_pit = strict_pit
        self.compute_diffs = compute_diffs
        self._snapshot_cache: dict[date, AACTSnapshot] = {}
        
        if not self.snapshots_root.exists():
            logger.warning(f"AACT snapshots root does not exist: {self.snapshots_root}")
    
    @property
    def provider_name(self) -> str:
        return "aact"
    
    def get_available_snapshots(self) -> list[date]:
        """
        Get list of available snapshot dates, sorted ascending.
        
        Returns:
            List of dates for which snapshots exist
        """
        if not self.snapshots_root.exists():
            return []
        
        snapshots = []
        for folder in self.snapshots_root.iterdir():
            if folder.is_dir():
                try:
                    snapshot_date = date.fromisoformat(folder.name)
                    # Verify required files exist
                    if (folder / "studies.csv").exists():
                        snapshots.append(snapshot_date)
                except ValueError:
                    logger.debug(f"Skipping non-date folder: {folder.name}")
        
        return sorted(snapshots)
    
    def select_snapshot(self, pit_cutoff: date) -> Optional[date]:
        """
        Select the latest snapshot that satisfies PIT constraint.
        
        Args:
            pit_cutoff: Latest allowed snapshot date
        
        Returns:
            Selected snapshot date, or None if no valid snapshot exists
        """
        available = self.get_available_snapshots()
        
        if not available:
            logger.warning("No AACT snapshots available")
            return None
        
        # Filter by PIT constraint
        if self.strict_pit:
            valid = [d for d in available if d < pit_cutoff]
        else:
            valid = [d for d in available if d <= pit_cutoff]
        
        if not valid:
            logger.warning(
                f"No AACT snapshots satisfy PIT constraint (pit_cutoff={pit_cutoff}, "
                f"strict={self.strict_pit}, available={available})"
            )
            return None
        
        selected = max(valid)
        logger.info(f"Selected AACT snapshot: {selected} (pit_cutoff={pit_cutoff})")
        return selected
    
    def load_snapshot(self, snapshot_date: date) -> AACTSnapshot:
        """
        Load a snapshot from disk (with caching).
        
        Args:
            snapshot_date: Date of snapshot to load
        
        Returns:
            Loaded AACTSnapshot
        
        Raises:
            ValueError: If required columns are missing from CSV files
        """
        if snapshot_date in self._snapshot_cache:
            return self._snapshot_cache[snapshot_date]
        
        snapshot_path = self.snapshots_root / snapshot_date.isoformat()
        
        # Load studies with schema validation
        studies_file = snapshot_path / "studies.csv"
        studies: dict[str, dict] = {}
        duplicates_found = 0
        
        if studies_file.exists():
            with open(studies_file, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                
                # Schema validation
                if reader.fieldnames:
                    missing_cols = self.REQUIRED_STUDIES_COLUMNS - set(reader.fieldnames)
                    if missing_cols:
                        raise ValueError(
                            f"Missing required columns in {studies_file}: {missing_cols}"
                        )
                
                for row in reader:
                    nct_id = row.get("nct_id", "").strip().upper()
                    if nct_id:
                        # Duplicate handling: keep row with latest last_update_posted_date
                        if nct_id in studies:
                            duplicates_found += 1
                            existing_date = studies[nct_id].get("last_update_posted_date", "")
                            new_date = row.get("last_update_posted_date", "")
                            if new_date > existing_date:
                                studies[nct_id] = row
                                logger.debug(
                                    f"Duplicate NCT ID {nct_id}: keeping row with "
                                    f"last_update_posted_date={new_date} over {existing_date}"
                                )
                        else:
                            studies[nct_id] = row
        
        if duplicates_found > 0:
            logger.warning(
                f"Found {duplicates_found} duplicate NCT IDs in {studies_file}; "
                f"kept rows with latest last_update_posted_date"
            )
        
        logger.info(f"Loaded {len(studies)} studies from {studies_file}")
        
        # Load sponsors with schema validation
        sponsors_file = snapshot_path / "sponsors.csv"
        sponsors: dict[str, str] = {}
        
        if sponsors_file.exists():
            with open(sponsors_file, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                
                # Schema validation
                if reader.fieldnames:
                    missing_cols = self.REQUIRED_SPONSORS_COLUMNS - set(reader.fieldnames)
                    if missing_cols:
                        raise ValueError(
                            f"Missing required columns in {sponsors_file}: {missing_cols}"
                        )
                
                for row in reader:
                    nct_id = row.get("nct_id", "").strip().upper()
                    lead_or_collab = row.get("lead_or_collaborator", "").strip().upper()
                    name = row.get("name", "").strip()
                    
                    # Only capture lead sponsors
                    if nct_id and lead_or_collab == "LEAD":
                        sponsors[nct_id] = name
        
        logger.info(f"Loaded {len(sponsors)} lead sponsors from {sponsors_file}")
        
        snapshot = AACTSnapshot(
            snapshot_date=snapshot_date,
            studies=studies,
            sponsors=sponsors,
        )
        
        self._snapshot_cache[snapshot_date] = snapshot
        return snapshot
    
    def build_trial_row(self, nct_id: str, study: dict, sponsor: str) -> TrialRow:
        """
        Build a canonical TrialRow from raw AACT data.
        
        Args:
            nct_id: Trial identifier
            study: Raw study dict from studies.csv
            sponsor: Lead sponsor name
        
        Returns:
            Normalized TrialRow with appropriate flags for missing data
        """
        flags: list[str] = []
        
        # Parse primary completion date
        pcd_str = study.get("primary_completion_date", "").strip()
        pcd: Optional[date] = None
        if pcd_str:
            try:
                pcd = date.fromisoformat(pcd_str)
            except ValueError:
                # Try parsing other formats if needed
                logger.debug(f"Could not parse PCD '{pcd_str}' for {nct_id}")
                flags.append("pcd_parse_error")
        else:
            flags.append("pcd_missing")
        
        # Parse PCD type
        pcd_type_str = study.get("primary_completion_date_type", "").strip()
        pcd_type = PCDType.from_aact(pcd_type_str)
        if pcd_type == PCDType.UNKNOWN:
            flags.append("pcd_type_missing")
        
        # Parse last update date
        last_update_str = study.get("last_update_posted_date", "").strip()
        last_update: Optional[date] = None
        if last_update_str:
            try:
                last_update = date.fromisoformat(last_update_str)
            except ValueError:
                logger.debug(f"Could not parse last_update '{last_update_str}' for {nct_id}")
                flags.append("last_update_parse_error")
        else:
            flags.append("last_update_missing")
        
        # Sort flags for deterministic hashing
        flags_tuple = tuple(sorted(flags))
        
        return TrialRow(
            nct_id=nct_id,
            phase=Phase.from_aact(study.get("phase")),
            overall_status=TrialStatus.from_aact(study.get("overall_status")),
            primary_completion_date=pcd,
            primary_completion_date_type=pcd_type,
            last_update_posted_date=last_update,
            lead_sponsor=sponsor,
            study_type=study.get("study_type", "Interventional"),
            flags=flags_tuple,
        )
    
    def _compute_diffs(
        self,
        nct_id: str,
        snapshots: list[AACTSnapshot],
        lookback_days: int = 548,  # ~18 months
        as_of_date: date = None,
    ) -> tuple[int, int]:
        """
        Compute PCD pushes and status flips for a trial across snapshots.
        
        A PCD push is counted when:
            - PCD moves to a later date (regardless of type change)
            - NULL -> date is NOT a push (initial setting)
            - date -> NULL is NOT a push (removal)
        
        A status flip is counted when:
            - Status changes between major categories
            - Active <-> Suspended counts
            - We ignore minor variations within "active" states
        
        Args:
            nct_id: Trial identifier
            snapshots: List of snapshots sorted by date (ascending)
            lookback_days: How far back to look for changes
            as_of_date: Reference date for lookback calculation
        
        Returns:
            Tuple of (pcd_pushes, status_flips)
        """
        if as_of_date is None:
            as_of_date = date.today()
        
        cutoff_date = as_of_date - timedelta(days=lookback_days)
        
        # Filter snapshots to lookback window
        relevant = [s for s in snapshots if s.snapshot_date >= cutoff_date]
        
        if len(relevant) < 2:
            return (0, 0)
        
        pcd_pushes = 0
        status_flips = 0
        
        for i in range(1, len(relevant)):
            prev_snap = relevant[i - 1]
            curr_snap = relevant[i]
            
            prev_study = prev_snap.studies.get(nct_id)
            curr_study = curr_snap.studies.get(nct_id)
            
            if not prev_study or not curr_study:
                continue
            
            # Check PCD push
            prev_pcd_str = prev_study.get("primary_completion_date", "").strip()
            curr_pcd_str = curr_study.get("primary_completion_date", "").strip()
            
            if prev_pcd_str and curr_pcd_str:
                try:
                    prev_pcd = date.fromisoformat(prev_pcd_str)
                    curr_pcd = date.fromisoformat(curr_pcd_str)
                    if curr_pcd > prev_pcd:
                        pcd_pushes += 1
                except ValueError:
                    pass
            
            # Check status flip
            prev_status = TrialStatus.from_aact(prev_study.get("overall_status"))
            curr_status = TrialStatus.from_aact(curr_study.get("overall_status"))
            
            # Count meaningful status changes
            if prev_status != curr_status:
                # Count flips between active and non-active states
                if prev_status.is_active() != curr_status.is_active():
                    status_flips += 1
                # Or between terminal and non-terminal
                elif prev_status.is_terminal() != curr_status.is_terminal():
                    status_flips += 1
        
        return (pcd_pushes, status_flips)
    
    def get_trials_as_of(
        self,
        as_of_date: date,
        pit_cutoff: date,
        tickers: list[str],
        trial_mapping: dict[str, list[str]],
    ) -> ProviderResult:
        """
        Return trials per ticker, filtered to PIT-safe snapshot.
        
        Args:
            as_of_date: The date we're generating the snapshot for
            pit_cutoff: Latest allowed data date
            tickers: List of tickers to get trials for
            trial_mapping: Mapping from ticker to NCT IDs
        
        Returns:
            ProviderResult with trials_by_ticker and metadata
        """
        # Select appropriate snapshot
        snapshot_date = self.select_snapshot(pit_cutoff)
        
        if snapshot_date is None:
            # Return empty result with metadata
            return ProviderResult(
                trials_by_ticker={ticker: [] for ticker in tickers},
                snapshot_date_used=date.min,
                snapshots_root=str(self.snapshots_root),
                provider_name=self.provider_name,
                pit_cutoff_applied=pit_cutoff,
                tickers_total=len(tickers),
                tickers_with_trials=0,
                trials_total=0,
                compute_diffs_enabled=self.compute_diffs,
                compute_diffs_available=False,
                snapshots_available_count=0,
            )
        
        snapshot = self.load_snapshot(snapshot_date)
        
        # Determine diff computation availability
        all_snapshots: list[AACTSnapshot] = []
        diffs_available = False
        
        if self.compute_diffs:
            # Load all available snapshots for diff computation
            for snap_date in self.get_available_snapshots():
                if snap_date <= snapshot_date:
                    all_snapshots.append(self.load_snapshot(snap_date))
            all_snapshots.sort(key=lambda s: s.snapshot_date)
            diffs_available = len(all_snapshots) >= 2
        
        # Determine diff flag to add to trials
        if not self.compute_diffs:
            diff_flag = "diffs_disabled"
        elif not diffs_available:
            diff_flag = "diffs_unavailable_insufficient_snapshots"
        else:
            diff_flag = None  # Diffs are computed, no flag needed
        
        # Build trials by ticker
        trials_by_ticker: dict[str, list[TrialRow]] = {}
        
        for ticker in tickers:
            nct_ids = trial_mapping.get(ticker, [])
            trials: list[TrialRow] = []
            
            for nct_id in nct_ids:
                nct_id_upper = nct_id.strip().upper()
                study = snapshot.studies.get(nct_id_upper)
                
                if study is None:
                    logger.debug(f"NCT ID {nct_id_upper} not found in snapshot for {ticker}")
                    continue
                
                sponsor = snapshot.sponsors.get(nct_id_upper, "Unknown")
                trial = self.build_trial_row(nct_id_upper, study, sponsor)
                
                # Compute diffs if enabled and available
                if self.compute_diffs and diffs_available:
                    pcd_pushes, status_flips = self._compute_diffs(
                        nct_id_upper,
                        all_snapshots,
                        lookback_days=548,
                        as_of_date=as_of_date,
                    )
                    trial = replace(trial, pcd_pushes_18m=pcd_pushes, status_flips_18m=status_flips)
                elif diff_flag:
                    # Add diff unavailability flag
                    new_flags = tuple(sorted(set(trial.flags) | {diff_flag}))
                    trial = replace(trial, flags=new_flags)
                
                trials.append(trial)
            
            # Sort deterministically by nct_id
            trials.sort(key=lambda t: t.nct_id)
            trials_by_ticker[ticker] = trials
        
        result = ProviderResult(
            trials_by_ticker=trials_by_ticker,
            snapshot_date_used=snapshot_date,
            snapshots_root=str(self.snapshots_root),
            provider_name=self.provider_name,
            pit_cutoff_applied=pit_cutoff,
            compute_diffs_enabled=self.compute_diffs,
            compute_diffs_available=diffs_available,
            snapshots_available_count=len(all_snapshots),
        )
        result.compute_coverage()
        
        return result


def load_trial_mapping(mapping_file: Path) -> dict[str, list[str]]:
    """
    Load trial mapping from CSV file.
    
    Expected columns: ticker, nct_id, effective_start, effective_end, source
    
    Args:
        mapping_file: Path to trial_mapping.csv
    
    Returns:
        Dict mapping ticker to list of NCT IDs
    """
    mapping: dict[str, list[str]] = {}
    
    if not mapping_file.exists():
        logger.warning(f"Trial mapping file not found: {mapping_file}")
        return mapping
    
    with open(mapping_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker", "").strip().upper()
            nct_id = row.get("nct_id", "").strip().upper()
            
            if ticker and nct_id:
                if ticker not in mapping:
                    mapping[ticker] = []
                if nct_id not in mapping[ticker]:
                    mapping[ticker].append(nct_id)
    
    logger.info(f"Loaded trial mapping: {len(mapping)} tickers, "
                f"{sum(len(v) for v in mapping.values())} total mappings")
    
    return mapping
