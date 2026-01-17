"""
Stub Clinical Trials Provider.

Returns empty trial data for baseline comparison.
Used when --clinical-provider=stub or as fallback.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

from .protocols import (
    ClinicalTrialsProvider,
    ProviderResult,
    TrialRow,
)


class StubClinicalTrialsProvider:
    """
    Stub provider that returns empty trial data.
    
    Used for:
        - Baseline comparison (stub vs real provider)
        - Testing modules in isolation
        - Fallback when AACT data unavailable
    """
    
    def __init__(self) -> None:
        pass
    
    @property
    def provider_name(self) -> str:
        return "stub"
    
    def get_trials_as_of(
        self,
        as_of_date: date,
        pit_cutoff: date,
        tickers: list[str],
        trial_mapping: dict[str, list[str]],
    ) -> ProviderResult:
        """
        Return empty trials for all tickers.
        
        Args:
            as_of_date: The date we're generating the snapshot for
            pit_cutoff: Latest allowed data date (ignored in stub)
            tickers: List of tickers
            trial_mapping: Mapping from ticker to NCT IDs (ignored in stub)
        
        Returns:
            ProviderResult with empty trials_by_ticker
        """
        return ProviderResult(
            trials_by_ticker={ticker: [] for ticker in tickers},
            snapshot_date_used=as_of_date,
            snapshots_root="stub",
            provider_name=self.provider_name,
            pit_cutoff_applied=pit_cutoff,
            tickers_total=len(tickers),
            tickers_with_trials=0,
            trials_total=0,
        )
