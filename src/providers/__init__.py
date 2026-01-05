"""
Data providers for Wake Robin Biotech Alpha System.

Provider architecture follows the factory pattern with PIT boundary enforcement.
All providers return data that is already PIT-safe, so downstream modules
can remain pure and deterministic.
"""

from .protocols import ClinicalTrialsProvider, TrialRow, TrialDiff, ProviderResult
from .aact_provider import AACTClinicalTrialsProvider
from .stub_provider import StubClinicalTrialsProvider

__all__ = [
    "ClinicalTrialsProvider",
    "TrialRow",
    "TrialDiff",
    "ProviderResult",
    "AACTClinicalTrialsProvider",
    "StubClinicalTrialsProvider",
]
