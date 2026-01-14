"""
Dossier Generator Package

Generates comprehensive institutional-grade investment dossiers for biotech
candidates from the Wake Robin screening system.
"""

from .generator import DossierGenerator
from .data_fetchers import DossierDataFetcher
from .section_generators import DossierSectionGenerator

__version__ = "1.0.0"
__all__ = ["DossierGenerator", "DossierDataFetcher", "DossierSectionGenerator"]
