#!/usr/bin/env python3
"""
Partnership Validation Engine

Analyzes partnership and licensing deals to assess validation signals.
Big pharma partnerships validate technology and provide non-dilutive funding.

Key signals:
- Partner quality (top 10 pharma vs smaller companies)
- Deal structure (upfront, milestones, royalties)
- Partnership type (licensing, collaboration, co-development)
- Recency and activity level
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import re


class PartnerTier(Enum):
    """Classification of partner quality."""
    TOP_10_PHARMA = "top_10_pharma"      # JNJ, Pfizer, Roche, etc.
    MAJOR_PHARMA = "major_pharma"         # Large but not top 10
    MID_TIER = "mid_tier"                 # Mid-size pharma/biotech
    SMALL_BIOTECH = "small_biotech"       # Smaller companies
    ACADEMIC = "academic"                  # Universities, research institutions
    UNKNOWN = "unknown"


class DealType(Enum):
    """Types of partnership deals."""
    EXCLUSIVE_LICENSE = "exclusive_license"
    NON_EXCLUSIVE_LICENSE = "non_exclusive_license"
    CO_DEVELOPMENT = "co_development"
    COLLABORATION = "collaboration"
    OPTION_AGREEMENT = "option_agreement"
    ACQUISITION_OPTION = "acquisition_option"
    SUPPLY_AGREEMENT = "supply_agreement"
    RESEARCH_COLLABORATION = "research_collaboration"
    UNKNOWN = "unknown"


class PartnershipStrength(Enum):
    """Overall partnership profile strength."""
    EXCEPTIONAL = "exceptional"    # Multiple top-tier deals
    STRONG = "strong"              # At least one major deal
    MODERATE = "moderate"          # Some partnerships
    WEAK = "weak"                  # Limited or no partnerships
    UNKNOWN = "unknown"


@dataclass
class Partnership:
    """Represents a single partnership deal."""
    ticker: str
    partner_name: str
    partner_tier: PartnerTier
    deal_type: DealType
    announcement_date: date
    upfront_payment: Optional[Decimal] = None      # $MM
    total_potential_value: Optional[Decimal] = None  # $MM including milestones
    indication: Optional[str] = None
    asset_name: Optional[str] = None
    is_active: bool = True
    notes: Optional[str] = None


@dataclass
class PartnershipProfile:
    """Aggregated partnership profile for a ticker."""
    ticker: str
    partnerships: List[Partnership] = field(default_factory=list)
    total_upfront: Decimal = Decimal("0")
    total_potential_value: Decimal = Decimal("0")
    top_tier_count: int = 0
    active_partnerships: int = 0
    most_recent_deal: Optional[date] = None
    strength: PartnershipStrength = PartnershipStrength.UNKNOWN


class PartnershipEngine:
    """
    Engine for analyzing partnership and licensing deals.

    Scoring philosophy:
    - Top pharma partnerships provide strong validation
    - Deal size indicates partner confidence
    - Recent activity shows ongoing interest
    - Multiple partnerships reduce dependency risk
    """

    VERSION = "1.0.0"

    # Top 10 pharma companies by market cap/revenue
    TOP_10_PHARMA = {
        "johnson & johnson", "jnj", "j&j",
        "pfizer", "pfe",
        "roche", "rhhby",
        "novartis", "nvs",
        "merck", "mrk",
        "abbvie", "abbv",
        "eli lilly", "lilly", "lly",
        "astrazeneca", "azn",
        "bristol-myers squibb", "bmy", "bristol myers",
        "sanofi", "sny",
        "glaxosmithkline", "gsk",
        "gilead", "gild",
        "amgen", "amgn",
        "regeneron", "regn",
        "vertex", "vrtx",
    }

    # Major pharma (not top 10 but significant)
    MAJOR_PHARMA = {
        "biogen", "biib",
        "moderna", "mrna",
        "biontech", "bntx",
        "takeda", "tak",
        "novo nordisk", "nvo",
        "astellas",
        "boehringer ingelheim",
        "bayer", "bayry",
        "teva", "teva",
        "allergan", "agn",
        "celgene",  # Now part of BMS
        "shire",    # Now part of Takeda
        "alexion", "alxn",
        "incyte", "incy",
        "jazz pharmaceuticals", "jazz",
        "biomarin", "bmrn",
        "seagen", "sgen",
        "alnylam", "alny",
    }

    # Academic/research institutions
    ACADEMIC_PARTNERS = {
        "university", "college", "institute", "hospital",
        "nih", "national institutes", "medical center",
        "research foundation", "cancer center",
    }

    # Scoring weights
    TIER_SCORES = {
        PartnerTier.TOP_10_PHARMA: Decimal("25"),
        PartnerTier.MAJOR_PHARMA: Decimal("18"),
        PartnerTier.MID_TIER: Decimal("10"),
        PartnerTier.SMALL_BIOTECH: Decimal("5"),
        PartnerTier.ACADEMIC: Decimal("8"),
        PartnerTier.UNKNOWN: Decimal("3"),
    }

    DEAL_TYPE_MULTIPLIERS = {
        DealType.EXCLUSIVE_LICENSE: Decimal("1.3"),
        DealType.CO_DEVELOPMENT: Decimal("1.2"),
        DealType.ACQUISITION_OPTION: Decimal("1.25"),
        DealType.OPTION_AGREEMENT: Decimal("1.1"),
        DealType.COLLABORATION: Decimal("1.0"),
        DealType.NON_EXCLUSIVE_LICENSE: Decimal("0.9"),
        DealType.RESEARCH_COLLABORATION: Decimal("0.8"),
        DealType.SUPPLY_AGREEMENT: Decimal("0.7"),
        DealType.UNKNOWN: Decimal("0.8"),
    }

    # Deal value thresholds ($MM)
    DEAL_SIZE_THRESHOLDS = {
        "mega": Decimal("1000"),      # $1B+ total value
        "large": Decimal("500"),       # $500M+
        "significant": Decimal("100"), # $100M+
        "modest": Decimal("25"),       # $25M+
    }

    def __init__(self):
        """Initialize the partnership engine."""
        self.partnerships_by_ticker: Dict[str, List[Partnership]] = {}
        self.profiles: Dict[str, PartnershipProfile] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        self._data_loaded = False

    def classify_partner(self, partner_name: str) -> PartnerTier:
        """
        Classify a partner into a tier based on name matching.

        Args:
            partner_name: Name of the partner company

        Returns:
            PartnerTier classification
        """
        if not partner_name:
            return PartnerTier.UNKNOWN

        name_lower = partner_name.lower().strip()

        # Check top 10 pharma
        for pharma in self.TOP_10_PHARMA:
            if pharma in name_lower or name_lower in pharma:
                return PartnerTier.TOP_10_PHARMA

        # Check major pharma
        for pharma in self.MAJOR_PHARMA:
            if pharma in name_lower or name_lower in pharma:
                return PartnerTier.MAJOR_PHARMA

        # Check academic
        for academic in self.ACADEMIC_PARTNERS:
            if academic in name_lower:
                return PartnerTier.ACADEMIC

        # Heuristics for mid-tier vs small
        # Mid-tier often have "pharma", "therapeutics" in name
        if any(term in name_lower for term in ["pharmaceuticals", "pharma", "therapeutics"]):
            return PartnerTier.MID_TIER

        return PartnerTier.SMALL_BIOTECH

    def classify_deal_type(self, deal_description: str) -> DealType:
        """
        Classify deal type from description.

        Args:
            deal_description: Text describing the deal

        Returns:
            DealType classification
        """
        if not deal_description:
            return DealType.UNKNOWN

        desc_lower = deal_description.lower()

        # Check non-exclusive BEFORE exclusive (since "non-exclusive" contains "exclusive")
        if "non-exclusive" in desc_lower or "nonexclusive" in desc_lower:
            return DealType.NON_EXCLUSIVE_LICENSE
        elif "exclusive" in desc_lower and "license" in desc_lower:
            return DealType.EXCLUSIVE_LICENSE
        elif "co-develop" in desc_lower or "codevelop" in desc_lower or "joint develop" in desc_lower:
            return DealType.CO_DEVELOPMENT
        elif "acquisition" in desc_lower and "option" in desc_lower:
            return DealType.ACQUISITION_OPTION
        elif "option" in desc_lower:
            return DealType.OPTION_AGREEMENT
        elif "license" in desc_lower:
            return DealType.EXCLUSIVE_LICENSE  # Default license to exclusive
        # Check research collaboration BEFORE general collaboration
        elif "research" in desc_lower:
            return DealType.RESEARCH_COLLABORATION
        elif "collaborat" in desc_lower:
            return DealType.COLLABORATION
        elif "supply" in desc_lower or "manufacturing" in desc_lower:
            return DealType.SUPPLY_AGREEMENT

        return DealType.UNKNOWN

    def load_partnerships(self, partnership_records: List[Dict[str, Any]]) -> int:
        """
        Load partnership data from records.

        Expected record format:
        {
            "ticker": "ACME",
            "partner_name": "Pfizer",
            "deal_type": "exclusive_license",  # or description
            "announcement_date": "2024-06-15",
            "upfront_payment": 150,  # $MM
            "total_potential_value": 1200,  # $MM
            "indication": "oncology",
            "asset_name": "ACM-123",
            "is_active": true
        }

        Args:
            partnership_records: List of partnership dictionaries

        Returns:
            Number of partnerships loaded
        """
        loaded = 0

        for record in partnership_records:
            try:
                ticker = record.get("ticker", "").upper()
                if not ticker:
                    continue

                # Parse partner tier
                partner_name = record.get("partner_name", "")
                partner_tier_str = record.get("partner_tier")
                if partner_tier_str:
                    try:
                        partner_tier = PartnerTier(partner_tier_str)
                    except ValueError:
                        partner_tier = self.classify_partner(partner_name)
                else:
                    partner_tier = self.classify_partner(partner_name)

                # Parse deal type
                deal_type_str = record.get("deal_type", "")
                try:
                    deal_type = DealType(deal_type_str)
                except ValueError:
                    deal_type = self.classify_deal_type(deal_type_str)

                # Parse date
                date_str = record.get("announcement_date", "")
                if isinstance(date_str, str):
                    try:
                        announcement_date = date.fromisoformat(date_str)
                    except ValueError:
                        announcement_date = date(2020, 1, 1)  # Default old date
                elif isinstance(date_str, date):
                    announcement_date = date_str
                else:
                    announcement_date = date(2020, 1, 1)

                # Parse financial values
                upfront = record.get("upfront_payment")
                upfront_decimal = Decimal(str(upfront)) if upfront is not None else None

                total_value = record.get("total_potential_value")
                total_decimal = Decimal(str(total_value)) if total_value is not None else None

                partnership = Partnership(
                    ticker=ticker,
                    partner_name=partner_name,
                    partner_tier=partner_tier,
                    deal_type=deal_type,
                    announcement_date=announcement_date,
                    upfront_payment=upfront_decimal,
                    total_potential_value=total_decimal,
                    indication=record.get("indication"),
                    asset_name=record.get("asset_name"),
                    is_active=record.get("is_active", True),
                    notes=record.get("notes"),
                )

                if ticker not in self.partnerships_by_ticker:
                    self.partnerships_by_ticker[ticker] = []
                self.partnerships_by_ticker[ticker].append(partnership)
                loaded += 1

            except Exception as e:
                # Skip malformed records
                continue

        self._data_loaded = loaded > 0
        return loaded

    def build_profile(self, ticker: str, as_of_date: date) -> PartnershipProfile:
        """
        Build a partnership profile for a ticker.

        Args:
            ticker: Stock ticker
            as_of_date: Point-in-time date for analysis

        Returns:
            PartnershipProfile with aggregated metrics
        """
        ticker = ticker.upper()
        partnerships = self.partnerships_by_ticker.get(ticker, [])

        # Filter to partnerships announced before as_of_date
        valid_partnerships = [
            p for p in partnerships
            if p.announcement_date <= as_of_date
        ]

        profile = PartnershipProfile(ticker=ticker)
        profile.partnerships = valid_partnerships

        if not valid_partnerships:
            profile.strength = PartnershipStrength.WEAK
            self.profiles[ticker] = profile
            return profile

        # Calculate aggregates
        total_upfront = Decimal("0")
        total_potential = Decimal("0")
        top_tier_count = 0
        active_count = 0
        most_recent = None

        for p in valid_partnerships:
            if p.upfront_payment:
                total_upfront += p.upfront_payment
            if p.total_potential_value:
                total_potential += p.total_potential_value
            if p.partner_tier in (PartnerTier.TOP_10_PHARMA, PartnerTier.MAJOR_PHARMA):
                top_tier_count += 1
            if p.is_active:
                active_count += 1
            if most_recent is None or p.announcement_date > most_recent:
                most_recent = p.announcement_date

        profile.total_upfront = total_upfront
        profile.total_potential_value = total_potential
        profile.top_tier_count = top_tier_count
        profile.active_partnerships = active_count
        profile.most_recent_deal = most_recent

        # Determine strength
        profile.strength = self._determine_strength(profile, as_of_date)

        self.profiles[ticker] = profile
        return profile

    def _determine_strength(
        self, profile: PartnershipProfile, as_of_date: date
    ) -> PartnershipStrength:
        """
        Determine overall partnership strength.

        Criteria:
        - EXCEPTIONAL: 2+ top-tier deals OR $1B+ total value
        - STRONG: 1 top-tier deal OR $500M+ total value
        - MODERATE: Any partnership with upfront payment
        - WEAK: No meaningful partnerships
        """
        if profile.top_tier_count >= 2:
            return PartnershipStrength.EXCEPTIONAL

        if profile.total_potential_value >= self.DEAL_SIZE_THRESHOLDS["mega"]:
            return PartnershipStrength.EXCEPTIONAL

        if profile.top_tier_count >= 1:
            return PartnershipStrength.STRONG

        if profile.total_potential_value >= self.DEAL_SIZE_THRESHOLDS["large"]:
            return PartnershipStrength.STRONG

        if profile.total_upfront >= self.DEAL_SIZE_THRESHOLDS["significant"]:
            return PartnershipStrength.MODERATE

        if len(profile.partnerships) > 0 and profile.active_partnerships > 0:
            return PartnershipStrength.MODERATE

        return PartnershipStrength.WEAK

    def score_ticker(self, ticker: str, as_of_date: date) -> Dict[str, Any]:
        """
        Score a ticker's partnership profile.

        Scoring components:
        1. Partner tier score (weighted by tier quality)
        2. Deal value score (based on upfront + potential)
        3. Recency score (recent deals score higher)
        4. Diversity score (multiple partnerships)

        Args:
            ticker: Stock ticker
            as_of_date: Point-in-time date

        Returns:
            Dictionary with score and metrics
        """
        ticker = ticker.upper()

        # Build or retrieve profile
        if ticker not in self.profiles:
            self.build_profile(ticker, as_of_date)

        profile = self.profiles.get(ticker)

        if not profile or not profile.partnerships:
            result = {
                "ticker": ticker,
                "partnership_score": Decimal("50"),  # Neutral
                "partnership_strength": PartnershipStrength.UNKNOWN.value,
                "partnership_count": 0,
                "top_tier_partners": 0,
                "total_deal_value": Decimal("0"),
                "total_upfront": Decimal("0"),
                "most_recent_deal": None,
                "active_partnerships": 0,
                "top_partners": [],
                "score_components": {
                    "tier_score": Decimal("0"),
                    "value_score": Decimal("0"),
                    "recency_score": Decimal("0"),
                    "diversity_score": Decimal("0"),
                },
            }
            self._add_audit(ticker, as_of_date, result)
            return result

        # Calculate component scores
        tier_score = self._calculate_tier_score(profile)
        value_score = self._calculate_value_score(profile)
        recency_score = self._calculate_recency_score(profile, as_of_date)
        diversity_score = self._calculate_diversity_score(profile)

        # Weighted combination (0-100 scale)
        # Tier: 35%, Value: 30%, Recency: 20%, Diversity: 15%
        raw_score = (
            tier_score * Decimal("0.35") +
            value_score * Decimal("0.30") +
            recency_score * Decimal("0.20") +
            diversity_score * Decimal("0.15")
        )

        # Clamp to 0-100
        partnership_score = max(Decimal("0"), min(Decimal("100"), raw_score))

        # Get top partners for display
        top_partners = self._get_top_partners(profile)

        result = {
            "ticker": ticker,
            "partnership_score": partnership_score.quantize(Decimal("0.01")),
            "partnership_strength": profile.strength.value,
            "partnership_count": len(profile.partnerships),
            "top_tier_partners": profile.top_tier_count,
            "total_deal_value": profile.total_potential_value,
            "total_upfront": profile.total_upfront,
            "most_recent_deal": profile.most_recent_deal.isoformat() if profile.most_recent_deal else None,
            "active_partnerships": profile.active_partnerships,
            "top_partners": top_partners,
            "score_components": {
                "tier_score": tier_score.quantize(Decimal("0.01")),
                "value_score": value_score.quantize(Decimal("0.01")),
                "recency_score": recency_score.quantize(Decimal("0.01")),
                "diversity_score": diversity_score.quantize(Decimal("0.01")),
            },
        }

        self._add_audit(ticker, as_of_date, result)
        return result

    def _calculate_tier_score(self, profile: PartnershipProfile) -> Decimal:
        """
        Calculate score based on partner quality.

        Max score: 100 (multiple top-tier partners)
        """
        if not profile.partnerships:
            return Decimal("0")

        total_tier_points = Decimal("0")

        for p in profile.partnerships:
            base_score = self.TIER_SCORES.get(p.partner_tier, Decimal("3"))
            multiplier = self.DEAL_TYPE_MULTIPLIERS.get(p.deal_type, Decimal("1.0"))
            total_tier_points += base_score * multiplier

        # Normalize to 0-100 scale
        # Cap at ~4 top-tier exclusive deals (4 * 25 * 1.3 = 130)
        normalized = (total_tier_points / Decimal("130")) * Decimal("100")
        return min(Decimal("100"), normalized)

    def _calculate_value_score(self, profile: PartnershipProfile) -> Decimal:
        """
        Calculate score based on deal values.

        Scoring:
        - $1B+ total: 100
        - $500M+: 80
        - $100M+: 60
        - $25M+: 40
        - Any upfront: 25
        - No value: 0
        """
        total = profile.total_potential_value
        upfront = profile.total_upfront

        if total >= self.DEAL_SIZE_THRESHOLDS["mega"]:
            return Decimal("100")
        elif total >= self.DEAL_SIZE_THRESHOLDS["large"]:
            return Decimal("80")
        elif total >= self.DEAL_SIZE_THRESHOLDS["significant"]:
            return Decimal("60")
        elif total >= self.DEAL_SIZE_THRESHOLDS["modest"]:
            return Decimal("40")
        elif upfront > 0:
            return Decimal("25")
        elif len(profile.partnerships) > 0:
            return Decimal("15")  # Has partnerships but no disclosed value

        return Decimal("0")

    def _calculate_recency_score(
        self, profile: PartnershipProfile, as_of_date: date
    ) -> Decimal:
        """
        Calculate score based on recency of deals.

        Scoring:
        - Deal within 6 months: 100
        - Deal within 1 year: 80
        - Deal within 2 years: 60
        - Deal within 3 years: 40
        - Older: 20
        """
        if not profile.most_recent_deal:
            return Decimal("0")

        days_since = (as_of_date - profile.most_recent_deal).days

        if days_since <= 180:  # 6 months
            return Decimal("100")
        elif days_since <= 365:  # 1 year
            return Decimal("80")
        elif days_since <= 730:  # 2 years
            return Decimal("60")
        elif days_since <= 1095:  # 3 years
            return Decimal("40")
        else:
            return Decimal("20")

    def _calculate_diversity_score(self, profile: PartnershipProfile) -> Decimal:
        """
        Calculate score based on partnership diversity.

        Multiple partnerships reduce dependency risk.

        Scoring:
        - 4+ active partnerships: 100
        - 3 active: 80
        - 2 active: 60
        - 1 active: 40
        - 0 active: 0
        """
        active = profile.active_partnerships

        if active >= 4:
            return Decimal("100")
        elif active == 3:
            return Decimal("80")
        elif active == 2:
            return Decimal("60")
        elif active == 1:
            return Decimal("40")

        return Decimal("0")

    def _get_top_partners(self, profile: PartnershipProfile, limit: int = 3) -> List[str]:
        """Get names of top-tier partners."""
        top_partners = []

        # Sort by tier (best first), then by deal value
        sorted_partnerships = sorted(
            profile.partnerships,
            key=lambda p: (
                -list(PartnerTier).index(p.partner_tier),
                -(p.total_potential_value or Decimal("0")),
            )
        )

        seen = set()
        for p in sorted_partnerships:
            if p.partner_name and p.partner_name not in seen:
                top_partners.append(p.partner_name)
                seen.add(p.partner_name)
            if len(top_partners) >= limit:
                break

        return top_partners

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        partnership_records: List[Dict[str, Any]],
        as_of_date: date,
    ) -> Dict[str, Any]:
        """
        Score partnership profile for entire universe.

        Args:
            universe: List of ticker dictionaries
            partnership_records: Partnership data
            as_of_date: Point-in-time date

        Returns:
            Dictionary with scores and diagnostics
        """
        # Load partnership data
        self.load_partnerships(partnership_records)

        scores_by_ticker = {}
        strength_distribution = {s.value: 0 for s in PartnershipStrength}

        for record in universe:
            ticker = record.get("ticker", "").upper()
            if not ticker:
                continue

            result = self.score_ticker(ticker, as_of_date)
            scores_by_ticker[ticker] = result

            strength = result["partnership_strength"]
            if strength in strength_distribution:
                strength_distribution[strength] += 1

        # Calculate summary statistics
        scores = [r["partnership_score"] for r in scores_by_ticker.values()]

        return {
            "scores_by_ticker": scores_by_ticker,
            "diagnostic_counts": {
                "total_scored": len(scores_by_ticker),
                "with_partnerships": sum(
                    1 for r in scores_by_ticker.values() if r["partnership_count"] > 0
                ),
                "with_top_tier": sum(
                    1 for r in scores_by_ticker.values() if r["top_tier_partners"] > 0
                ),
                "strength_distribution": strength_distribution,
            },
            "summary_stats": {
                "mean_score": (
                    sum(scores) / len(scores) if scores else Decimal("0")
                ).quantize(Decimal("0.01")),
                "max_score": max(scores) if scores else Decimal("0"),
                "min_score": min(scores) if scores else Decimal("0"),
            },
            "provenance": {
                "engine": "PartnershipEngine",
                "version": "1.0.0",
                "as_of_date": as_of_date.isoformat(),
                "partnerships_loaded": sum(
                    len(p) for p in self.partnerships_by_ticker.values()
                ),
            },
        }

    def get_partnership_details(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get detailed partnership information for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            List of partnership details
        """
        ticker = ticker.upper()
        partnerships = self.partnerships_by_ticker.get(ticker, [])

        return [
            {
                "partner_name": p.partner_name,
                "partner_tier": p.partner_tier.value,
                "deal_type": p.deal_type.value,
                "announcement_date": p.announcement_date.isoformat(),
                "upfront_payment": float(p.upfront_payment) if p.upfront_payment else None,
                "total_potential_value": float(p.total_potential_value) if p.total_potential_value else None,
                "indication": p.indication,
                "asset_name": p.asset_name,
                "is_active": p.is_active,
            }
            for p in partnerships
        ]

    def _add_audit(
        self, ticker: str, as_of_date: date, result: Dict[str, Any]
    ) -> None:
        """Add entry to audit trail."""
        self.audit_trail.append({
            "ticker": ticker,
            "as_of_date": as_of_date.isoformat(),
            "partnership_score": float(result["partnership_score"]),
            "partnership_strength": result["partnership_strength"],
            "partnership_count": result["partnership_count"],
            "top_tier_partners": result["top_tier_partners"],
        })

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return audit trail."""
        return self.audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear audit trail."""
        self.audit_trail = []

    def get_strength_rating(self, score: Decimal) -> str:
        """
        Convert numeric score to strength rating.

        Args:
            score: Partnership score (0-100)

        Returns:
            Rating string
        """
        if score >= Decimal("80"):
            return "exceptional"
        elif score >= Decimal("60"):
            return "strong"
        elif score >= Decimal("40"):
            return "moderate"
        else:
            return "weak"
