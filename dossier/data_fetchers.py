"""
data_fetchers.py - Data fetching for institutional dossiers

Fetches data from:
- Screening system outputs (production_data/)
- Yahoo Finance (market data)
- SEC EDGAR (filings)
- ClinicalTrials.gov (trial data)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompanyProfile:
    """Company profile data."""
    ticker: str
    company_name: str = ""
    description: str = ""
    sector: str = "Healthcare"
    industry: str = "Biotechnology"
    employees: int = 0
    headquarters: str = ""
    website: str = ""
    founded: str = ""


@dataclass
class MarketData:
    """Market data for a security."""
    ticker: str
    price: float = 0.0
    market_cap: float = 0.0
    volume_avg_30d: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    beta: float = 1.0
    shares_outstanding: float = 0.0


@dataclass
class FinancialData:
    """Financial statement data."""
    ticker: str
    cash: float = 0.0
    debt: float = 0.0
    revenue_ttm: float = 0.0
    net_income_ttm: float = 0.0
    burn_rate_monthly: float = 0.0
    runway_months: float = 0.0
    last_filing_date: str = ""


@dataclass
class ClinicalData:
    """Clinical trial data."""
    ticker: str
    lead_stage: str = ""
    active_trials: int = 0
    completed_trials: int = 0
    total_trials: int = 0
    trials: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CatalystData:
    """Catalyst events data."""
    ticker: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    catalyst_score: float = 0.0
    near_term_events: int = 0
    severe_negatives: int = 0


@dataclass
class RankingData:
    """Screening system ranking data."""
    ticker: str
    rank: int = 0
    composite_score: float = 0.0
    financial_score: float = 0.0
    clinical_score: float = 0.0
    catalyst_score: float = 0.0
    weight: float = 0.0
    excluded: bool = False
    exclusion_reason: str = ""


@dataclass
class DossierData:
    """Complete data package for dossier generation."""
    ticker: str
    as_of_date: str
    profile: CompanyProfile = None
    market: MarketData = None
    financial: FinancialData = None
    clinical: ClinicalData = None
    catalyst: CatalystData = None
    ranking: RankingData = None
    data_quality: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.profile is None:
            self.profile = CompanyProfile(ticker=self.ticker)
        if self.market is None:
            self.market = MarketData(ticker=self.ticker)
        if self.financial is None:
            self.financial = FinancialData(ticker=self.ticker)
        if self.clinical is None:
            self.clinical = ClinicalData(ticker=self.ticker)
        if self.catalyst is None:
            self.catalyst = CatalystData(ticker=self.ticker)
        if self.ranking is None:
            self.ranking = RankingData(ticker=self.ticker)


class DossierDataFetcher:
    """
    Fetches all data needed for institutional dossier generation.

    Integrates with:
    - Screening system outputs (production_data/)
    - Universe data (company profiles, market data)
    - Trial records
    - Catalyst events
    - Financial data
    """

    def __init__(
        self,
        data_dir: str = "./production_data",
        use_cache: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.use_cache = use_cache
        self._cache: Dict[str, Any] = {}

        # Load core data files
        self._universe_data: Dict[str, Dict] = {}
        self._market_data: Dict[str, Dict] = {}
        self._financial_data: Dict[str, Dict] = {}
        self._trial_records: List[Dict] = []

    def load_base_data(self) -> None:
        """Load all base data files from production_data/."""
        logger.info("Loading base data files...")

        # Load universe
        universe_path = self.data_dir / "universe.json"
        if universe_path.exists():
            with open(universe_path) as f:
                universe = json.load(f)
                if isinstance(universe, list):
                    for company in universe:
                        ticker = company.get("ticker", "")
                        if ticker:
                            # Flatten nested structures for easier access
                            flat_company = {"ticker": ticker}
                            # Copy market_data fields to top level
                            if "market_data" in company:
                                flat_company.update(company["market_data"])
                            # Copy financial fields
                            if "financials" in company:
                                flat_company.update(company["financials"])
                            if "financial_data" in company:
                                flat_company.update(company["financial_data"])
                            # Copy clinical fields
                            if "clinical" in company:
                                flat_company["clinical"] = company["clinical"]
                            if "clinical_data" in company:
                                flat_company["clinical"] = company["clinical_data"]
                            # Keep original structure too
                            flat_company["_raw"] = company
                            self._universe_data[ticker] = flat_company
                elif isinstance(universe, dict):
                    self._universe_data = universe
            logger.info(f"Loaded universe: {len(self._universe_data)} companies")

        # Load market data
        market_path = self.data_dir / "market_data.json"
        if market_path.exists():
            with open(market_path) as f:
                market_raw = json.load(f)
                if isinstance(market_raw, list):
                    for item in market_raw:
                        ticker = item.get("ticker", "")
                        if ticker:
                            self._market_data[ticker] = item
                elif isinstance(market_raw, dict):
                    self._market_data = market_raw
            logger.info(f"Loaded market data: {len(self._market_data)} records")

        # Load financial data
        financial_path = self.data_dir / "financial_data.json"
        if financial_path.exists():
            with open(financial_path) as f:
                fin_raw = json.load(f)
                if isinstance(fin_raw, list):
                    for item in fin_raw:
                        ticker = item.get("ticker", "")
                        if ticker:
                            self._financial_data[ticker] = item
                elif isinstance(fin_raw, dict):
                    self._financial_data = fin_raw
            logger.info(f"Loaded financial data: {len(self._financial_data)} records")

        # Load trial records
        trial_path = self.data_dir / "trial_records.json"
        if trial_path.exists():
            with open(trial_path) as f:
                self._trial_records = json.load(f)
            logger.info(f"Loaded trial records: {len(self._trial_records)} trials")

    def fetch_company_profile(self, ticker: str) -> CompanyProfile:
        """Fetch company profile from universe data."""
        ticker = ticker.upper()
        company = self._universe_data.get(ticker, {})

        return CompanyProfile(
            ticker=ticker,
            company_name=company.get("company_name", company.get("name", ticker)),
            description=company.get("description", ""),
            sector=company.get("sector", "Healthcare"),
            industry=company.get("industry", "Biotechnology"),
            employees=company.get("employees", 0),
            headquarters=company.get("headquarters", company.get("hq", "")),
            website=company.get("website", ""),
            founded=company.get("founded", ""),
        )

    def fetch_market_data(self, ticker: str) -> MarketData:
        """Fetch market data for ticker."""
        ticker = ticker.upper()
        market = self._market_data.get(ticker, {})
        universe = self._universe_data.get(ticker, {})

        # Try multiple sources for market data
        price = market.get("price", 0) or universe.get("price", 0)
        market_cap = market.get("market_cap", 0) or universe.get("market_cap", 0)

        return MarketData(
            ticker=ticker,
            price=float(price) if price else 0.0,
            market_cap=float(market_cap) if market_cap else 0.0,
            volume_avg_30d=float(market.get("volume_avg_30d", 0) or 0),
            high_52w=float(market.get("52_week_high", 0) or market.get("high_52w", 0) or 0),
            low_52w=float(market.get("52_week_low", 0) or market.get("low_52w", 0) or 0),
            beta=float(market.get("beta", 1.0) or 1.0),
            shares_outstanding=float(market.get("shares_outstanding", 0) or 0),
        )

    def fetch_financial_data(self, ticker: str) -> FinancialData:
        """Fetch financial data for ticker."""
        ticker = ticker.upper()
        fin = self._financial_data.get(ticker, {})
        universe = self._universe_data.get(ticker, {})

        # Extract financial fields
        cash = float(fin.get("cash", 0) or universe.get("cash", 0) or 0)
        debt = float(fin.get("debt", 0) or universe.get("debt", 0) or 0)
        revenue = float(fin.get("revenue_ttm", 0) or fin.get("revenue", 0) or 0)
        net_income = float(fin.get("net_income_ttm", 0) or fin.get("net_income", 0) or 0)

        # Estimate burn rate (if revenue < 0 or company is pre-revenue)
        burn_rate = 0.0
        if net_income < 0:
            burn_rate = abs(net_income) / 12
        elif cash > 0:
            # Assume typical biotech burn rate if no data
            burn_rate = cash / 24  # Assume 24 month runway as default

        # Calculate runway
        runway = 0.0
        if burn_rate > 0:
            runway = cash / burn_rate

        return FinancialData(
            ticker=ticker,
            cash=cash,
            debt=debt,
            revenue_ttm=revenue,
            net_income_ttm=net_income,
            burn_rate_monthly=burn_rate,
            runway_months=runway,
            last_filing_date=fin.get("last_filing_date", ""),
        )

    def fetch_clinical_data(self, ticker: str) -> ClinicalData:
        """Fetch clinical trial data for ticker."""
        ticker = ticker.upper()

        # Filter trials for this ticker
        ticker_trials = [
            t for t in self._trial_records
            if t.get("ticker", "").upper() == ticker
        ]

        # Count by status
        active_count = 0
        completed_count = 0
        for trial in ticker_trials:
            status = trial.get("status", "").upper()
            if status in ["RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"]:
                active_count += 1
            elif status in ["COMPLETED"]:
                completed_count += 1

        # Determine lead stage
        lead_stage = "preclinical"
        for trial in ticker_trials:
            phase = trial.get("phase", "").lower()
            if "phase 3" in phase or "phase3" in phase:
                lead_stage = "phase_3"
                break
            elif "phase 2" in phase or "phase2" in phase:
                if lead_stage not in ["phase_3"]:
                    lead_stage = "phase_2"
            elif "phase 1" in phase or "phase1" in phase:
                if lead_stage not in ["phase_2", "phase_3"]:
                    lead_stage = "phase_1"

        return ClinicalData(
            ticker=ticker,
            lead_stage=lead_stage,
            active_trials=active_count,
            completed_trials=completed_count,
            total_trials=len(ticker_trials),
            trials=ticker_trials[:20],  # Top 20 trials
        )

    def fetch_catalyst_data(self, ticker: str, as_of_date: str) -> CatalystData:
        """Fetch catalyst events for ticker."""
        ticker = ticker.upper()

        # Load catalyst events file
        catalyst_file = self.data_dir / f"catalyst_events_vnext_{as_of_date}.json"
        if not catalyst_file.exists():
            # Try to find most recent
            catalyst_files = sorted(self.data_dir.glob("catalyst_events_vnext_*.json"))
            if catalyst_files:
                catalyst_file = catalyst_files[-1]

        events = []
        catalyst_score = 0.0

        if catalyst_file.exists():
            with open(catalyst_file) as f:
                catalyst_data = json.load(f)

            # Find ticker in catalyst data - handle vnext format
            if isinstance(catalyst_data, dict):
                # Check for vnext format (summaries dict keyed by ticker)
                summaries = catalyst_data.get("summaries", {})
                if ticker in summaries:
                    summary = summaries[ticker]
                    events = summary.get("events", [])
                    # Get score from scores sub-dict
                    scores = summary.get("scores", {})
                    catalyst_score = float(scores.get("score_blended", 50) or 50)
                else:
                    # Try legacy format
                    ticker_events = catalyst_data.get("events_by_ticker", {}).get(ticker, [])
                    if ticker_events:
                        events = ticker_events
                    # Get score from ticker_summaries
                    ticker_summaries = catalyst_data.get("ticker_summaries", {})
                    if ticker in ticker_summaries:
                        summary = ticker_summaries[ticker]
                        catalyst_score = float(summary.get("catalyst_score", 0) or 0)

        # Count near-term and severe negative
        near_term = len([e for e in events if e.get("days_until", 999) <= 90])
        severe_neg = len([e for e in events if e.get("severity") == "SEVERE_NEGATIVE"])

        return CatalystData(
            ticker=ticker,
            events=events,
            catalyst_score=catalyst_score,
            near_term_events=near_term,
            severe_negatives=severe_neg,
        )

    def fetch_ranking_data(self, ticker: str, results_path: str) -> RankingData:
        """Fetch ranking from screening results."""
        ticker = ticker.upper()

        results_file = Path(results_path)
        if not results_file.exists():
            return RankingData(ticker=ticker)

        with open(results_file) as f:
            results = json.load(f)

        # Get from module_5_composite
        m5 = results.get("module_5_composite", {})
        ranked = m5.get("ranked_securities", [])
        excluded = m5.get("excluded_securities", [])

        # Find ticker in ranked
        for i, r in enumerate(ranked):
            if r.get("ticker", "").upper() == ticker:
                return RankingData(
                    ticker=ticker,
                    rank=i + 1,
                    composite_score=float(r.get("composite_score", 0) or 0),
                    financial_score=float(r.get("financial_score", 0) or 0),
                    clinical_score=float(r.get("clinical_score", 0) or 0),
                    catalyst_score=float(r.get("catalyst_score", 0) or 0),
                    weight=float(r.get("weight", 0) or 0),
                    excluded=False,
                )

        # Check excluded
        for e in excluded:
            if e.get("ticker", "").upper() == ticker:
                return RankingData(
                    ticker=ticker,
                    excluded=True,
                    exclusion_reason=e.get("exclusion_reason", "Unknown"),
                )

        return RankingData(ticker=ticker)

    def fetch_all_data(
        self,
        ticker: str,
        as_of_date: str,
        results_path: Optional[str] = None,
    ) -> DossierData:
        """
        Fetch all data needed for dossier generation.

        Args:
            ticker: Stock ticker symbol
            as_of_date: Snapshot date (YYYY-MM-DD)
            results_path: Path to screening results file

        Returns:
            DossierData with all fetched data
        """
        ticker = ticker.upper()

        # Ensure base data is loaded
        if not self._universe_data:
            self.load_base_data()

        # Fetch all components
        profile = self.fetch_company_profile(ticker)
        market = self.fetch_market_data(ticker)
        financial = self.fetch_financial_data(ticker)
        clinical = self.fetch_clinical_data(ticker)
        catalyst = self.fetch_catalyst_data(ticker, as_of_date)

        # Fetch ranking if results path provided
        ranking = RankingData(ticker=ticker)
        if results_path:
            ranking = self.fetch_ranking_data(ticker, results_path)

        # Build data quality metrics
        data_quality = {
            "has_profile": bool(profile.company_name and profile.company_name != ticker),
            "has_market_data": market.price > 0,
            "has_financial_data": financial.cash > 0 or financial.revenue_ttm > 0,
            "has_clinical_data": clinical.total_trials > 0,
            "has_catalyst_data": len(catalyst.events) > 0,
            "has_ranking": ranking.rank > 0,
            "overall_coverage": 0.0,
        }

        # Calculate overall coverage
        coverage_count = sum([
            data_quality["has_profile"],
            data_quality["has_market_data"],
            data_quality["has_financial_data"],
            data_quality["has_clinical_data"],
            data_quality["has_catalyst_data"],
        ])
        data_quality["overall_coverage"] = coverage_count / 5 * 100

        return DossierData(
            ticker=ticker,
            as_of_date=as_of_date,
            profile=profile,
            market=market,
            financial=financial,
            clinical=clinical,
            catalyst=catalyst,
            ranking=ranking,
            data_quality=data_quality,
        )

    def get_top_n_tickers(self, results_path: str, n: int) -> List[str]:
        """Get top N tickers from screening results."""
        results_file = Path(results_path)
        if not results_file.exists():
            return []

        with open(results_file) as f:
            results = json.load(f)

        ranked = results.get("module_5_composite", {}).get("ranked_securities", [])
        return [r.get("ticker", "") for r in ranked[:n]]

    def get_all_ranked_tickers(self, results_path: str) -> List[str]:
        """Get all ranked tickers from screening results."""
        results_file = Path(results_path)
        if not results_file.exists():
            return []

        with open(results_file) as f:
            results = json.load(f)

        ranked = results.get("module_5_composite", {}).get("ranked_securities", [])
        return [r.get("ticker", "") for r in ranked]


if __name__ == "__main__":
    # Quick test
    fetcher = DossierDataFetcher(data_dir="./production_data")
    fetcher.load_base_data()

    data = fetcher.fetch_all_data("KMDA", "2026-01-14", "results_fixed.json")
    print(f"Ticker: {data.ticker}")
    print(f"Company: {data.profile.company_name}")
    print(f"Price: ${data.market.price:.2f}")
    print(f"Market Cap: ${data.market.market_cap/1e6:.1f}M")
    print(f"Rank: {data.ranking.rank}")
    print(f"Score: {data.ranking.composite_score}")
    print(f"Data Quality: {data.data_quality['overall_coverage']:.0f}%")
