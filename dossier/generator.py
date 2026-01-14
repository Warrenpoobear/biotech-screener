"""
generator.py - Main dossier generator orchestrator

Coordinates data fetching and section generation to produce
complete institutional investment dossiers.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data_fetchers import DossierDataFetcher, DossierData
from .section_generators import DossierSectionGenerator

logger = logging.getLogger(__name__)


class DossierGenerator:
    """
    Main orchestrator for institutional dossier generation.

    Coordinates:
    - Data fetching from screening outputs and external sources
    - Section generation for each part of the dossier
    - Template filling and output generation
    """

    def __init__(
        self,
        data_dir: str = "./production_data",
        output_dir: str = "./reports/dossiers",
        use_cache: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

        # Initialize components
        self.fetcher = DossierDataFetcher(data_dir=data_dir, use_cache=use_cache)
        self.section_gen = DossierSectionGenerator()

        # Load base data
        self.fetcher.load_base_data()

    def get_top_n_tickers(self, results_path: str, n: int) -> List[str]:
        """Get top N tickers from screening results."""
        return self.fetcher.get_top_n_tickers(results_path, n)

    def get_all_ranked_tickers(self, results_path: str) -> List[str]:
        """Get all ranked tickers from screening results."""
        return self.fetcher.get_all_ranked_tickers(results_path)

    def _load_template(self) -> str:
        """Load the dossier template."""
        template_path = Path(__file__).parent / "templates" / "institutional_dossier.md"
        if template_path.exists():
            with open(template_path) as f:
                return f.read()

        # Default template if file not found
        return """# INSTITUTIONAL INVESTMENT DOSSIER: {company_name} ({ticker})

**Prepared for Investment Committee Review | {report_date}**
**Strictly Confidential**

---

{executive_summary}

---

{investment_thesis}

---

{catalyst_analysis}

---

{clinical_review}

---

{financial_analysis}

---

{risk_assessment}

---

{position_sizing}

---

{final_recommendation}

---

## DATA QUALITY & DISCLAIMERS

{data_quality}

---

**Report Prepared By:** Wake Robin Algorithmic Screening System
**Generated:** {report_date}
**Distribution:** Investment Committee Members Only
**Classification:** Strictly Confidential
"""

    def _generate_data_quality_section(self, data: DossierData) -> str:
        """Generate data quality disclaimer section."""
        dq = data.data_quality
        lines = [
            "### Data Coverage",
            "",
            "| Data Source | Available |",
            "|-------------|-----------|",
            f"| Company Profile | {'Yes' if dq.get('has_profile') else 'No'} |",
            f"| Market Data | {'Yes' if dq.get('has_market_data') else 'No'} |",
            f"| Financial Data | {'Yes' if dq.get('has_financial_data') else 'No'} |",
            f"| Clinical Trials | {'Yes' if dq.get('has_clinical_data') else 'No'} |",
            f"| Catalyst Events | {'Yes' if dq.get('has_catalyst_data') else 'No'} |",
            "",
            f"**Overall Data Coverage:** {dq.get('overall_coverage', 0):.0f}%",
            "",
            "### Disclaimers",
            "",
            "- This report is algorithmically generated for informational purposes only",
            "- Not financial advice - verify all data before making investment decisions",
            "- Historical data does not guarantee future results",
            "- Clinical trial outcomes are inherently uncertain",
            "- Past screening performance is not indicative of future results",
        ]
        return "\n".join(lines)

    def generate(
        self,
        ticker: str,
        as_of_date: str,
        results_path: Optional[str] = None,
        output_format: str = "md",
        skip_sec: bool = False,
        skip_trials: bool = False,
    ) -> str:
        """
        Generate complete institutional dossier for a ticker.

        Args:
            ticker: Stock ticker symbol
            as_of_date: Snapshot date (YYYY-MM-DD)
            results_path: Path to screening results file
            output_format: Output format ("md", "pdf", "both")
            skip_sec: Skip SEC filing fetch
            skip_trials: Skip clinical trials fetch

        Returns:
            Path to generated dossier file
        """
        ticker = ticker.upper()
        logger.info(f"Generating dossier for {ticker}...")

        # Fetch all data
        data = self.fetcher.fetch_all_data(
            ticker=ticker,
            as_of_date=as_of_date,
            results_path=results_path,
        )

        # Generate all sections
        sections = self.section_gen.generate_all_sections(data)

        # Add data quality section
        sections["data_quality"] = self._generate_data_quality_section(data)

        # Load and fill template
        template = self._load_template()

        report = template.format(
            company_name=data.profile.company_name or ticker,
            ticker=ticker,
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **sections,
        )

        # Save markdown
        md_path = self.output_dir / f"{ticker}_dossier_{as_of_date}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Saved markdown: {md_path}")

        # Generate PDF if requested
        if output_format in ["pdf", "both"]:
            pdf_path = self._convert_to_pdf(md_path)
            if pdf_path:
                logger.info(f"Saved PDF: {pdf_path}")

        return str(md_path)

    def _convert_to_pdf(self, md_path: Path) -> Optional[Path]:
        """Convert markdown to PDF (requires pandoc or similar)."""
        try:
            import subprocess

            pdf_path = md_path.with_suffix(".pdf")

            # Try pandoc first
            result = subprocess.run(
                ["pandoc", str(md_path), "-o", str(pdf_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return pdf_path

            logger.warning("pandoc not available, skipping PDF generation")
            return None

        except Exception as e:
            logger.warning(f"PDF conversion failed: {e}")
            return None

    def generate_batch(
        self,
        tickers: List[str],
        as_of_date: str,
        results_path: Optional[str] = None,
        output_format: str = "md",
    ) -> List[str]:
        """
        Generate dossiers for multiple tickers.

        Args:
            tickers: List of ticker symbols
            as_of_date: Snapshot date
            results_path: Path to screening results
            output_format: Output format

        Returns:
            List of generated file paths
        """
        generated = []

        for ticker in tickers:
            try:
                path = self.generate(
                    ticker=ticker,
                    as_of_date=as_of_date,
                    results_path=results_path,
                    output_format=output_format,
                )
                generated.append(path)
                logger.info(f"Generated: {ticker}")
            except Exception as e:
                logger.error(f"Failed to generate dossier for {ticker}: {e}")
                continue

        return generated


def main():
    """CLI entry point - see scripts/generate_dossier.py for full CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate biotech dossiers")
    parser.add_argument("ticker", help="Stock ticker symbol")
    parser.add_argument("--date", required=True, help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--results", help="Path to screening results")
    parser.add_argument("--data-dir", default="./production_data")
    parser.add_argument("--output-dir", default="./reports/dossiers")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    generator = DossierGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    path = generator.generate(
        ticker=args.ticker,
        as_of_date=args.date,
        results_path=args.results,
    )

    print(f"\nDossier generated: {path}")


if __name__ == "__main__":
    main()
