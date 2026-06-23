"""Ingest existing screener universe from local cache-only sources."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Optional

from scientific_cartography.normalize.constants import GENERIC_COMPANY_NAMES
from scientific_cartography.schemas.company_schema import CompanyRecord


class ExistingUniverseIngest:
    """Read screener universe from local cache/snapshot files."""

    GENERIC_COMPANY_NAMES = GENERIC_COMPANY_NAMES

    def __init__(self, as_of_date: str = ""):
        """Initialize ingester.

        Args:
            as_of_date: Date for records (YYYY-MM-DD).
        """
        self.as_of_date = as_of_date

    def _make_company_id(self, ticker: Optional[str], company_name: Optional[str]) -> str:
        """Create stable company ID from ticker or company name."""
        if ticker:
            return f"COMPANY_TICKER_{ticker.upper()}"

        if company_name:
            normalized = company_name.lower().strip()
            hash_hex = hashlib.sha256(normalized.encode()).hexdigest()[:16]
            return f"COMPANY_{hash_hex}"

        return f"COMPANY_UNKNOWN_{hash([company_name, ticker])}"

    def _extract_json_company_name(self, item: dict) -> Optional[str]:
        """Extract company name from supported universe JSON schemas."""
        market_data = item.get("market_data") if isinstance(item.get("market_data"), dict) else {}
        for value in (
            item.get("company"),
            market_data.get("company_name"),
            item.get("company_name"),
            item.get("name"),
        ):
            if value and str(value).strip():
                company_name = str(value).strip()
                if company_name.lower() in self.GENERIC_COMPANY_NAMES:
                    continue
                return company_name
        return None

    def _extract_json_aliases(self, item: dict, company_name: Optional[str], ticker: Optional[str]) -> list[str]:
        """Extract deterministic aliases from alternate universe name fields."""
        market_data = item.get("market_data") if isinstance(item.get("market_data"), dict) else {}
        aliases = []
        seen = {value.lower() for value in (company_name, ticker) if value}
        for value in (item.get("company"), market_data.get("company_name"), item.get("company_name"), item.get("name")):
            if not value:
                continue
            alias = str(value).strip()
            if not alias or alias.lower() in self.GENERIC_COMPANY_NAMES:
                continue
            key = alias.lower()
            if key in seen:
                continue
            seen.add(key)
            aliases.append(alias)
        return aliases

    def ingest_from_csv(self, csv_path: Path) -> list[CompanyRecord]:
        """Ingest universe from CSV with ticker/company columns.

        Expected columns: ticker, company (name), cik (optional)

        Args:
            csv_path: Path to CSV file.

        Returns:
            List of CompanyRecords.
        """
        records = []
        if not csv_path.exists():
            return records

        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                seen = set()

                for row in reader:
                    ticker = row.get("ticker", "").strip() if row.get("ticker") else None
                    company = (
                        row.get("company", "").strip()
                        if row.get("company")
                        else row.get("name", "").strip() if row.get("name") else None
                    )
                    cik = row.get("cik", "").strip() if row.get("cik") else None

                    # Skip if no identifying information
                    if not (ticker or company):
                        continue

                    # Deduplicate by ticker
                    key = ticker or company
                    if key.lower() in seen:
                        continue
                    seen.add(key.lower())

                    company_id = self._make_company_id(ticker, company)
                    record = CompanyRecord(
                        company_id=company_id,
                        ticker=ticker,
                        company_name=company or ticker or "Unknown",
                        cik=cik,
                        is_public=bool(ticker),
                        as_of_date=self.as_of_date,
                        source_refs=[str(csv_path)],
                        confidence=0.95 if (ticker and company) else 0.85,
                    )
                    records.append(record)

        except Exception as e:
            print(f"Warning: Failed to ingest universe from {csv_path}: {e}")

        return records

    def ingest_from_json(self, json_path: Path) -> list[CompanyRecord]:
        """Ingest universe from JSON array of company objects.

        Expected structure:
        - [{"ticker": "...", "company": "...", "cik": "..."}, ...]
        - production_data/universe.json rows with market_data.company_name/name.

        Args:
            json_path: Path to JSON file.

        Returns:
            List of CompanyRecords.
        """
        records = []
        if not json_path.exists():
            return records

        try:
            with open(json_path) as f:
                data = json.load(f)

            if not isinstance(data, list):
                return records

            seen = set()

            for item in data:
                if not isinstance(item, dict):
                    continue

                ticker = item.get("ticker", "").strip() if item.get("ticker") else None
                company = self._extract_json_company_name(item)
                cik = item.get("cik", "").strip() if item.get("cik") else None
                market_data = item.get("market_data") if isinstance(item.get("market_data"), dict) else {}
                exchange = item.get("exchange") or market_data.get("exchange")

                if not (ticker or company):
                    continue

                key = (ticker or company).lower()
                if key in seen:
                    continue
                seen.add(key)

                company_id = self._make_company_id(ticker, company)
                record = CompanyRecord(
                    company_id=company_id,
                    ticker=ticker,
                    company_name=company or ticker or "Unknown",
                    cik=cik,
                    aliases=self._extract_json_aliases(item, company, ticker),
                    exchange=exchange,
                    is_public=bool(ticker),
                    as_of_date=self.as_of_date,
                    source_refs=[str(json_path)],
                    confidence=0.95 if (ticker and company and company != ticker) else 0.85,
                )
                records.append(record)

        except Exception as e:
            print(f"Warning: Failed to ingest universe from {json_path}: {e}")

        return records

    def ingest_from_rankings_csv(self, csv_path: Path) -> list[CompanyRecord]:
        """Ingest from standard rankings.csv format.

        Looks for ticker and company columns for sponsor resolution.

        Args:
            csv_path: Path to rankings.csv.

        Returns:
            List of CompanyRecords.
        """
        records = []
        if not csv_path.exists():
            return records

        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                seen = set()

                for row in reader:
                    ticker = row.get("ticker", "").strip() if row.get("ticker") else None
                    company = row.get("company", "").strip() if row.get("company") else None

                    if not ticker:
                        continue

                    if ticker.lower() in seen:
                        continue
                    seen.add(ticker.lower())

                    # Use company name if available, otherwise fall back to ticker
                    company_name = company or ticker
                    company_id = self._make_company_id(ticker, company_name)

                    # Higher confidence if we have both ticker and company name
                    confidence = 0.95 if company else 0.85

                    record = CompanyRecord(
                        company_id=company_id,
                        ticker=ticker,
                        company_name=company_name,
                        is_public=True,
                        as_of_date=self.as_of_date,
                        source_refs=[str(csv_path)],
                        confidence=confidence,
                    )
                    records.append(record)

        except Exception as e:
            print(f"Warning: Failed to ingest from {csv_path}: {e}")

        return records
