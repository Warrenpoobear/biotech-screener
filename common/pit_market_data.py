"""Point-in-time market data for historical (research) runs.

``run_screen.py --as-of-date <past date>`` loads ``market_data.json`` exactly as it
stands today. Market cap, price, average volume, volatility and trailing returns in
that file are current values, so a historical run sees the future. The financial
module divides cash and burn by market cap, which made ``financial_score`` a proxy
for "this company's market cap grew later" in every regenerated history before
2026-11-13.

This module rebuilds one ``market_data.json`` per historical date from data that was
available on that date:

* price-derived fields come from closes on or before the date;
* market cap = share count from the latest SEC fact *filed* on or before the date,
  times that day's close, restated for splits between the fact date and the date.

Share-count rules (``r`` = point-in-time shares / current shares, measured as
point-in-time cap over the current cap scaled by the price path):

* a ticker whose ``r`` never reaches ``FOREIGN_MAX_RATIO`` is treated as a filer that
  reports ordinary shares instead of the traded ADS, and its share facts are not used;
* a point with ``r`` above ``MAX_RATIO`` is rejected (unhandled reverse split or a
  bad fact);
* a point with ``r`` below ``PRE_LISTING_FRACTION`` of the largest accepted ``r`` in
  the following ``LOOKAHEAD_DAYS`` is rejected as a pre-listing or pre-conversion
  count. It takes the share count of the next accepted filing instead. This uses a
  later filing to learn a share count that was public at the listing date; it does
  not use later prices.
* any other rejected point, and any ticker with no usable fact, falls back to the
  current market cap scaled by the price path. That removes the price look-ahead but
  still carries today's share count. ``market_cap_source`` records which rule applied.

Fields with no point-in-time source (short interest, float, beta) are set to None.

Research tooling only. Nothing here is read by the daily production run.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SOURCE_PIT = "pit_shares_x_close"
SOURCE_NEXT_FILING = "pit_shares_next_filing"
SOURCE_SCALED = "current_cap_scaled_by_price"

MAX_PRICE_GAP_DAYS = 10
MAX_SHARE_FACT_AGE_DAYS = 550
FOREIGN_MAX_RATIO = 0.3
MAX_RATIO = 1.6
PRE_LISTING_FRACTION = 0.25
LOOKAHEAD_DAYS = 185
LADDER_START = "2019-12-31"

NO_PIT_SOURCE_FIELDS: Tuple[str, ...] = ("short_percent", "short_ratio", "float_shares", "beta")


def _days(later: str, earlier: str) -> int:
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


@dataclass
class PriceData:
    """As-traded closes with volume, and split-adjusted closes, per ticker."""

    raw: Dict[str, List[Tuple[str, float, float]]] = field(default_factory=dict)
    adj: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for series in (self.raw, self.adj):
            for rows in series.values():
                rows.sort()
        self._raw_dates = {t: [r[0] for r in rows] for t, rows in self.raw.items()}
        self._adj_dates = {t: [r[0] for r in rows] for t, rows in self.adj.items()}

    @classmethod
    def load(cls, raw_csv: Path, adj_csv: Optional[Path] = None) -> "PriceData":
        raw: Dict[str, List[Tuple[str, float, float]]] = defaultdict(list)
        adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        with open(raw_csv, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    close = float(row["close"])
                    volume = float(row.get("volume") or 0)
                except (TypeError, ValueError, KeyError):
                    continue
                if close > 0 and row.get("ticker") and row.get("date"):
                    raw[row["ticker"].upper()].append((row["date"], close, volume))
        if adj_csv is not None and Path(adj_csv).exists():
            with open(adj_csv, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        close = float(row["close"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if close > 0 and row.get("ticker") and row.get("date"):
                        adj[row["ticker"].upper()].append((row["date"], close))
        return cls(dict(raw), dict(adj))

    def raw_index(self, ticker: str, day: str, max_gap: int = MAX_PRICE_GAP_DAYS) -> Optional[int]:
        return self._index(self._raw_dates.get(ticker), day, max_gap)

    def adj_index(self, ticker: str, day: str, max_gap: int = MAX_PRICE_GAP_DAYS) -> Optional[int]:
        return self._index(self._adj_dates.get(ticker), day, max_gap)

    @staticmethod
    def _index(dates: Optional[List[str]], day: str, max_gap: int) -> Optional[int]:
        if not dates:
            return None
        i = bisect.bisect_right(dates, day) - 1
        if i < 0 or _days(day, dates[i]) > max_gap:
            return None
        return i

    def split_basis(self, ticker: str, day: str) -> float:
        """As-traded close over split-adjusted close at ``day`` (1.0 when no later split)."""
        i = self.raw_index(ticker, day, max_gap=36500)
        j = self.adj_index(ticker, day, max_gap=36500)
        if i is None or j is None:
            return 1.0
        return self.raw[ticker][i][1] / self.adj[ticker][j][1]

    def adjusted_closes(self, ticker: str, day: str, lookback: int) -> List[float]:
        """Up to ``lookback`` closes ending at ``day`` on one split basis."""
        j = self.adj_index(ticker, day)
        if j is not None:
            return [c for _, c in self.adj[ticker][max(0, j - lookback + 1) : j + 1]]
        i = self.raw_index(ticker, day)
        if i is None:
            return []
        return [c for _, c, _ in self.raw[ticker][max(0, i - lookback + 1) : i + 1]]

    def last_date(self) -> Optional[str]:
        lasts = [dates[-1] for dates in self._raw_dates.values() if dates]
        return max(lasts) if lasts else None


def latest_fact(facts: Dict[str, list], key: str, day: str) -> Optional[dict]:
    """Most recent period-end fact for ``key`` that was filed on or before ``day``."""
    best: Optional[Tuple[Tuple[str, str], dict]] = None
    for fact in facts.get(key) or []:
        filed, end, val = fact.get("filed"), fact.get("end"), fact.get("val")
        if not filed or not end or val is None or filed > day:
            continue
        rank = (end, filed)
        if best is None or rank > best[0]:
            best = (rank, fact)
    return best[1] if best else None


def pit_share_count(facts: Dict[str, list], prices: PriceData, ticker: str, day: str) -> Optional[float]:
    """Share count known on ``day``, restated to the share basis traded on ``day``."""
    fact = latest_fact(facts, "shares_outstanding", day)
    if not fact:
        return None
    try:
        value = float(fact["val"])
    except (TypeError, ValueError):
        return None
    if value <= 0 or _days(day, fact["end"]) > MAX_SHARE_FACT_AGE_DAYS:
        return None
    basis_then = prices.split_basis(ticker, fact["end"])
    basis_now = prices.split_basis(ticker, day)
    return value * basis_then / basis_now if basis_now > 0 else None


def scaled_current_cap(current: dict, prices: PriceData, ticker: str, day: str) -> Optional[float]:
    """Today's market cap moved back along the split-adjusted price path."""
    cap = current.get("market_cap")
    series = prices.adj.get(ticker)
    j = prices.adj_index(ticker, day)
    if not cap or not series or j is None:
        return None
    return float(cap) * series[j][1] / series[-1][1]


def month_end_ladder(start: str, end: str) -> List[str]:
    """Calendar month ends from ``start`` to ``end`` inclusive."""
    out: List[str] = []
    cursor = date.fromisoformat(start).replace(day=1)
    last = date.fromisoformat(end)
    while True:
        nxt = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end = nxt - timedelta(days=1)
        if month_end > last:
            break
        if month_end.isoformat() >= start:
            out.append(month_end.isoformat())
        cursor = nxt
    return out


class ShareRatioJudge:
    """Decides, per ticker and date, whether a point-in-time share count is usable."""

    def __init__(self, prices: PriceData, current: Dict[str, dict], facts_by_ticker: Dict[str, Dict[str, list]]):
        self.prices = prices
        self.current = current
        self.facts = facts_by_ticker
        end = prices.last_date() or date.today().isoformat()
        self.ladder = month_end_ladder(LADDER_START, end)
        self._max_ratio: Dict[str, Optional[float]] = {}

    def ratio(self, ticker: str, day: str) -> Optional[float]:
        """Point-in-time shares over current shares, or None when it cannot be measured."""
        shares = pit_share_count(self.facts.get(ticker) or {}, self.prices, ticker, day)
        i = self.prices.raw_index(ticker, day)
        ref = scaled_current_cap(self.current.get(ticker) or {}, self.prices, ticker, day)
        if shares is None or i is None or not ref:
            return None
        return shares * self.prices.raw[ticker][i][1] / ref

    def _ticker_max(self, ticker: str) -> Optional[float]:
        if ticker not in self._max_ratio:
            values = [r for r in (self.ratio(ticker, d) for d in self.ladder) if r is not None]
            self._max_ratio[ticker] = max(values) if values else None
        return self._max_ratio[ticker]

    def is_foreign_basis(self, ticker: str) -> bool:
        top = self._ticker_max(ticker)
        return top is not None and top < FOREIGN_MAX_RATIO

    def _plausible(self, ticker: str, day: str) -> Optional[float]:
        """The ratio at ``day`` when it passes the level checks, else None."""
        r = self.ratio(ticker, day)
        if r is None or self.is_foreign_basis(ticker) or r > MAX_RATIO:
            return None
        return r

    def _later(self, ticker: str, day: str) -> List[float]:
        limit = (date.fromisoformat(day) + timedelta(days=LOOKAHEAD_DAYS)).isoformat()
        later = [d for d in self.ladder if day < d <= limit]
        return [r for r in (self._plausible(ticker, d) for d in later) if r is not None]

    def decide(self, ticker: str, day: str) -> Tuple[str, Optional[float]]:
        """Return (source, ratio to apply to the scaled current cap or None)."""
        r = self.ratio(ticker, day)
        if r is None:
            # Either no usable fact (caller falls back) or no reference to test against.
            has_shares = pit_share_count(self.facts.get(ticker) or {}, self.prices, ticker, day) is not None
            return (SOURCE_PIT, None) if has_shares else (SOURCE_SCALED, None)
        later = self._later(ticker, day)
        if self._plausible(ticker, day) is not None and not (later and r < PRE_LISTING_FRACTION * max(later)):
            return SOURCE_PIT, None
        # Rejected: take the next accepted filing's ratio when one is in range.
        for k, candidate in enumerate(later):
            following = later[k + 1 :]
            if not (following and candidate < PRE_LISTING_FRACTION * max(following)):
                return SOURCE_NEXT_FILING, candidate
        return SOURCE_SCALED, None


def build_market_records(
    day: str,
    current_records: Sequence[dict],
    prices: PriceData,
    facts_by_ticker: Dict[str, Dict[str, list]],
    judge: Optional[ShareRatioJudge] = None,
) -> Tuple[List[dict], Dict[str, int]]:
    """Market records as they could have been known on ``day``.

    Tickers with no close within ``MAX_PRICE_GAP_DAYS`` of ``day`` are left out: they
    were not trading and cannot be scored or held.
    """
    current = {str(r.get("ticker", "")).upper(): r for r in current_records if r.get("ticker")}
    judge = judge or ShareRatioJudge(prices, current, facts_by_ticker)
    out: List[dict] = []
    counts: Counter = Counter()
    for ticker, rec in current.items():
        i = prices.raw_index(ticker, day)
        if i is None:
            counts["not_trading"] += 1
            continue
        px_date, px, _ = prices.raw[ticker][i]
        new = dict(rec)
        new["price"] = px
        new["price_date"] = px_date
        new["price_source"] = "price_history_pit"
        new["collected_at"] = day
        new["market_cap_date"] = day

        volumes = [v for _, _, v in prices.raw[ticker][max(0, i - 89) : i + 1]]
        new["avg_volume_90d"] = sum(volumes) / len(volumes) if volumes else None
        new["avg_volume"] = int(sum(volumes[-30:]) / len(volumes[-30:])) if volumes else None

        closes = prices.adjusted_closes(ticker, day, 252)
        scale = px / closes[-1] if closes else 1.0  # back to as-traded terms on `day`
        if closes:
            quarter = closes[-63:]
            new["52w_high"] = max(closes) * scale
            new["52w_low"] = min(closes) * scale
            new["high_90d"] = max(quarter) * scale
            new["low_90d"] = min(quarter) * scale
            log_returns = [math.log(b / a) for a, b in zip(quarter[:-1], quarter[1:]) if a > 0 and b > 0]
            new["volatility_90d"] = statistics.stdev(log_returns) if len(log_returns) > 5 else None
            new["returns_1m"] = closes[-1] / closes[-22] - 1 if len(closes) > 22 else None
            new["returns_3m"] = closes[-1] / closes[-64] - 1 if len(closes) > 64 else None

        facts = facts_by_ticker.get(ticker) or {}
        source, ratio = judge.decide(ticker, day)
        ref = scaled_current_cap(rec, prices, ticker, day)
        cap: Optional[float] = None
        if source == SOURCE_PIT:
            shares = pit_share_count(facts, prices, ticker, day)
            cap = shares * px if shares else None
        elif source == SOURCE_NEXT_FILING and ref and ratio:
            cap = ref * ratio
        if cap is None:
            source, cap = SOURCE_SCALED, ref
        if cap is None:
            source = "none"
        new["market_cap"] = cap
        new["market_cap_source"] = source
        new["shares_outstanding"] = int(cap / px) if cap else None

        cash = latest_fact(facts, "cash", day)
        debt = latest_fact(facts, "long_term_debt", day)
        new["enterprise_value"] = (
            cap - float(cash["val"] if cash else 0) + float(debt["val"] if debt else 0) if cap else None
        )
        for name in NO_PIT_SOURCE_FIELDS:
            new[name] = None
        counts[source] += 1
        out.append(new)
    return out, dict(counts)


def load_pit_facts(pit_dir: Path, tickers: Iterable[str]) -> Dict[str, Dict[str, list]]:
    """Read ``<pit_dir>/<TICKER>.json`` fact stores for the given tickers."""
    out: Dict[str, Dict[str, list]] = {}
    for ticker in tickers:
        path = Path(pit_dir) / f"{ticker}.json"
        if not path.exists():
            continue
        try:
            out[ticker] = json.loads(path.read_text(encoding="utf-8")).get("facts") or {}
        except (OSError, ValueError):
            continue
    return out


class PitMarketDataBuilder:
    """Loads the inputs once and writes ``market_data.json`` for any number of dates."""

    def __init__(self, production_dir: Path):
        base = Path(production_dir)
        self.current_records = json.loads((base / "market_data.json").read_text(encoding="utf-8"))
        tickers = [str(r.get("ticker", "")).upper() for r in self.current_records if r.get("ticker")]
        self.prices = PriceData.load(base / "price_history.csv", base / "price_history_split_adj.csv")
        self.facts = load_pit_facts(base / "pit_financials", tickers)
        current = {str(r["ticker"]).upper(): r for r in self.current_records if r.get("ticker")}
        self.judge = ShareRatioJudge(self.prices, current, self.facts)

    def build(self, day: str) -> Tuple[List[dict], Dict[str, int]]:
        return build_market_records(day, self.current_records, self.prices, self.facts, self.judge)

    def write(self, day: str, out_path: Path) -> Dict[str, int]:
        records, counts = self.build(day)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(records), encoding="utf-8")
        tmp.replace(out_path)
        return counts
