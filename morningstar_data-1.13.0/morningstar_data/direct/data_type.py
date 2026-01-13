"""
Public method parameter and return types
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ._data_type import RaiseEnum


class Universe:
    def __init__(self, n: str, u: str):
        self.name = n
        self.universe_id = u


class Frequency(str, Enum, metaclass=RaiseEnum):
    def __new__(cls, value: str, data_point_id: str, abbr: str) -> Frequency:
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    def __init__(self, value: str, data_point_id: str, abbr: str) -> None:
        self.data_point_id = data_point_id
        self.abbr = abbr

    daily = ("daily", "HS793", "d")
    weekly = ("weekly", "HP002", "w")
    monthly = ("monthly", "HP010", "m")
    quarterly = ("quarterly", "HP020", "q")
    yearly = ("yearly", "HS803", "a")


class Blank(str, Enum, metaclass=RaiseEnum):
    warning = ("warning",)
    ignore = ("ignore",)
    update = ("update",)

    @classmethod
    def options(cls) -> str:
        return ", ".join(cls._member_map_.keys())


class PeerGroupMethodology(str, Enum):
    MORNINGSTAR = ("MORNINGSTAR",)
    SIMPLE = ("SIMPLE",)


class Order(str, Enum):
    ASC = ("ASC",)
    DESC = ("DESC",)


class TimeSeriesFormat(str, Enum):
    LONG = "Long"
    WIDE = "Wide"
    LONG_WITHOUT_NAME = "LongWithoutName"


# Calculation window type
class CalculationWindowType(str, Enum):
    SINGLE_DATA_POINT = ("1",)
    ROLLING_WINDOW = ("2",)
    FORWARD_EXTENDING_WINDOW = ("3",)
    BACKWARD_EXTENDING_WINDOW = ("4",)


class WarningBehavior(str, Enum, metaclass=RaiseEnum):
    WARNING = "WARNING"
    IGNORE = "IGNORE"
    FAIL = "FAIL"


@dataclass
class InvestmentIdentifier:
    sec_id: Optional[str] = None
    ticker: Optional[str] = None
    cusip: Optional[str] = None
    isin: Optional[str] = None
    base_currency: Optional[str] = None
    exchange: Optional[str] = None


class HoldingsView(str, Enum):
    FULL = "full"
    BEST_AVAILABLE = "bestAvailable"


class PortfolioDepth(str, Enum):
    BASE_LEVEL = "baseLevel"
    LOOK_THROUGH = "lookThrough"


class EnrichmentType(str, Enum):
    LATEST_AVAILABLE = "latestAvailable"
    POINT_IN_TIME = "pointInTime"
    LATEST_MONTH_END = "latestMonthEnd"
