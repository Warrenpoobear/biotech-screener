from enum import Enum


class PortfolioType(Enum):
    def __init__(self, key: str, abbr: str) -> None:
        self.key = key
        self.abbr = abbr

    @classmethod
    def get_full_name_by_abbr(cls, abbr: str) -> str:
        # Api return values are "MD","AC","BM","UA", we use abbr to get the full portfolio type name.
        for member in cls._member_map_.values():
            if member._value_[1] == abbr:
                return member.name
        return ""

    # The key is used to create and get a portfolio.
    # The abbr is the portfolio list response,we can use this to get the full portfolio type name.
    model_portfolios = ("60", "MD")
    custom_benchmarks = ("63", "BM")
