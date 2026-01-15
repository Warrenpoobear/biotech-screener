"""Wake Robin Data Pipeline - Market data providers with Morningstar integration."""

from .morningstar_data_provider import (
    MorningstarDataProvider,
    BatchMorningstarProvider,
    MORNINGSTAR_AVAILABLE,
    check_morningstar_availability,
    get_daily_returns,
    get_prices,
    get_morningstar_data_sets,
    get_morningstar_daily_returns_schema,
)

from .market_data_provider import (
    PriceDataProvider,
    BatchPriceProvider,
    get_log_returns,
    get_adv,
)

__all__ = [
    # Morningstar provider
    "MorningstarDataProvider",
    "BatchMorningstarProvider",
    "MORNINGSTAR_AVAILABLE",
    "check_morningstar_availability",
    "get_daily_returns",
    "get_morningstar_data_sets",
    "get_morningstar_daily_returns_schema",
    # Market data provider
    "PriceDataProvider",
    "BatchPriceProvider",
    "get_prices",
    "get_log_returns",
    "get_adv",
]
