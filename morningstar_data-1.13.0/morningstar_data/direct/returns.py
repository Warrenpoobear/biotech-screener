import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from . import _decorator, investment
from ._config_key import FORMAT_DATE
from .data_type import CalculationWindowType, Frequency, TimeSeriesFormat


@_decorator.typechecked
def returns(
    investments: Union[List[str], str, Dict[str, Any]],
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    freq: Union[Frequency, str] = Frequency.monthly,
    currency: Optional[str] = None,
) -> pd.DataFrame:
    warnings.warn(
        "The returns function is deprecated and will be removed in the next major version. Use get_returns instead",
        FutureWarning,
        stacklevel=2,
    )
    return get_returns(investments, start_date, end_date, freq, currency)


@_decorator.typechecked
def get_returns(
    investments: Union[List[str], str, Dict[str, Any]],
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    freq: Union[Frequency, str] = Frequency.monthly,
    currency: Optional[str] = None,
) -> pd.DataFrame:
    """A shortcut function to fetch return data for the specified investments.

    Args:
        investments (:obj:`Union`, `required`): Defines the investments to fetch. Input can be:

            * Investment IDs (:obj:`list`, `optional`): Investment identifiers, in the format of SecId;Universe or just SecId. E.g., ["F00000YOOK;FO","FOUSA00CFV;FO"] or ["F00000YOOK","FOUSA00CFV"]. Use the `investments <./lookup.html#morningstar_data.direct.investments>`_ function to discover identifiers.
            * Investment List ID (:obj:`str`, `optional`): Saved investment list in Morningstar Direct. Use the `get_investment_lists <./lists.html#morningstar_data.direct.user_items.get_investment_lists>`_ function to discover saved lists.
            * Search Criteria  ID (:obj:`str`, `optional`): Saved search criteria in Morningstar Direct. Use the `get_search_criteria <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria>`_ function to discover saved search criteria.
            * Search Criteria Condition (:obj:`dict`, `optional`): Search criteria definition. See details in the Reference section of `get_investment_data <#morningstar_data.direct.get_investment_data>`_ or use the `get_search_criteria_conditions <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria_conditions>`_ function to discover the definition of a saved search criteria.

        start_date (:obj:`str`): Start date of a date range for retrieving data. The format is
            YYYY-MM-DD, e.g., "2020-01-01".
        end_date (:obj:`str`, `optional`): End date of a date range for retrieving data. If no value is provided for
            end_date, current date will be used. The format is YYYY-MM-DD, e.g., "2020-01-01".
        freq (:obj:`md.direct.data_type.Frequency`): Enumeration of type md.direct.data_type.Frequency, which can be 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'. E.g., ``md.direct.data_type.Frequency.monthly``
        currency (:obj:`str`, `optional`): Three character code for the desired currency of returns, e.g., "USD".  Use the `currency_codes <./lookup.html#morningstar_data.lookup.currency_codes>`_ function to discover possible values.

    :Returns:
        DataFrame: A DataFrame object with returns data.

    :Examples:
        Get monthly returns.

    ::

        import morningstar_data as md


        df = md.direct.get_returns(
            investments=["F00000VKPI", "F000014B1Y"], start_date="2020-10-01", freq=md.direct.data_type.Frequency.monthly
        )
        df

    :Output:
        ==============  ==========  ====================================
        ID              Date          Monthly Return
        ==============  ==========  ====================================
        F00000VKPI      2020-10-31     -2.121865
        F00000VKPI      2020-11-30     6.337255
        F00000VKPI      2020-12-31     1.464777
        ...                ...         ...
        ==============  ==========  ====================================

    :Errors:
        AccessDeniedError: Raised when the user is not properly authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user lacks permission to access a resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """

    if not isinstance(freq, Frequency):
        warnings.warn(
            "The use of string values for the 'freq' parameter is deprecated and will be removed in the next major version. Use Frequency enum values instead",
            FutureWarning,
            stacklevel=2,
        )
        freq = Frequency[freq]
    return_data_point = {
        "datapointId": freq.data_point_id,
        "isTsdp": True,
        "startDate": pd.to_datetime(start_date).strftime(FORMAT_DATE),
    }
    if end_date:
        return_data_point["endDate"] = pd.to_datetime(end_date).strftime(FORMAT_DATE)
    if currency:
        return_data_point["currency"] = currency
    return investment.get_investment_data(
        investments=investments,
        data_points=[return_data_point],
        time_series_format=TimeSeriesFormat.LONG_WITHOUT_NAME,
    )


@_decorator.typechecked
def excess_returns(
    investments: Union[List, str, Dict[str, Any]],
    benchmark_sec_id: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    freq: Union[Frequency, str] = Frequency.monthly,
    currency: Optional[str] = None,
) -> pd.DataFrame:
    warnings.warn(
        "The excess_returns function is deprecated and will be removed in the next major version. Use get_excess_returns instead",
        FutureWarning,
        stacklevel=2,
    )
    return get_excess_returns(investments, benchmark_sec_id, start_date, end_date, freq, currency)


@_decorator.typechecked
def get_excess_returns(
    investments: Union[List, str, Dict[str, Any]],
    benchmark_sec_id: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    freq: Union[Frequency, str] = Frequency.monthly,
    currency: Optional[str] = None,
) -> pd.DataFrame:
    """A shortcut function to fetch excess return data for the specified investments.

    Args:
        investments (:obj:`Union`, `required`): Defines the investments to fetch. Input can be:

            * Investment IDs (:obj:`list`, `optional`): Investment identifiers, in the format of SecId;Universe or just SecId. E.g., ["F00000YOOK;FO","FOUSA00CFV;FO"] or ["F00000YOOK","FOUSA00CFV"]. Use the `investments <./lookup.html#morningstar_data.direct.investments>`_ function to discover identifiers.
            * Investment List ID (:obj:`str`, `optional`): Saved investment list in Morningstar Direct. Use the `get_investment_lists <./lists.html#morningstar_data.direct.user_items.get_investment_lists>`_ function to discover saved lists.
            * Search Criteria  ID (:obj:`str`, `optional`): Saved search criteria in Morningstar Direct. Use the `get_search_criteria <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria>`_ function to discover saved search criteria.
            * Search Criteria Condition (:obj:`dict`, `optional`): Search criteria definition. See details in the Reference section of `get_investment_data <#morningstar_data.direct.get_investment_data>`_ or use the `get_search_criteria_conditions <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria_conditions>`_ function to discover the definition of a saved search criteria.

        benchmark_sec_id (:obj:`str`): SecId of the security to use as the benchmark. Use the `investments <./lookup.html#morningstar_data.direct.investments>`_ function to discover identifiers.
        start_date (:obj:`str`): Start date of a date range for retrieving data. The format is
            YYYY-MM-DD, e.g., "2020-01-01".
        end_date (:obj:`str`, `optional`): End date of a date range for retrieving data. If no value is provided for
            end_date, current date will be used. The format is YYYY-MM-DD, e.g., "2020-01-01".
        freq (:obj:`md.direct.data_type.Frequency`): Enumeration of type md.direct.data_type.Frequency, which can be 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'. E.g., ``md.direct.data_type.Frequency.monthly``
        currency (:obj:`str`, `optional`): Three character code for the desired currency of returns, e.g., "USD".  Use the `currency_codes <./lookup.html#morningstar_data.lookup.currency_codes>`_ function to discover possible values.


    :Returns:
        DataFrame: A DataFrame object with excess return data.

    :Examples:
        Get monthly excess returns.

    ::

        import morningstar_data as md

        df = md.direct.get_excess_returns(
            investments=["F00000VKPI", "F000014B1Y"],
            benchmark_sec_id="F00000PLYW",
            freq=md.direct.data_type.Frequency.daily
        )
        df

    :Output:
        ==============  ==========  ====================================
        ID              Date          Monthly Return
        ==============  ==========  ====================================
        F00000VKPI      2020-10-31     -2.121865
        F00000VKPI      2020-11-30     6.337255
        F00000VKPI      2020-12-31     1.464777
        ...                ...         ...
        ==============  ==========  ====================================

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    if not isinstance(freq, Frequency):
        warnings.warn(
            "The use of string values for the 'freq' parameter is deprecated and will be removed in the next major version. Use Frequency enum values instead",
            FutureWarning,
            stacklevel=2,
        )
        freq = Frequency[freq]
        # This is a Direct custom calculation data point and measure of an investment's return in excess of a benchmark.
    return_data_point = {
        "datapointId": "20",
        "sourceId": freq.data_point_id,
        "windowType": CalculationWindowType.ROLLING_WINDOW.value,
        "annualized": False,
        "startDate": pd.to_datetime(start_date).strftime(FORMAT_DATE),
        "frequency": freq.abbr,
        "windowSize": 1,
        "stepSize": 1,
        "benchmark": benchmark_sec_id,
    }

    if currency:
        return_data_point["currency"] = currency
    if end_date:
        end_date = pd.to_datetime(end_date).strftime(FORMAT_DATE)
        return_data_point["endDate"] = end_date
    return investment.get_investment_data(
        investments=investments,
        data_points=[return_data_point],
        time_series_format=TimeSeriesFormat.LONG_WITHOUT_NAME,
    )
