import os
import time
import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import simplejson as json
from pandas import DataFrame

from .._base import _logger
from . import _decorator, _error_messages
from ._config import _Config
from ._data_objects._data_points_object import DataPoints
from ._data_type import ErrorMessages
from ._exceptions import BadRequestException, ResourceNotFoundError, ValueErrorException
from ._investment._normal_data import _parse_raw_normal_data_values
from ._portfolio_data_set import (
    DataSetType,
    PortfolioDataSet,
    _get_data_points_default_settings,
    _get_data_set_with_id,
    _get_data_sets,
)
from ._utils import _get_iso_today
from .user_items.portfolio import PortfolioDataApiBackend, get_portfolios

_config = _Config()
_portfolio_api = PortfolioDataApiBackend()


@_decorator.error_handler
@_decorator.typechecked
def get_data_sets() -> DataFrame:
    """Returns all Morningstar-created portfolio data sets.

    :Returns:
        DataFrame: A DataFrame object with data sets. DataFrame columns include:

        * data_set_id
        * name

    :Examples:
        Get all portfolio data sets.

    ::

        import morningstar_data as md

        df = md.direct.portfolio.get_data_sets()
        df

    :Output:
        ===========    ===============
        data_set_id	   name
        ===========    ===============
        1              Snapshot
        2              Returns (Daily)
        ===========    ===============

    """
    all_view_df = pd.DataFrame({"data_set_id": [], "name": []})
    df = _get_data_sets()
    df = df.sort_values(by=["id"])
    all_view_df["data_set_id"] = df["id"].astype(str)
    all_view_df["name"] = df["name"]
    all_view_df = all_view_df.reset_index(drop=True)
    return all_view_df


@_decorator.typechecked
def get_data_set_data_points(data_set_id: str) -> DataFrame:
    """Returns all data points for the given portfolio data set ID.

    Args:
        data_set_id (:obj:`str`): The unique identifier of a portfolio data set, e.g., '1'.

    :Returns:
        DataFrame: A DataFrame object with data points. DataFrame columns include:

        * data_point_id
        * name

    :Examples:
        Get data points by data set ID.

    ::

        import morningstar_data as md

        df = md.direct.portfolio.get_data_set_data_points(data_set_id="1")
        df

    :Output:

        ==============   ===============
        data_point_id    name
        ==============   ===============
        OS01W		     Name
        LS05M		     Base Currency
        ...              ...
        ==============   ===============

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    view = _get_data_set_with_id(data_set_id)
    data_points = pd.DataFrame(view["datapoints"])
    data_points = data_points[["datapointId", "alias"]]
    data_points = data_points.rename(columns={"alias": "name"})
    data_points = data_points.rename(columns={"datapointId": "data_point_id"})
    return data_points


def _get_data_points_by_view_id(id: str, currency: Optional[str]) -> List[Dict[Any, Any]]:
    view = _get_data_set_with_id(id)
    return _set_currency(view["datapoints"], currency)


def _set_currency(data_points: List[Dict], currency: Optional[str]) -> List[Dict[Any, Any]]:
    if currency is not None:
        for data_point in data_points:
            if data_point.get("currency") is not None:
                data_point["currency"] = currency
    return data_points


@_decorator.not_null
@_decorator.typechecked
def get(portfolio_type: Optional[str] = None) -> DataFrame:
    warnings.warn(
        "The portfolio.get function is deprecated and will be removed in the next major version. Use user_items.get_portfolios instead",
        FutureWarning,
        stacklevel=2,
    )
    return get_portfolios(portfolio_type=portfolio_type)


@_decorator.typechecked
def get_holdings(
    portfolio_ids: List[str],
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> DataFrame:
    """Returns holdings for the given portfolio and date range.

    Args:
        portfolio_ids (:obj:`list`): List of portfolio IDs. The maximum number is 5. Use the `get_portfolios <./portfolio_list.html#morningstar_data.direct.user_items.get_portfolios>`_ function to discover saved portfolios.
        date (:obj:`str`, `optional`): Holdings date. When this value is provided, `start_date` and `end_date` parameters are ignored. Use the `get_holding_dates <#morningstar_data.direct.portfolio.get_holding_dates>`_ function to discover available holding dates.
        start_date (:obj:`str`, `optional`): Start date of the date range.
        end_date (:obj:`str`, `optional`): End date of the date range.

    :Returns:
        DataFrame: A DataFrame object with portfolio holdings data. DataFrame columns include:

        * ObjectId
        * Portfolio Date
        * HoldingId
        * Weight
        * Name
        * Currency
        * ISIN
        * Ticker

    :Examples:
        Get holdings by portfolio ID and date.

    ::

        import morningstar_data as md

        df = md.direct.portfolio.get_holdings(
            portfolio_ids=['de21828b-91b7-4fe1-b66e-760fd5e657bc;BM'], # Replace with valid portfolio ID
            date = "2017-09-30")
        df

    :Output:
        =======================================   ==============   ==============   =======   =================================   ============   ============   ==========
        ObjectId                                  Portfolio Date   HoldingId	    Weight	  Name                                Currency       ISIN           Ticker
        =======================================   ==============   ==============   =======   =================================   ============   ============   ==========
        de21828b-91b7-4fe1-b66e-760fd5e657bc;BM   2017-09-30	   FOUSA00DFS;F     100	      BlackRock Global Allocation Inv A	  USD	         US09251T1034   MDLOX
        ...
        =======================================   ==============   ==============   =======   =================================   ============   ============   ==========

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    if not portfolio_ids:
        raise BadRequestException("Please input portfolio_ids to proceed with your query.")
    combined_holdings = pd.DataFrame(
        {
            "ObjectId": [],
            "Portfolio Date": [],
            "HoldingId": [],
            "Weight": [],
            "Name": [],
            "Currency": [],
            "ISIN": [],
            "Ticker": [],
        }
    )
    if portfolio_ids:
        if len(portfolio_ids) > 5:
            raise BadRequestException("The number of portfolio_ids per query are limited to 5.")
        portfolio_available_dates = _get_dates(portfolio_ids, date, start_date, end_date)
        if not portfolio_available_dates and date is not None and len(date.strip()) > 0:
            portfolio_available_dates = {x: [date] for x in portfolio_ids}
        _check_volume(portfolio_available_dates)
        holdings_df = _get_holdings(portfolio_available_dates)
        combined_holdings = pd.concat([combined_holdings, holdings_df])

    if combined_holdings.empty:
        if len(portfolio_ids) == 1:
            raise ResourceNotFoundError(
                (_error_messages.RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_HOLDING).format(portfolio_ids)
            ) from None
        raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_HOLDINGS) from None
    else:
        combined_holdings = combined_holdings.where(combined_holdings.notnull(), None)
        combined_holdings.reset_index(drop=True, inplace=True)
    return combined_holdings


def _get_holdings(portfolio_available_dates: dict) -> DataFrame:
    holdings_list = []
    for portfolio_id, dates in portfolio_available_dates.items():
        for date in dates:
            holdings = _get_holdings_by_id(portfolio_id, date)
            holdings = holdings if holdings else []
            holdings_list.extend(
                [
                    {
                        "ObjectId": portfolio_id,
                        "Portfolio Date": date,
                        "HoldingId": x.get("id", None),
                        "Weight": x.get("weight", None),
                        "Name": x.get("name", None),
                        "Currency": x.get("currency", None),
                        "ISIN": x.get("isin", None),
                        "Ticker": x.get("ticker", None),
                    }
                    for x in holdings
                ]
            )
    return DataFrame(holdings_list)


def _check_volume(portfolio_available_dates: dict) -> None:
    if portfolio_available_dates:
        total = 0
        for dates in portfolio_available_dates.values():
            total += len(dates) if dates is not None and isinstance(dates, list) else 0
        if total > 10:
            raise ValueErrorException(
                "Please reduce the number of investments or shorten the query time period to proceed your query without interfering by too large data size."
            )


def _get_holdings_by_id(portfolio_id: str, date: str) -> list:
    try:
        url = f"{_config.portfolio_service_url()}/portfoliodataservice/v1/portfolios/{portfolio_id}/holdings-summary?portfolioDate={date}"
        response_json: list = _portfolio_api.do_get_request(url)
        return response_json
    except Exception as portfolio_error:
        if portfolio_error.__class__.__name__ != "ResourceNotFoundError":
            raise portfolio_error from None
        pass
    return []


def _get_dates(
    portfolio_ids: list,
    date: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> dict:
    if date is not None and len(date.strip()) > 0:
        date = _format_date(date)
    elif start_date is not None and len(start_date.strip()) > 0:
        start_date = _format_date(start_date)
        if end_date is None or len(end_date.strip()) == 0:
            end_date = _get_iso_today()
        end_date = _format_date(end_date)
    elif end_date is not None and len(end_date.strip()) > 0:
        raise BadRequestException("Please specify a portfolio_date or time period to proceed with your query.")

    response_json = _get_multi_portfolio_dates(portfolio_ids)

    if date is not None and len(date.strip()) > 0:
        return _handle_single_date(response_json, date)
    elif start_date is not None and len(start_date.strip()) > 0 and end_date is not None and len(end_date.strip()) > 0:
        return _handle_date_range(response_json, start_date, end_date)
    else:
        return _handle_no_date(response_json)


def _get_multi_portfolio_dates(portfolio_ids: list) -> dict:
    portfolio_date_dict = dict()
    for id in portfolio_ids:
        if id:
            portfolio_date_dict[id] = _get_holding_dates(id)
    return portfolio_date_dict


def _handle_single_date(portfolio_dates: dict, input_portfolio_date: str) -> dict:
    portfolio_available_dates = dict()
    for portfolio_id, dates in portfolio_dates.items():
        available_date = _get_available_date(input_portfolio_date, dates)
        if available_date is not None:
            portfolio_available_dates[portfolio_id] = [available_date]
    return portfolio_available_dates


def _get_available_date(target_date: str, dates: list) -> Optional[str]:
    dates = dates if dates else []
    if target_date in dates:
        return target_date

    available_date = None
    dates.sort()
    dates.reverse()
    for date in dates:
        if date < target_date:
            available_date = date
            break
    return available_date


def _handle_no_date(portfolio_dates: dict) -> dict:
    portfolio_available_dates = dict()
    for portfolio_id, dates in portfolio_dates.items():
        dates = dates if dates else []
        date = next((x for x in dates if x is not None and len(x.strip()) > 0), None)
        if date is not None and len(date) > 0:
            portfolio_available_dates[portfolio_id] = [date]
    return portfolio_available_dates


def _handle_date_range(portfolio_dates: dict, start_date: str, end_date: str) -> dict:
    portfolio_available_dates = dict()
    for portfolio_id, dates in portfolio_dates.items():
        available_dates = list()
        dates = dates if dates else []
        for date in dates:
            if start_date <= date <= end_date:
                available_dates.append(date)
        if available_dates:
            portfolio_available_dates[portfolio_id] = available_dates

    return portfolio_available_dates


@_decorator.typechecked
def get_holding_dates(portfolio_ids: Optional[Union[List[str], str]] = None) -> DataFrame:
    """Returns portfolio holdings dates for the given portfolios.


    Args:
        portfolio_ids (:obj:`list`, `optional`): A portfolio ID or list of portfolio IDs. If no ID is specified, the function will return data for all portfolios saved in the user's account. Use the `get_portfolios <./portfolio_list.html#morningstar_data.direct.user_items.get_portfolios>`_ function to discover saved portfolios.

    :Returns:
        DataFrame: A DataFrame object with all portfolio holdings dates. DataFrame columns include:

        * portfolioId
        * date

    :Examples:
        Get portfolio holdings dates for the given portfolio ID.

    ::

        import morningstar_data as md

        df = md.direct.portfolio.get_holding_dates(
            portfolio_ids=['07c317df-a4a7-4297-afc3-0c18bb79a672;UA']) # Replace with a valid portfolio ID
        df

    :Output:
        =======================================        ==========
        PortfolioId                                    Date
        =======================================        ==========
        07c317df-a4a7-4297-afc3-0c18bb79a672;UA        2021-10-01
        07c317df-a4a7-4297-afc3-0c18bb79a672;UA        2021-09-01
        ...
        =======================================        ==========

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    if not isinstance(portfolio_ids, list) and portfolio_ids is not None:
        single_id = portfolio_ids
        portfolio_ids = list()
        portfolio_ids.append(single_id)
    portfolio_date_list = list()
    if not portfolio_ids:
        portfolio_df = get()
        portfolio_ids = list(portfolio_df["PortfolioId"])
    for portfolio_id in sorted(set(portfolio_ids), key=portfolio_ids.index):
        if portfolio_id is None or len(portfolio_id.strip()) == 0:
            continue
        dates = _get_holding_dates(portfolio_id, portfolio_ids)
        if not dates:
            continue
        portfolio_date_list.extend([{"PortfolioId": portfolio_id, "Date": date} for date in dates])
    portfolio_df = DataFrame(portfolio_date_list)
    if portfolio_df.empty:
        if len(portfolio_ids) == 1:
            raise ResourceNotFoundError((_error_messages.RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_ID).format(portfolio_id)) from None
        raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_IDS) from None
    return portfolio_df


def _get_holding_dates(portfolio_id: str, portfolio_ids: Optional[Union[List[str], str]] = None) -> list:
    try:
        url = f"{_config.portfolio_service_url()}/portfoliodataservice/v1/portfolios/{portfolio_id}/dates"
        response_json: Dict[str, list] = _portfolio_api.do_get_request(url)
        return response_json["dates"]
    except Exception as portfolio_error:
        if portfolio_error.__class__.__name__ != "ResourceNotFoundError":
            raise portfolio_error from None
        pass
    return []


def _format_date(date: str) -> str:
    try:
        return time.strftime("%Y-%m-%d", time.strptime(date, "%Y-%m-%d"))
    except Exception as e:
        # _logger.error(f"Date format error: {e}")
        raise BadRequestException(ErrorMessages.date_format_error.value + f" Date format error: {e}.") from None


@_decorator.typechecked
def get_data(
    portfolio_id: str,
    data_set_id: Optional[Union[str, PortfolioDataSet, None]] = None,
    currency: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_point_settings: Optional[DataFrame] = None,
) -> DataFrame:
    """Returns data for the given portfolio and data points.

    Args:
        portfolio_id (:obj:`str` ): Portfolio ID. Use the `get_portfolios <./portfolio_list.html#morningstar_data.direct.user_items.get_portfolios>`_ function to discover saved portfolios.
        data_set_id (:obj:`str`, `optional`): Saved portfolio data set, e.g., "1". Use the `get_data_sets <./data_set.html#morningstar_data.direct.portfolio.get_data_sets>`_ function to discover available data sets.
        currency (:obj:`str`, `optional`): Currency setting for the data point values, when applicable. Use the `currency_codes <../morningstar_data/lookup.html#morningstar_data.lookup.currency_codes>`_ function to discover available currency codes.
        start_date (:obj:`str`, `optional`): Start date of a date range for retrieving data.
            The format is YYYY-MM-DD, e.g., '2020-01-01'.
        end_date (:obj:`str`, `optional`): End date of a date range for retrieving data. If no value is provided for `end_date`, current date will be used.
            The format is YYYY-MM-DD, e.g., '2020-01-01'.
        data_point_settings (:obj:`DataFrame`, `optional`): A DataFrame of data points with all defined settings. Each row represents a data point.
            Each column is a configurable setting. Users can get this DataFrame by using `get_data_set_data_points <./data_set.html#morningstar_data.direct.portfolio.get_data_set_data_points>`_ or `get_data_point_settings <../morningstar_data/lookup.html#morningstar_data.direct.get_data_point_settings>`_ for a given data point ID(s).
            Users can update setting values in this DataFrame if desired. Additionally, the priority of currency/start_date/end_date in `data_point_settings`
            is lower than the currency/start_date/end_date function parameters.


    :Returns:
        DataFrame: A DataFrame object with portfolio data.

    :Examples:
        Get portfolio data for the given portfolio and data set.

    ::

        import morningstar_data as md

        df = md.direct.portfolio.get_data(
            portfolio_id="8c6de08f-a668-4e78-a688-3d809aeb29b7;UA", # Replace with a valid portfolio ID
            data_set_id="1")
        df

    :Output:
        =========================================  =====   ==============   ================   =====   =================================   ==================================
        Id                                         Name    Base Currency    Portfolio Date     ...     Total Ret Annlzd 10 Yr (Year-End)   Total Ret Annlzd 15 Yr (Year-End)
        =========================================  =====   ==============   ================   =====   =================================   ==================================
        8c6de08f-a668-4e78-a688-3d809aeb29b7;UA    test1   US Dollar        2022-02-28         ...     None                                None
        ...
        =========================================  =====   ==============   ================   =====   =================================   ==================================

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    all_data = _get_portfolio_data(
        portfolio_id=portfolio_id,
        data_set_id=data_set_id,
        currency=currency,
        start_date=start_date,
        end_date=end_date,
        data_point_settings=data_point_settings,
    )
    col_name = all_data.columns.tolist()
    all_data = all_data.reindex(columns=col_name)
    return all_data


def _get_portfolio_data(
    portfolio_id: str,
    data_set_id: Optional[Union[str, PortfolioDataSet, None]] = None,
    currency: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_point_settings: Optional[DataFrame] = None,
) -> DataFrame:
    """
    As the "get_portfolio_data" API may been called by other APIs, so we extracted it as a private method.
    """
    if data_set_id is not None and data_point_settings is not None and not data_point_settings.empty:
        raise BadRequestException("Please specify either data_set_id or data_point_settings but not both.")
    # try:
    # build columns
    settings_data_frame = None
    if data_point_settings is not None and not data_point_settings.empty:
        settings_data_frame = _get_data_settings_data_points(data_point_settings, currency, start_date, end_date)
    else:
        if isinstance(data_set_id, PortfolioDataSet):
            data_set_id = data_set_id.data_set_id
            if get_data_set_data_points(data_set_id=data_set_id).empty:
                raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_PORTFOLIO_DATA_SET) from None
        if not data_set_id:
            raise BadRequestException("Please specify either data_set_id or data_point_settings to proceed with your query.")
        settings_data_frame = DataFrame(_get_data_set_data_points(data_set_id, currency, start_date, end_date))
    portfolio_ids = [portfolio_id]

    settings_data_frame = settings_data_frame.where(settings_data_frame.notnull(), None)

    # get data
    raw_data = _get_normal_data(
        portfolio_ids,
        _filter_disable_data_points(settings_data_frame).to_dict(orient="records"),
    )

    investment_results = _parse_raw_normal_data_values(raw_data)

    # add id
    all_data = investment_results.as_data_frame()
    col_name = all_data.columns.tolist()
    all_data = all_data.reindex(columns=col_name)

    return all_data


def _filter_disable_data_points(data_point_settings: DataFrame = None) -> DataFrame:
    columns = data_point_settings.columns
    if "disable" in columns:
        return data_point_settings.loc[data_point_settings["disable"] is not True]
    return data_point_settings


def _get_normal_data(portfolio_ids: list, data_points: list) -> DataFrame:
    if not portfolio_ids:
        raise BadRequestException("There is no portfolio data returned.")
    postbody = {
        "datapoints": data_points,
        "investments": list(map(lambda x: {"id": x}, portfolio_ids)),
    }
    resp: DataFrame = _request_porfolio_data(postbody, portfolio_ids)
    if resp is None:
        raise BadRequestException("There is no return data available.")
    return resp


def _request_porfolio_data(postbody: dict, portfolio_ids: list) -> DataFrame:
    job_detail, portfolio_data = _get_portfolio_job(postbody)
    job_id = job_detail["id"]
    try_times = int(os.environ.get("MD_PORTFOLIO_DATA_TIMEOUT_SECONDS", 600)) / 2
    while True:
        job_detail, portfolio_data = _get_portfolio_job(postbody, job_id)
        time.sleep(2)
        try_times = try_times - 1
        if job_detail.get("percentage", "0") == "100" or try_times == 0:
            break
    if not portfolio_data:
        raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_GET_DATA_RETURN.format(portfolio_ids[0]))
    return portfolio_data


def _get_portfolio_job(postbody: dict, job_id: Optional[str] = None) -> tuple:
    url = f"{_config.portfolio_service_url()}/portfoliodataservice/v1/portfolios/calculation?maxExecTime=600000&maxWaitTime=600000&timeOut=600000"
    if job_id:
        url = url + f"&jobId={job_id}"
    response_json = _portfolio_api.do_post_request(url, json.dumps(postbody, ignore_nan=True).replace("NaN", "null"))
    job_detail = response_json["job"]
    portfolio_data = None

    if job_detail.get("percentage", "0") == "100":
        portfolio_data = response_json
    return job_detail, portfolio_data


def _get_data_settings_data_points(
    data_point_settings: DataFrame,
    currency: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> DataFrame:
    if "datapointId" not in data_point_settings.columns:
        raise BadRequestException("Please input valid data_point_settings to proceed with your query.")
    data_point_settings = data_point_settings.where(data_point_settings.notnull(), None)
    start_date, end_date = _format_start_end_date(start_date, end_date)
    # only set currency to the data_points with currency.
    # only set start_date/end_date to the datapoints without isTsdp=False.

    data_point_default_settings = _get_data_points_default_settings(data_point_settings["datapointId"].tolist())

    settings_list = data_point_settings.to_dict(orient="records")
    for settings in settings_list:
        data_point_id = settings["datapointId"]
        defined_data_point = data_point_default_settings.get(data_point_id)
        if defined_data_point is not None and defined_data_point.get("currency") is None:
            settings["currency"] = None
        elif currency is not None and len(currency.strip()) > 0:
            settings["currency"] = currency
        if defined_data_point is not None and defined_data_point.get("isTsdp") is not None and settings.get("isTsdp") is None:
            settings["isTsdp"] = defined_data_point.get("isTsdp")

        if (
            settings.get("isTsdp") is not False
            and start_date is not None
            and len(start_date.strip()) > 0
            and end_date is not None
            and len(end_date.strip()) > 0
        ):
            settings["startDate"] = start_date
            settings["endDate"] = end_date

        esg_data_point = data_point_default_settings.get(data_point_id)
        if data_point_id != "OS01W" and esg_data_point is not None and settings.get("isTsdp") is None:
            if esg_data_point.get("isTsdp") is not None:
                settings["isTsdp"] = esg_data_point.get("isTsdp")
            else:
                start = settings.get("startDate")
                end = settings.get("endDate")
                if start is not None and len(start.strip()) > 0 and end is not None and len(end.strip()) > 0:
                    settings["isTsdp"] = True
                else:
                    settings["isTsdp"] = False
    data_point_settings = DataFrame(settings_list)
    data_point_settings = data_point_settings.where(data_point_settings.notnull(), None)
    data_points = DataPoints(data_point_settings)
    settings_data_frame = DataFrame(data_points.get_data_points().to_dict(orient="records"))
    settings_data_frame = _disable_data_points(settings_data_frame, data_point_default_settings)

    return settings_data_frame


def _disable_data_points(settings_data_frame: DataFrame, defined_data_points: dict) -> DataFrame:
    settings_list = settings_data_frame.to_dict(orient="records")
    for settings in settings_list:
        data_point_id = settings["datapointId"]
        frequency_m_data_point = defined_data_points.get(data_point_id)
        if frequency_m_data_point is not None:
            frequency = settings.get("frequency")
            defined_freq = frequency_m_data_point.get("frequency", None)
            if defined_freq is not None and frequency is not None and len(frequency.strip()) > 0 and frequency != defined_freq:
                settings["disable"] = True

        defined_data_point = defined_data_points.get(data_point_id)
        # for defined datapoint with isTsdp = False
        if defined_data_point is not None:
            if defined_data_point.get("isTsdp") is False and settings.get("isTsdp") is True:
                settings["disable"] = True
            if defined_data_point.get("isTsdp") is True and settings.get("isTsdp") is False:
                settings["disable"] = True
    settings_data_frame = DataFrame(settings_list)
    settings_data_frame = settings_data_frame.where(settings_data_frame.notnull(), None)
    return settings_data_frame


def _get_data_set_data_points(
    data_set_id: str,
    currency: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> list:
    data_point_ids = _get_data_points_by_view_id(data_set_id, currency)
    data_points = DataPoints(data_point_ids)
    if not data_point_ids:
        return list(data_points.get_data_points().to_dict(orient="records"))
    data_set_type = PortfolioDataSet.get_data_set_type(data_set_id)
    if data_set_type == DataSetType.TimeSeries:
        if start_date is None or len(start_date.strip()) == 0:
            raise BadRequestException(_error_messages.BAD_REQUEST_ERROR_NO_START_AND_END_DATE) from None
        if end_date is None or len(end_date.strip()) == 0:
            raise BadRequestException(_error_messages.BAD_REQUEST_ERROR_NO_END_DATE)
        for data_point_id in data_point_ids:
            if data_point_id.get("isTsdp") is True:
                data_point_id["startDate"] = start_date
                data_point_id["endDate"] = end_date

    if data_set_type == DataSetType.Custom_Calculation:
        start_date, end_date = _format_start_end_date(start_date, end_date)
        if start_date is not None and len(start_date.strip()) > 0 and end_date is not None and len(end_date.strip()) > 0:
            for data_point_id in data_point_ids:
                data_point_id["startDate"] = start_date
                data_point_id["endDate"] = end_date
        return list(data_points.get_data_points().to_dict(orient="records"))

    if data_set_type == DataSetType.Current_Or_TimeSeries:
        start_date, end_date = _format_start_end_date(start_date, end_date)
        for data_point_id in data_point_ids:
            if (
                data_point_id.get("isTsdp") is not False
                and start_date is not None
                and len(start_date.strip()) > 0
                and end_date is not None
                and len(end_date.strip()) > 0
            ):
                data_point_id["startDate"] = start_date
                data_point_id["endDate"] = end_date
                data_point_id["isTsdp"] = True
                # support 'm' only
                data_point_id["frequency"] = "m"
            else:
                data_point_id["isTsdp"] = False
        return list(data_points.get_data_points().to_dict(orient="records"))

    return data_point_ids


def _format_start_end_date(start_date: Optional[str], end_date: Optional[str]) -> tuple:
    if start_date is not None and len(start_date.strip()) > 0:
        start_date = _format_date(start_date)
        if end_date is None or len(end_date.strip()) == 0:
            raise BadRequestException(ErrorMessages.start_end_date_error.value)
        end_date = _format_date(end_date)
    elif end_date is not None and len(end_date.strip()) > 0:
        raise BadRequestException(ErrorMessages.start_end_date_error.value)
    return start_date, end_date


@_decorator.typechecked
def get_benchmark_settings(portfolio_id: str) -> DataFrame:
    """
    Returns a portfolio's benchmark settings.

    Args:
        portfolio_id (:obj:`str` ): Portfolio ID. Use the `get_portfolios <./portfolio_list.html#morningstar_data.direct.user_items.get_portfolios>`_ function to discover saved portfolios.

    :Returns:
        DataFrame: A DataFrame object containing the portfolio settings.

    :Examples:

    Get portfolio settings for a custom model portfolio.
    ::
        import morningstar_data as md

        df = md.direct.portfolio.get_settings(
            portfolio_id="d38b96dd-fbe1-4750-b05d-fa7cf4b6e35a;MD") # Replace with a valid portfolio ID
        df

    :Output:

        ========================================  ====================================  ====================================  ================  ======================  ==============  ================  ======================  =====================
        Portfolio ID                              Risk Free Proxy ID                    Benchmark 1 ID                        Benchmark 1 Name  Benchmark 1 SecurityId  Benchmark 2 ID  Benchmark 2 Name  Benchmark 2 SecurityId  Internal Portfolio ID
        ========================================  ====================================  ====================================  ================  ======================  ==============  ================  ======================  =====================
        d38b96dd-fbe1-4750-b05d-fa7cf4b6e35a;MD   e9b23e21-9c6f-4696-9583-3b3729b4ad99  fcff80f8-2fbc-4211-bbce-45252008a9cc  S&P 500 TR USD    XIUSA04G92;XI           None            None              None                    None

        ========================================  ====================================  ====================================  ================  ======================  ==============  ================  ======================  =====================

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """

    _logger.info("Fetching portfolio settings by id")
    portfolio_settings = _get_settings_by_id(portfolio_id=portfolio_id)
    settings = {}
    settings["Porfolio ID"] = portfolio_settings["portfolio"]["portfolioId"]
    settings["Internal Portfolio ID"] = portfolio_settings["portfolio"]["portfolio2Id"]
    settings["Risk Free Proxy ID"] = portfolio_settings["portfolio"]["riskFreeProxyId"]

    primary_benchmark = portfolio_settings["portfolio"]["benchmarkId"]
    secondary_benchmark = portfolio_settings["portfolio"]["secondaryBenchmarkId"]
    # The secondary benchmark is optional so not all portfolios will have this information
    secondary_benchmark_name = None
    secondary_benchmark_id = None

    for benchmark in portfolio_settings["benchmarks"]:
        benchmark_id = benchmark["benchmarkId"]
        if benchmark_id == primary_benchmark:
            primary_benchmark_name = benchmark["name"]
            primary_benchmark_sec_id = benchmark["securityId"]

        elif benchmark_id == secondary_benchmark:
            secondary_benchmark_name = benchmark["name"]
            secondary_benchmark_id = benchmark["securityId"]

    settings["Benchmark 1 ID"] = primary_benchmark
    settings["Benchmark 1 Name"] = primary_benchmark_name
    settings["Benchmark 1 Security ID"] = primary_benchmark_sec_id

    settings["Benchmark 2 ID"] = secondary_benchmark
    settings["Benchmark 2 Name"] = secondary_benchmark_name
    settings["Benchmark 2 Security ID"] = secondary_benchmark_id

    return DataFrame(settings, index=[0])


def _get_settings_by_id(portfolio_id: str) -> dict:
    try:
        url = f"{_config.portfolio_service_url()}/portfoliodataservice/v1/portfolios/{portfolio_id}/settings"
        response_json = _portfolio_api.do_get_request(url)
        settings: dict = response_json["settings"]  # settings key is always present when 200 status code is returned.
        return settings
    except Exception as portfolio_error:
        if portfolio_error.__class__.__name__ != "ResourceNotFoundError":
            raise portfolio_error from None
        pass
    return {}
