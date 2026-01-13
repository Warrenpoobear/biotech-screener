import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pandas import DataFrame

from .. import mdapi
from .._base import _logger
from ..direct.data_type import InvestmentIdentifier
from ..mdapi import RequestObject
from ..mdapi._exceptions import MdApiTaskException
from ..mdapi._mdapi import (
    AsyncHoldingsRequest,
    _wait_for_completed_task_response,
    get_holding_dry_run_results,
    init_holdings_async,
    init_lookthrough_holdings_async,
)
from ..mdapi._types import MdapiTask, TaskResult
from . import _decorator
from ._config import _Config
from ._data_type import DryRunResults
from ._error_messages import (
    BAD_REQUEST_ERROR_NO_INVESTMENT_IDS,
)
from ._exceptions import BadRequestException
from .data_type import (
    EnrichmentType,
    Frequency,
    HoldingsView,
    PortfolioDepth,
)

_config = _Config()


MASTER_PORTFOLIO_ID = "MasterPortfolio Id"


@dataclass
class HoldingsRequest(RequestObject):
    investments: Union[List[str], str, Dict[str, Any]]
    date: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]


@dataclass
class HoldingDateRequest(RequestObject):
    investments: List[str]


@_decorator.typechecked
def holdings(
    investment_ids: List[str],
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> DataFrame:
    warnings.warn(
        "The holdings function is deprecated and will be removed in the next major version. Use get_holdings instead",
        FutureWarning,
        stacklevel=2,
    )
    return get_holdings(investment_ids, date, start_date, end_date)


@_decorator.typechecked
def get_holdings(
    investments: Optional[Union[List[str], str, Dict[str, Any], List[InvestmentIdentifier]]] = None,
    date: Optional[Union[str, List[str]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    investment_ids: Optional[List[str]] = None,
    data_points: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None,
    frequency: Frequency = Frequency.monthly,
    holdings_view: Union[int, HoldingsView] = HoldingsView.FULL,
    suppression_client_id: Optional[str] = None,
    enrichment_type: EnrichmentType = EnrichmentType.POINT_IN_TIME,
    delta_start_time: Optional[str] = None,
    delta_include_unchanged_data: Optional[bool] = None,
    dry_run: Optional[bool] = False,
    preview: Optional[bool] = False,
) -> Union[DataFrame, DryRunResults]:
    """Returns base level holdings details for the specified investments. Note that if none of the date parameters are provided, the function uses the latest portfolio date by default.

    Args:
        investments (:obj:`Union`, `required`): Defines the investments to fetch. Input can be:

                * Investment IDs (:obj:`list`, `optional`): Investment identifiers, in the format of SecId;Universe or just SecId. E.g., ["F00000YOOK;FO","FOUSA00CFV;FO"] or ["F00000YOOK","FOUSA00CFV"]. Use the `investments <./lookup.html#morningstar_data.direct.investments>`_ function to discover identifiers.
                * Investment List ID (:obj:`str`, `optional`): Saved investment list in Morningstar Direct. Currently, this function does not support lists that combine investments and user-created portfolios. Use the `get_investment_lists <./lists.html#morningstar_data.direct.user_items.get_investment_lists>`_ function to discover saved lists.
                * Search Criteria  ID (:obj:`str`, `optional`): Saved search criteria in Morningstar Direct. Use the `get_search_criteria <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria>`_ function to discover saved search criteria.
                * Search Criteria Condition (:obj:`dict`, `optional`): Search criteria definition. See details in the Reference section below or use the `get_search_criteria_conditions <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria_conditions>`_ function to discover the definition of a saved search criteria.
                * InvestmentIdentifiers (:obj:`list`, `optional`): A list of :class:`~morningstar_data.direct.InvestmentIdentifier` objects, allowing users to specify investments using standard identifiers such as ISIN, CUSIP, and/or ticker symbol instead of Morningstar SecIds.

                    * Multiple Matches: If multiple valid matches exist for a single InvestmentIdentifier, results will be prioritized according to the :ref:`security matching logic<Security Matching Logic>`. The highest ranked security will be used.
                    * Request Limit: Supports up to 500 InvestmentIdentifier objects per request.

        date (:obj:`str, list[str]`, `optional`): The portfolio date for which to retrieve data. The format is YYYY-MM-DD.
            For example, "2020-01-01". If a date is provided, then the `start_date` and `end_date` parameters are ignored.
            Can also be a list of dates. The list can contain a maximum of 20 dates.
            An exception is thrown if `start_date` or `end_date` is provided along with `date`.
        start_date (:obj:`str`, `optional`): The start date for retrieving data. The format is
            YYYY-MM-DD. For example, "2020-01-01". An exception is thrown if `date` is provided along with `start_date`.
        end_date (:obj:`str`, `optional`): The end date for retrieving data. If no value is provided for
            `end_date`, current date will be used. The format is YYYY-MM-DD. For example, "2020-01-01". An exception is
            thrown if `date` is provided along with `end_date`.
        investment_ids (:obj:`list`): DEPRECATED, A list of investment IDs. The investment ID format is SecId;Universe or just SecId.
            For example: ["F00000YOOK;FO","FOUSA00CFV;FO"] or ["F00000YOOK","FOUSA00CFV"].
        data_points (:obj:`Union`, `optional`): Defines the data points to fetch. If no data points are provided, a default list of data points with predefined aliases are returned, below are the list of default data points and associated aliases.

                +---------------------+---------------+------------------------------+
                | Alias               | Data Point ID | Description                  |
                +=====================+===============+==============================+
                | holdingStorageId    | IHS01         | Holding Storage Id           |
                +---------------------+---------------+------------------------------+
                | secId               | OS00I         | SecId                        |
                +---------------------+---------------+------------------------------+
                | name                | HS765         | Security Name                |
                +---------------------+---------------+------------------------------+
                | defaultHoldingTypeCode  | IHS10         | Detail Holding Type Code     |
                +---------------------+---------------+------------------------------+
                | isin                | HS761         | ISIN                         |
                +---------------------+---------------+------------------------------+
                | cusip               | HS762         | CUSIP                        |
                +---------------------+---------------+------------------------------+
                | weight              | HS750         | Portfolio Weighting Percent  |
                +---------------------+---------------+------------------------------+
                | shares              | HS752         | Number of Shares             |
                +---------------------+---------------+------------------------------+
                | marketValue         | HS751         | Position Market Value        |
                +---------------------+---------------+------------------------------+
                | sharesChanged       | HS753         | Share Change                 |
                +---------------------+---------------+------------------------------+
                | ticker              | OS385         | Ticker                       |
                +---------------------+---------------+------------------------------+
                | currency            | HS763         | Currency                     |
                +---------------------+---------------+------------------------------+
                | detailHoldingType   | HS766         | Detail Holding Type          |
                +---------------------+---------------+------------------------------+

                * Data Point IDs (:obj:`List[Dict]`, `optional`): A list of dictionaries, each defining a data point and its (optional) associated settings. If an alias is provided in the data point setting, it will be used as the column name in the returned DataFrame. Otherwise, the data point ID will be used as the column name.
                * Data Point Settings (:obj:`DataFrame`, `optional`): A DataFrame of data point identifiers and their associated settings. Use the `get_data_set_details <./data_set.html#morningstar_data.direct.user_items.get_data_set_details>`_ function to discover data point settings from a saved data set.

        frequency (:obj:`md.direct.data_type.Frequency`, `optional`): Frequency at which data point values are retrieved. The default is `Frequency.monthly`.

                * daily: Portfolio holdings data reported daily.
                * monthly: Portfolio holdings data reported on or close to the month-end calendar dates.

        holdings_view (:obj:`Union`, `optional`): Specifies the level of detail returned for portfolio holdings. This can be an integer or `md.direct.data_type.HoldingsView`. The default is `HoldingsView.FULL`.

                * Integer Value N: Returns only top N holdings for each portfolio.
                * FULL: Returns all available holdings for the portfolio on requested dates.
                * BEST_AVAILABLE: Returns the most complete holdings data available for each requested date.

        suppression_client_id (:obj:`str`, `optional`): If you have a privileged access to suppressed portfolios and would like to apply it, provide your Client ID.

        enrichment_type (:obj:`md.direct.data_type.EnrichmentType`, `optional`): Specifies the enrichment type of the data returned. The default is `EnrichmentType.POINT_IN_TIME`.

                * LATEST_AVAILABLE: Uses the most current data at feed generation time.
                * POINT_IN_TIME: Matches the data to each portfolio’s effective date.
                * LATEST_MONTH_END: Uses the latest available month-end data at request time.

        dry_run (:obj:`bool`, `optional`): When True, the query will not be executed. Instead, a DryRunResults object will be returned with details about the query's impact on daily cell limit usage.
        preview (:obj:`bool`, `optional`): Defaults to False. Setting to True allows access to data points outside of your current subscription, but limits the output to 25 rows.

    :Returns:

        There are two return types:

        DataFrame: A DataFrame object with holdings data. DataFrame columns include:

        * masterPortfolioId
        * investmentId
        * portfolioCurrency
        * portfolioDate
        * holdingsView
        * holdingStorageId
        * secId
        * name
        * defaultHoldingTypeCode
        * isin
        * cusip
        * weight
        * shares
        * marketValue
        * sharesChanged
        * ticker
        * currency
        * detailHoldingType
        * Other columns based on requested data points (column name will be the alias if provided, otherwise the data point ID)

        DryRunResults: Is returned if dry_run=True is passed

        * estimated_cells_used: Number of cells by this query
        * daily_cells_remaining_before: How many cells are remaining in your daily cell limit before running this query
        * daily_cells_remaining_after: How many cells would be remaining in your daily cell limit after running this query
        * daily_cell_limit: Your total daily cell limit

    Raises:
        ValueErrorException: Raised when input parameters are invalid.

    :Examples:
        Retrieve holdings for investment "FOUSA00KZH" on "2020-12-31".

    ::

        import morningstar_data as md

        df = md.direct.get_holdings(investments=["FOUSA00KZH"], date="2020-12-31")
        df

    :Output:
        =================  =============  ===  =========   =================
        masterPortfolioId  investmentId   ...  currency    detailHoldingType
        =================  =============  ===  =========   =================
        6079               FOUSA00KZH     ...  US Dollar   EQUITY
        6079               FOUSA00KZH     ...  US Dollar   EQUITY
        ...
        =================  =============  ===  =========   =================

    """
    date = date or None
    start_date = start_date or None
    end_date = end_date or None

    if investment_ids is not None:
        warnings.warn(
            "The investment_ids argument is deprecated and will be removed in the next major version. Use investments instead",
            FutureWarning,
            stacklevel=2,
        )

    investments_object = investments or investment_ids

    if not investments_object:
        raise mdapi.BadRequestError("The `investments` parameter must be included when calling get_holdings") from None

    holdings_request = _create_holdings_request_object(
        investments=investments_object,
        date=date,
        start_date=start_date,
        end_date=end_date,
        data_points=data_points,
        frequency=frequency,
        holdings_view=holdings_view,
        suppression_client_id=suppression_client_id,
        preview=preview,
        enrichment_type=enrichment_type,
        delta_start_time=delta_start_time,
        delta_include_unchanged_data=delta_include_unchanged_data,
        portfolio_depth=PortfolioDepth.BASE_LEVEL,
    )

    if dry_run:
        return get_holding_dry_run_results(holdings_request)

    return get_holdings_data_frame(init_holdings_async(holdings_request))


@_decorator.typechecked
def get_holdings_task(
    investments: Optional[Union[List[str], str, Dict[str, Any], List[InvestmentIdentifier]]] = None,
    date: Optional[Union[str, List[str]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_points: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None,
    frequency: Frequency = Frequency.monthly,
    holdings_view: Union[int, HoldingsView] = HoldingsView.FULL,
    suppression_client_id: Optional[str] = None,
    delta_start_time: Optional[str] = None,
    delta_include_unchanged_data: Optional[bool] = None,
    enrichment_type: EnrichmentType = EnrichmentType.POINT_IN_TIME,
    dry_run: Optional[bool] = False,
    preview: Optional[bool] = False,
) -> Union[str, DryRunResults, Any]:
    """Creates an asynchronous task that fetches base level holdings details for the specified investments. Note that if none of the date parameters are provided, the function uses latest portfolio date by default.

    Args:
        investments (:obj:`Union`, `required`): Defines the investments to fetch. Input can be:

                * Investment IDs (:obj:`list`, `optional`): Investment identifiers, in the format of SecId;Universe or just SecId. E.g., ["F00000YOOK;FO","FOUSA00CFV;FO"] or ["F00000YOOK","FOUSA00CFV"]. Use the `investments <./lookup.html#morningstar_data.direct.investments>`_ function to discover identifiers.
                * Investment List ID (:obj:`str`, `optional`): Saved investment list in Morningstar Direct. Currently, this function does not support lists that combine investments and user-created portfolios. Use the `get_investment_lists <./lists.html#morningstar_data.direct.user_items.get_investment_lists>`_ function to discover saved lists.
                * Search Criteria  ID (:obj:`str`, `optional`): Saved search criteria in Morningstar Direct. Use the `get_search_criteria <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria>`_ function to discover saved search criteria.
                * Search Criteria Condition (:obj:`dict`, `optional`): Search criteria definition. See details in the Reference section below or use the `get_search_criteria_conditions <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria_conditions>`_ function to discover the definition of a saved search criteria.
                * InvestmentIdentifiers (:obj:`list`, `optional`): A list of :class:`~morningstar_data.direct.InvestmentIdentifier` objects, allowing users to specify investments using standard identifiers such as ISIN, CUSIP, and/or ticker symbol instead of Morningstar SecIds.

                    * Multiple Matches: If multiple valid matches exist for a single InvestmentIdentifier, results will be prioritized according to the :ref:`security matching logic<Security Matching Logic>`. The highest ranked security will be used.
                    * Request Limit: Supports up to 500 InvestmentIdentifier objects per request.

        date (:obj:`str, list[str]`, `optional`): The portfolio date for which to retrieve data. The format is YYYY-MM-DD.
            For example, "2020-01-01". If a date is provided, then the `start_date` and `end_date` parameters are ignored.
            Can also be a list of dates. The list can contain a maximum of 20 dates.
            An exception is thrown if `start_date` or `end_date` is provided along with `date`.
        start_date (:obj:`str`, `optional`): The start date for retrieving data. The format is
            YYYY-MM-DD. For example, "2020-01-01". An exception is thrown if `date` is provided along with `start_date`.
        end_date (:obj:`str`, `optional`): The end date for retrieving data. If no value is provided for
            `end_date`, current date will be used. The format is YYYY-MM-DD. For example, "2020-01-01". An exception is
            thrown if `date` is provided along with `end_date`.
        data_points (:obj:`Union`, `optional`): Defines the data points to fetch. If no data points are provided, a default list of data points with predefined aliases are returned, below are the list of default data points and associated aliases.

                +---------------------+---------------+------------------------------+
                | Alias               | Data Point ID | Description                  |
                +=====================+===============+==============================+
                | holdingStorageId    | IHS01         | Holding Storage Id           |
                +---------------------+---------------+------------------------------+
                | secId               | OS00I         | SecId                        |
                +---------------------+---------------+------------------------------+
                | name                | HS765         | Security Name                |
                +---------------------+---------------+------------------------------+
                | defaultHoldingTypeCode  | IHS10         | Detail Holding Type Code     |
                +---------------------+---------------+------------------------------+
                | isin                | HS761         | ISIN                         |
                +---------------------+---------------+------------------------------+
                | cusip               | HS762         | CUSIP                        |
                +---------------------+---------------+------------------------------+
                | weight              | HS750         | Portfolio Weighting Percent  |
                +---------------------+---------------+------------------------------+
                | shares              | HS752         | Number of Shares             |
                +---------------------+---------------+------------------------------+
                | marketValue         | HS751         | Position Market Value        |
                +---------------------+---------------+------------------------------+
                | sharesChanged       | HS753         | Share Change                 |
                +---------------------+---------------+------------------------------+
                | ticker              | OS385         | Ticker                       |
                +---------------------+---------------+------------------------------+
                | currency            | HS763         | Currency                     |
                +---------------------+---------------+------------------------------+
                | detailHoldingType   | HS766         | Detail Holding Type          |
                +---------------------+---------------+------------------------------+

                * Data Point IDs (:obj:`List[Dict]`, `optional`): A list of dictionaries, each defining a data point and its (optional) associated settings. If an alias is provided in the data point setting, it will be used as the column name in the returned DataFrame. Otherwise, the data point ID will be used as the column name.
                * Data Point Settings (:obj:`DataFrame`, `optional`): A DataFrame of data point identifiers and their associated settings. Use the `get_data_set_details <./data_set.html#morningstar_data.direct.user_items.get_data_set_details>`_ function to discover data point settings from a saved data set.

        frequency (:obj:`md.direct.data_type.Frequency`, `optional`): Frequency at which data point values are retrieved. The default is `Frequency.monthly`.

                * daily: Portfolio holdings data reported daily.
                * monthly: Portfolio holdings data reported on or close to the month-end calendar dates.

        holdings_view (:obj:`Union`, `optional`): Specifies the level of detail returned for portfolio holdings. This can be an integer or `md.direct.data_type.HoldingsView`. The default is `HoldingsView.FULL`.

                * Integer Value N: Returns only top N holdings for each portfolio.
                * FULL: Returns all available holdings for the portfolio on requested dates.
                * BEST_AVAILABLE: Returns the most complete holdings data available for each requested date.

        suppression_client_id (:obj:`str`, `optional`): If you have a privileged access to suppressed portfolios and would like to apply it, provide your Client ID.

        enrichment_type (:obj:`md.direct.data_type.EnrichmentType`, `optional`): Specifies the enrichment type of the data returned. The default is `EnrichmentType.POINT_IN_TIME`.

                * LATEST_AVAILABLE: Uses the most current data at feed generation time.
                * POINT_IN_TIME: Matches the data to each portfolio’s effective date.
                * LATEST_MONTH_END: Uses the latest available month-end data at request time.

        dry_run (:obj:`bool`, `optional`): When True, the query will not be executed. Instead, a DryRunResults object will be returned with details about the query's impact on daily cell limit usage.
        preview (:obj:`bool`, `optional`): Defaults to False. Setting to True allows access to data points outside of your current subscription, but limits the output to 25 rows.

    :Returns:

        There are two return types:

        String: Returns created task id.
                Once the task is created, use the `get_task_status <./holdings_async.html#morningstar_data.get_task_status>`_ function to check the status of the task. Then, use the `get_holdings_task_result <./holdings_async.html#morningstar_data.get_holdings_task_result>`_ function to retrieve the DataFrame once the task is completed.
                These are the columns returned:

                * masterPortfolioId
                * secIds
                * portfolioCurrency
                * portfolioDate
                * holdingsView
                * holdingStorageId
                * secId
                * name
                * defaultHoldingTypeCode
                * isin
                * cusip
                * weight
                * shares
                * marketValue
                * sharesChanged
                * ticker
                * currency
                * detailHoldingType
                * Other columns based on requested data points (column name will be the alias if provided, otherwise the data point ID)

        DryRunResults: Is returned if dry_run=True is passed

        * estimated_cells_used: Number of cells used by this query
        * daily_cells_remaining_before: How many cells are remaining in your daily cell limit before running this query
        * daily_cells_remaining_after: How many cells would be remaining in your daily cell limit after running this query
        * daily_cell_limit: Your total daily cell limit

    Raises:
        ValueErrorException: Raised when input parameters are invalid.

    :Examples:
        Retrieve holdings for investment "FOUSA00KZH" on "2020-12-31".

    ::

        import morningstar_data as md

        task_id = md.direct.get_holdings_task(investments=["FOUSA00KZH"], date="2020-12-31")
        task_id

    :Output:
        '10000000-0000-0000-0000-000000000000'

    ------------------------------
    Example asynchronous workflow
    ------------------------------

        Retrieve holdings for investment "FOUSA00KZH" on "2020-12-31".
        Please refer to `get_task_status <./holdings_async.html#morningstar_data.get_task_status>`_ and `get_holdings_task_result <./holdings_async.html#morningstar_data.direct.get_holdings_task_result>`_ documentation on this page.

    ::

        import morningstar_data as md

        task_id = md.direct.get_holdings_task(investments=["FOUSA00KZH"], date="2020-12-31")

        status = md.mdapi.get_task_status(task_id)

        df = md.direct.get_holdings_task_result(s3_url=status.result.pages[0].url)
        df

    :Output:
        =================  =============  ===  =========   =================
        masterPortfolioId  secIds         ...  currency    detailHoldingType
        =================  =============  ===  =========   =================
        6079               FOUSA00KZH     ...  US Dollar   EQUITY
        6079               FOUSA00KZH     ...  US Dollar   EQUITY
        ...
        =================  =============  ===  =========   =================
    """
    holdings_request = _create_holdings_request_object(
        investments=investments,
        date=date,
        start_date=start_date,
        end_date=end_date,
        data_points=data_points,
        frequency=frequency,
        holdings_view=holdings_view,
        suppression_client_id=suppression_client_id,
        preview=preview,
        enrichment_type=enrichment_type,
        delta_start_time=delta_start_time,
        delta_include_unchanged_data=delta_include_unchanged_data,
        portfolio_depth=PortfolioDepth.BASE_LEVEL,
    )

    if dry_run:
        return get_holding_dry_run_results(holdings_request)

    return init_holdings_async(holdings_request).id


@_decorator.typechecked
def get_lookthrough_holdings_task(
    investments: Optional[Union[List[str], str, Dict[str, Any], List[InvestmentIdentifier]]] = None,
    date: Optional[Union[str, List[str]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_points: Optional[Union[List[Dict[str, Any]], str, pd.DataFrame, List[Any]]] = None,
    frequency: Frequency = Frequency.monthly,
    holdings_view: Union[int, HoldingsView] = HoldingsView.FULL,
    suppression_client_id: Optional[str] = None,
    delta_start_time: Optional[str] = None,
    delta_include_unchanged_data: Optional[bool] = None,
    enrichment_type: EnrichmentType = EnrichmentType.POINT_IN_TIME,
    dry_run: Optional[bool] = False,
    preview: Optional[bool] = False,
) -> Union[str, DryRunResults, Any]:
    """Creates an asynchronous task that fetches holdings details and expands all holdings into their nested underlying securities. Note that if none of the date parameters are provided, the function uses latest portfolio date by default.

    Args:
        investments (:obj:`Union`, `required`): Defines the investments to fetch. Input can be:

                * Investment IDs (:obj:`list`, `optional`): Investment identifiers, in the format of SecId;Universe or just SecId. E.g., ["F00000YOOK;FO","FOUSA00CFV;FO"] or ["F00000YOOK","FOUSA00CFV"]. Use the `investments <./lookup.html#morningstar_data.direct.investments>`_ function to discover identifiers.
                * Investment List ID (:obj:`str`, `optional`): Saved investment list in Morningstar Direct. Currently, this function does not support lists that combine investments and user-created portfolios. Use the `get_investment_lists <./lists.html#morningstar_data.direct.user_items.get_investment_lists>`_ function to discover saved lists.
                * Search Criteria  ID (:obj:`str`, `optional`): Saved search criteria in Morningstar Direct. Use the `get_search_criteria <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria>`_ function to discover saved search criteria.
                * Search Criteria Condition (:obj:`dict`, `optional`): Search criteria definition. See details in the Reference section below or use the `get_search_criteria_conditions <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria_conditions>`_ function to discover the definition of a saved search criteria.
                * InvestmentIdentifiers (:obj:`list`, `optional`): A list of :class:`~morningstar_data.direct.InvestmentIdentifier` objects, allowing users to specify investments using standard identifiers such as ISIN, CUSIP, and/or ticker symbol instead of Morningstar SecIds.

                    * Multiple Matches: If multiple valid matches exist for a single InvestmentIdentifier, results will be prioritized according to the :ref:`security matching logic<Security Matching Logic>`. The highest ranked security will be used.
                    * Request Limit: Supports up to 500 InvestmentIdentifier objects per request.

        date (:obj:`str, list[str]`, `optional`): The portfolio date for which to retrieve data. The format is YYYY-MM-DD.
            For example, "2020-01-01". If a date is provided, then the `start_date` and `end_date` parameters are ignored.
            Can also be a list of dates. The list can contain a maximum of 20 dates.
            An exception is thrown if `start_date` or `end_date` is provided along with `date`.
        start_date (:obj:`str`, `optional`): The start date for retrieving data. The format is
            YYYY-MM-DD. For example, "2020-01-01". An exception is thrown if `date` is provided along with `start_date`.
        end_date (:obj:`str`, `optional`): The end date for retrieving data. If no value is provided for
            `end_date`, current date will be used. The format is YYYY-MM-DD. For example, "2020-01-01". An exception is
            thrown if `date` is provided along with `end_date`.
        data_points (:obj:`Union`, `optional`): Defines the data points to fetch. If no data points are provided, a default list of data points with predefined aliases are returned, below are the list of default data points and associated aliases.

                +---------------------+---------------+------------------------------+
                | Alias               | Data Point ID | Description                  |
                +=====================+===============+==============================+
                | holdingStorageId    | IHS01         | Holding Storage Id           |
                +---------------------+---------------+------------------------------+
                | secId               | OS00I         | SecId                        |
                +---------------------+---------------+------------------------------+
                | name                | HS765         | Security Name                |
                +---------------------+---------------+------------------------------+
                | defaultHoldingTypeCode  | IHS10         | Detail Holding Type Code     |
                +---------------------+---------------+------------------------------+
                | weight              | HS750         | Portfolio Weighting Percent  |
                +---------------------+---------------+------------------------------+

                * Data Point IDs (:obj:`List[Dict]`, `optional`): A list of dictionaries, each defining a data point and its (optional) associated settings. If an alias is provided in the data point setting, it will be used as the column name in the returned DataFrame. Otherwise, the data point ID will be used as the column name.
                * Data Point Settings (:obj:`DataFrame`, `optional`): A DataFrame of data point identifiers and their associated settings. Use the `get_data_set_details <./data_set.html#morningstar_data.direct.user_items.get_data_set_details>`_ function to discover data point settings from a saved data set.

        frequency (:obj:`md.direct.data_type.Frequency`, `optional`): Frequency at which data point values are retrieved. The default is `Frequency.monthly`.

                * daily: Portfolio holdings data reported daily.
                * monthly: Portfolio holdings data reported on or close to the month-end calendar dates.

        holdings_view (:obj:`Union`, `optional`): Specifies the level of detail returned for portfolio holdings. This can be an integer or `md.direct.data_type.HoldingsView`. The default is `HoldingsView.FULL`.

                * Integer Value N: Returns only top N holdings for each portfolio.
                * FULL: Returns all available holdings for the portfolio on requested dates.
                * BEST_AVAILABLE: Returns the most complete holdings data available for each requested date.

        suppression_client_id (:obj:`str`, `optional`): If you have a privileged access to suppressed portfolios and would like to apply it, provide your Client ID.

        enrichment_type (:obj:`md.direct.data_type.EnrichmentType`, `optional`): Specifies the enrichment type of the data returned. The default is `EnrichmentType.POINT_IN_TIME`.

                * LATEST_AVAILABLE: Uses the most current data at feed generation time.
                * POINT_IN_TIME: Matches the data to each portfolio’s effective date.
                * LATEST_MONTH_END: Uses the latest available month-end data at request time.

        dry_run (:obj:`bool`, `optional`): When True, the query will not be executed. Instead, a DryRunResults object will be returned with details about the query's impact on daily cell limit usage.
        preview (:obj:`bool`, `optional`): Defaults to False. Setting to True allows access to data points outside of your current subscription, but limits the output to 25 rows.

    :Returns:

        There are two return types:

        String: Returns created task id.
                Once the task is created, use the `get_task_status <./holdings_async.html#morningstar_data.get_task_status>`_ function to check the status of the task. And use the `get_holdings_task_result <./holdings_async.html#morningstar_data.direct.get_holdings_task_result>`_ function to retrieve the result data once the task is completed.
                These are the columns returned in the resultant dataframe:

                * masterPortfolioId
                * secIds
                * portfolioCurrency
                * portfolioDate
                * holdingsView
                * holdingStorageId
                * secId
                * name
                * defaultHoldingTypeCode
                * weight
                * Other columns based on requested data points (column name will be the alias if provided, otherwise the data point ID)

        DryRunResults: Is returned if dry_run=True is passed

        * estimated_cells_used: Number of cells by this query
        * daily_cells_remaining_before: How many cells are remaining in your daily cell limit before running this query
        * daily_cells_remaining_after: How many cells would be remaining in your daily cell limit after running this query
        * daily_cell_limit: Your total daily cell limit

    Raises:
        ValueErrorException: Raised when input parameters are invalid.

    :Examples:
        Retrieve holdings for investment "FOUSA00KZH" on "2020-12-31".

    ::

        import morningstar_data as md

        task_id = md.direct.get_lookthrough_holdings_task(investments=["FOUSA00KZH"], date="2020-12-31")
        task_id

    :Output:
        '10000000-0000-0000-0000-000000000000'

    ------------------------------
    Example asynchronous workflow
    ------------------------------

        Retrieve holdings for investment "FOUSA00KZH" on "2020-12-31".
        Please refer to `get_task_status <./holdings_async.html#morningstar_data.get_task_status>`_ and `get_holdings_task_result <./holdings_async.html#morningstar_data.direct.get_holdings_task_result>`_ documentation on this page.

    ::

        import morningstar_data as md

        task_id = md.direct.get_lookthrough_holdings_task(investments=["FOUSA00KZH"], date="2020-12-31")

        status = md.mdapi.get_task_status(task_id)

        df = md.direct.get_holdings_task_result(s3_url=status.result.pages[0].url)
        df

    :Output:

        =================  =============  ===  ======================   =================
        masterPortfolioId  secIds         ...  defaultHoldingTypeCode   weight
        =================  =============  ===  ======================   =================
        1356725            F00000WK3T     ...  E                        11.16649
        1356725            F00000WK3T     ...  E                        10.49672
        ...
        =================  =============  ===  ======================   =================
    """
    holdings_request = _create_holdings_request_object(
        investments=investments,
        date=date,
        start_date=start_date,
        end_date=end_date,
        data_points=data_points,
        frequency=frequency,
        holdings_view=holdings_view,
        suppression_client_id=suppression_client_id,
        preview=preview,
        enrichment_type=enrichment_type,
        delta_start_time=delta_start_time,
        delta_include_unchanged_data=delta_include_unchanged_data,
        portfolio_depth=PortfolioDepth.LOOK_THROUGH,
    )

    if dry_run:
        return get_holding_dry_run_results(holdings_request)

    return init_lookthrough_holdings_async(holdings_request).id


@_decorator.typechecked
def holding_dates(investment_ids: List[str]) -> DataFrame:
    warnings.warn(
        "The holding_dates function is deprecated and will be removed in the next major version. Use get_holding_dates instead",
        FutureWarning,
        stacklevel=2,
    )
    return get_holding_dates(investment_ids)


@_decorator.typechecked
def get_holding_dates(investment_ids: List[str]) -> DataFrame:
    """Returns all dates with available holdings data for the given investment.

    Args:
        investment_ids (:obj:`list`): A list of investment IDs. The investment ID format is SecId;Universe or just SecId.
            For example: ["F00000YOOK;FO","FOUSA00CFV;FO"] or ["F00000YOOK","FOUSA00CFV"].

    :Returns:
        DataFrame: A DataFrame object with portfolio date data. DataFrame columns include:

        * secId
        * date

    :Examples:
        Retrieve portfolio dates for investment "FOUSA00KZH".

    ::

        import morningstar_data as md

        df = md.direct.get_holding_dates(investment_ids=["FOUSA06JNH"])
        df

    :Output:
        ==========  ==========
        secId       date
        ==========  ==========
        FOUSA06JNH  2021-08-31
        FOUSA06JNH  2021-07-31
        ...
        ==========  ==========

    """
    # TODO: update the docs
    if not investment_ids:
        raise BadRequestException(BAD_REQUEST_ERROR_NO_INVESTMENT_IDS) from None

    holding_date_request = HoldingDateRequest(investments=investment_ids)

    data = mdapi.get_holding_dates(holding_date_request)

    return pd.DataFrame(data.holdings_dates)


def _get_pandas_version() -> tuple:
    return tuple(map(int, pd.__version__.split(".")[:2]))


@_decorator.typechecked
def get_holdings_task_result(s3_url: str, flatten_secids: bool = True) -> DataFrame:
    """
    Given an S3 URL from  `get_task_status <./holdings_async.html#morningstar_data.get_task_status>`_, fetches the holdings data and returns it as a DataFrame.

    Args:
        s3_url (:obj:`str`, `required`): S3 URL to fetch the holdings data from.
        flatten_secids (:obj:`bool`, `optional`): Default is True. If set to False, the 'secIds' column will be a list of investment ids.

    :Examples:

        task_id = md.direct.get_lookthrough_holdings_task(investments=["FOUSA00KZH"], date="2020-12-31")

        status = md.mdapi.get_task_status(task_id)

        df = md.direct.get_holdings_task_result(s3_url=status.result.pages[0].url)
        df

    :Output:

        =================  =============  ===  ======================   =================
        masterPortfolioId  secIds         ...  defaultHoldingTypeCode   weight
        =================  =============  ===  ======================   =================
        1356725            F00000WK3T     ...  E                        11.16649
        1356725            F00000WK3T     ...  E                        10.49672
        ...
        =================  =============  ===  ======================   =================
    """
    if _get_pandas_version() < (2, 0):
        df_data = pd.read_json(s3_url, lines=True, compression="gzip")
    else:
        df_data = pd.read_json(s3_url, lines=True, compression="gzip", dtype_backend="pyarrow")

    if df_data.get("holdingDetails") is not None:
        # Use a funky separator to be extra sure we know which columns got suffixes added during flattening
        normalized_df = pd.json_normalize(df_data["holdingDetails"], sep=".!,*")
        df_data = df_data.drop(columns=["holdingDetails"]).join(normalized_df)

    if flatten_secids and "secIds" in df_data.columns:
        df_data = df_data.explode("secIds").reset_index(drop=True)

    # Investment API sends timeseries datapoints as a key-value pair {"i": <date>, "v": value}
    # and other special specific datapoints like {"code": 103, "value": "USA"}
    # Pandas will add the `.!,*` suffixes to all columns in "holdingDetails" when we flatten
    # The weird seperator is specified above to avoid conflicts with existing column names
    # Let's drop the "date" and "code" columns and remove those suffixes for value columns

    df_data = df_data[[col for col in df_data.columns if not col.endswith(".!,*i") and not col.endswith(".!,*code")]]
    df_data = df_data.rename(columns=lambda c: re.sub(r"\.!,\*.+$", "", c))
    return df_data


@_decorator.typechecked
def get_lookthrough_holdings(
    investments: Optional[Union[List[str], str, Dict[str, Any], List[InvestmentIdentifier]]] = None,
    date: Optional[Union[str, List[str]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_points: Optional[Union[List[Dict[str, Any]], str, pd.DataFrame, List[Any]]] = None,
    frequency: Frequency = Frequency.monthly,
    holdings_view: Union[int, HoldingsView] = HoldingsView.FULL,
    suppression_client_id: Optional[str] = None,
    enrichment_type: EnrichmentType = EnrichmentType.POINT_IN_TIME,
    delta_start_time: Optional[str] = None,
    delta_include_unchanged_data: Optional[bool] = None,
    dry_run: Optional[bool] = False,
    preview: Optional[bool] = False,
) -> Union[DataFrame, DryRunResults]:
    """Returns holdings details and expands all holdings into their nested underlying securities. Note that if none of the date parameters are provided, the function uses latest portfolio date by default.

    Args:
        investments (:obj:`Union`, `required`): Defines the investments to fetch. Input can be:

                * Investment IDs (:obj:`list`, `optional`): Investment identifiers, in the format of SecId;Universe or just SecId. E.g., ["F00000YOOK;FO","FOUSA00CFV;FO"] or ["F00000YOOK","FOUSA00CFV"]. Use the `investments <./lookup.html#morningstar_data.direct.investments>`_ function to discover identifiers.
                * Investment List ID (:obj:`str`, `optional`): Saved investment list in Morningstar Direct. Currently, this function does not support lists that combine investments and user-created portfolios. Use the `get_investment_lists <./lists.html#morningstar_data.direct.user_items.get_investment_lists>`_ function to discover saved lists.
                * Search Criteria  ID (:obj:`str`, `optional`): Saved search criteria in Morningstar Direct. Use the `get_search_criteria <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria>`_ function to discover saved search criteria.
                * Search Criteria Condition (:obj:`dict`, `optional`): Search criteria definition. See details in the Reference section below or use the `get_search_criteria_conditions <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria_conditions>`_ function to discover the definition of a saved search criteria.
                * InvestmentIdentifiers (:obj:`list`, `optional`): A list of :class:`~morningstar_data.direct.InvestmentIdentifier` objects, allowing users to specify investments using standard identifiers such as ISIN, CUSIP, and/or ticker symbol instead of Morningstar SecIds.

                    * Multiple Matches: If multiple valid matches exist for a single InvestmentIdentifier, results will be prioritized according to the :ref:`security matching logic<Security Matching Logic>`. The highest ranked security will be used.
                    * Request Limit: Supports up to 500 InvestmentIdentifier objects per request.

        date (:obj:`str, list[str]`, `optional`): The portfolio date for which to retrieve data. The format is YYYY-MM-DD.
            For example, "2020-01-01". If a date is provided, then the `start_date` and `end_date` parameters are ignored.
            Can also be a list of dates. The list can contain a maximum of 20 dates.
            An exception is thrown if `start_date` or `end_date` is provided along with `date`.
        start_date (:obj:`str`, `optional`): The start date for retrieving data. The format is
            YYYY-MM-DD. For example, "2020-01-01". An exception is thrown if `date` is provided along with `start_date`.
        end_date (:obj:`str`, `optional`): The end date for retrieving data. If no value is provided for
            `end_date`, current date will be used. The format is YYYY-MM-DD. For example, "2020-01-01". An exception is
            thrown if `date` is provided along with `end_date`.
        data_points (:obj:`Union`, `optional`): Defines the data points to fetch. If no data points are provided, a default list of data points with predefined aliases are returned, below are the list of default data points and associated aliases.

                +---------------------+---------------+------------------------------+
                | Alias               | Data Point ID | Description                  |
                +=====================+===============+==============================+
                | holdingStorageId    | IHS01         | Holding Storage Id           |
                +---------------------+---------------+------------------------------+
                | secId               | OS00I         | SecId                        |
                +---------------------+---------------+------------------------------+
                | name                | HS765         | Security Name                |
                +---------------------+---------------+------------------------------+
                | defaultHoldingTypeCode  | IHS10         | Detail Holding Type Code     |
                +---------------------+---------------+------------------------------+
                | weight              | HS750         | Portfolio Weighting Percent  |
                +---------------------+---------------+------------------------------+

                * Data Point IDs (:obj:`List[Dict]`, `optional`): A list of dictionaries, each defining a data point and its (optional) associated settings. If alias is provided in the datapoint setting, it will be used as the column name in the returned DataFrame. Otherwise, the data point ID will be used as the column name.
                * Data Point Settings (:obj:`DataFrame`, `optional`): A DataFrame of data point identifiers and their associated settings. Use the `get_data_set_details <./data_set.html#morningstar_data.direct.user_items.get_data_set_details>`_ function to discover data point settings from a saved data set.

        frequency (:obj:`md.direct.data_type.Frequency`, `optional`): Frequency at which data point values are retrieved. The default is `Frequency.monthly`.

                * daily: Portfolio holdings data reported daily.
                * monthly: Portfolio holdings data reported on or close to the month-end calendar dates.

        holdings_view (:obj:`Union`, `optional`): Specifies the level of detail returned for portfolio holdings. This can be an integer or `md.direct.data_type.HoldingsView`. The default is `HoldingsView.FULL`.

                * Integer Value N: Returns only top N holdings for each portfolio.
                * FULL: Returns all available holdings for the portfolio on requested dates.
                * BEST_AVAILABLE: Returns the most complete holdings data available for each requested date.

        suppression_client_id (:obj:`str`, `optional`): If you have a privileged access to suppressed portfolios and would like to apply it, provide your Client ID.

        enrichment_type (:obj:`md.direct.data_type.EnrichmentType`, `optional`): Specifies the enrichment type of the data returned. The default is `EnrichmentType.POINT_IN_TIME`.

                * LATEST_AVAILABLE: Uses the most current data at feed generation time.
                * POINT_IN_TIME: Matches the data to each portfolio’s effective date.
                * LATEST_MONTH_END: Uses the latest available month-end data at request time.

        dry_run (:obj:`bool`, `optional`): When True, the query will not be executed. Instead, a DryRunResults object will be returned with details about the query's impact on daily cell limit usage.
        preview (:obj:`bool`, `optional`): Defaults to False. Setting to True allows access to data points outside of your current subscription, but limits the output to 25 rows.

    :Returns:

        There are two return types:

        DataFrame: A DataFrame object with holdings data. DataFrame columns include

        * masterPortfolioId
        * investmentId
        * portfolioCurrency
        * portfolioDate
        * holdingsView
        * holdingStorageId
        * secId
        * name
        * defaultHoldingTypeCode
        * weight
        * Other columns based on requested data points (column name will be the alias if provided, otherwise the data point ID)

        DryRunResults: Is returned if dry_run=True is passed

        * estimated_cells_used: Number of cells by this query
        * daily_cells_remaining_before: How many cells are remaining in your daily cell limit before running this query
        * daily_cells_remaining_after: How many cells would be remaining in your daily cell limit after running this query
        * daily_cell_limit: Your total daily cell limit

    :Examples:

    Retrieve the look-through holdings for a portfolio.
    ::

        import morningstar_data as md

        df = md.direct.get_lookthrough_holdings(investments=["FOUSA00KZH"], date="2020-12-31")
        df

    :Output:

        =================  =============  ===  ======================   =================
        masterPortfolioId  investmentId   ...  defaultHoldingTypeCode   weight
        =================  =============  ===  ======================   =================
        1356725            F00000WK3T     ...  E                        11.16649
        1356725            F00000WK3T     ...  E                        10.49672
        ...
        =================  =============  ===  ======================   =================
    """

    _logger.info("morningstar_data.direct.get_lookthrough_holdings")

    holdings_request = _create_holdings_request_object(
        investments=investments,
        date=date,
        start_date=start_date,
        end_date=end_date,
        data_points=data_points,
        frequency=frequency,
        holdings_view=holdings_view,
        suppression_client_id=suppression_client_id,
        preview=preview,
        enrichment_type=enrichment_type,
        delta_start_time=delta_start_time,
        delta_include_unchanged_data=delta_include_unchanged_data,
        portfolio_depth=PortfolioDepth.LOOK_THROUGH,
    )

    if dry_run:
        return get_holding_dry_run_results(holdings_request)

    return get_holdings_data_frame(init_lookthrough_holdings_async(holdings_request))


def _create_holdings_request_object(
    investments: Optional[Union[List[str], str, Dict[str, Any], List[InvestmentIdentifier]]],
    date: Optional[Union[str, List[str]]],
    start_date: Optional[str],
    end_date: Optional[str],
    data_points: Optional[Union[List[Dict[str, Any]], str, pd.DataFrame, List[Any]]],
    frequency: Frequency,
    holdings_view: Union[int, HoldingsView],
    suppression_client_id: Optional[str],
    preview: Optional[bool],
    enrichment_type: EnrichmentType,
    delta_start_time: Optional[str],
    delta_include_unchanged_data: Optional[bool],
    portfolio_depth: PortfolioDepth,
) -> AsyncHoldingsRequest:
    """
    Create an AsyncHoldingsRequest object with the provided parameters.
    This exists mostly to standardize the passing of `dates` to the API as a list even if a single date is provided.
    """
    req = AsyncHoldingsRequest(
        investments=investments,
        date=None,
        start_date=start_date,
        end_date=end_date,
        data_points=data_points,
        dates=None,
        frequency=frequency.abbr,
        holdings_view=holdings_view,
        suppression_client_id=suppression_client_id,
        preview=preview,
        enrichment_type=enrichment_type,
        delta_start_time=delta_start_time,
        delta_include_unchanged_data=delta_include_unchanged_data,
        portfolio_depth=portfolio_depth,
    )

    req.check_size_of_request()

    if isinstance(date, str):
        req.dates = [date]
    elif isinstance(date, list):
        req.dates = date
    return req


def get_holdings_data_frame(task: MdapiTask) -> pd.DataFrame:
    try:
        response = _wait_for_completed_task_response(task)
        if response.is_successful():
            task_result = response.result
            if task_result and not task_result.pages:
                raise MdApiTaskException(task_id=str(task.id), detail="No S3 URLs found in task result")
            else:
                return merge_holdings_data_frames(task_result)

    except Exception as e:  # pylint: disable=broad-exception-caught
        raise MdApiTaskException(task_id=str(task.id), detail=repr(e)) from e


def merge_holdings_data_frames(task_result: Optional[TaskResult]) -> pd.DataFrame:
    """Merge DataFrames containing holdings data."""
    if task_result and task_result.pages:
        dfs = [get_holdings_task_result(x.url, True) for x in task_result.pages]
        df = pd.concat(dfs, axis=0, ignore_index=True)

        # For backwards compatibility with existing code that expects "investmentId" column,
        # rename "secIds" to "investmentId". All other column names remain unchanged.
        df.rename(columns={"secIds": "investmentId"}, inplace=True)
        return df
    else:
        return pd.DataFrame()
