from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pandas import DataFrame

from .. import mdapi
from .._base import _logger
from .._utils import (
    _get_data_points_total_columns,
    _get_user_cells_quota,
    data_point_dataframe_to_list,
)
from ..mdapi import RequestObject
from . import _decorator, _error_messages, _utils
from ._base_api import APIBackend
from ._config import _Config
from ._data_objects import Investments
from ._data_point import _request_asset_flow_data_points
from ._data_type import DryRunResults
from ._exceptions import BadRequestException, QueryLimitException, ResourceNotFoundError

_config = _Config()


class AssetFlowAPIBackend(APIBackend):
    """
    Subclass to call the Asset Flow Data API and handle any HTTP errors that occur.
    """

    def __init__(self) -> None:
        super().__init__()

    def _handle_custom_http_errors(self, res: Any) -> Any:
        """
        Handle HTTP errors with custom error messages
        """
        response_message = res.json().get("message")
        if res.status_code == 403 and "Exceed query limitation." in response_message:
            _logger.debug(f"Query Limit Exception: {res.status_code} {response_message}")
            query_limit = self._get_security_data_query_limit()
            raise QueryLimitException(query_limit) from None
        elif res.status_code == 404:
            _logger.debug(f"Resource Not Found Error: {res.status_code} {response_message}")
            raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_ASSET_FLOW) from None

    def _get_security_data_query_limit(self) -> str:
        try:
            url = f"{_config.securitydata_service_url()}v1/limitation/summary"
            res = self.do_get_request(url)
            return str(res["limitationTotal"])
        except Exception as e:
            _logger.error(f"Error getting security data query limit: {e}")
            raise QueryLimitException from None


_asset_flow_api_request = AssetFlowAPIBackend()


@_decorator.typechecked
def get_asset_flow_markets() -> DataFrame:
    """

    Returns:

        Returns all investment markets that can be used to retrieve asset flow data. For example, "US Open-end & ETFs ex MM ex FoF".

        DataFrame: A DataFrame object with asset flow markets data. DataFrame columns include:

        * marketId
        * marketName
        * currency

    :Examples:

    ::

        import morningstar_data as md

        df = md.direct.get_asset_flow_markets()
        df

    :Output:
        ========  ===============================  ========
        marketId  marketName                       currency
        ========  ===============================  ========
        5         US Open-end & ETF ex MM ex FoF   USD
        6         US Open-end, ETF, and MM ex FoF  USD
        ...
        ========  ===============================  ========

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    url = f"{_config.securitydata_service_url()}v1/assetflow/all-markets"
    response_json = _asset_flow_api_request.do_get_request(url)
    if response_json and isinstance(response_json, list):
        return DataFrame(response_json)[["marketId", "marketName", "currency"]]
    else:
        return DataFrame({"marketId": [], "marketName": [], "currency": []})


def get_asset_flow_data_points() -> DataFrame:
    """

    Returns:

        Returns all available data points related to asset flows.

        DataFrame: A DataFrame object with asset flow data points data. DataFrame columns include:

        * datapointId
        * datapointName
        * asOfDate
        * alias
        * startDate
        * endDate
        * frequency

    :Examples:

    ::

        import morningstar_data as md

        df = md.direct.get_asset_flow_data_points()
        df

    :Output:
        ===========  =============  ==========  ==========================================  =========  =======  =========
        datapointId  datapointName  asOfDate    alias                                       startDate  endDate  frequency
        ===========  =============  ==========  ==========================================  =========  =======  =========
        TNA0M        XXX            2021-09-30  Total Net Assets-Market Value(Share Class)  None       None     None
        ...
        ===========  =============  ==========  ==========================================  =========  =======  =========

    """
    response_json = _request_asset_flow_data_points()
    if response_json and isinstance(response_json, list):
        for settings in response_json:
            settings["datapointName"] = settings.pop("name")
            settings["alias"] = settings["datapointName"]

        settings = _utils._extract_data(response_json)
        settings_data_frame = DataFrame(settings)

        column_list = settings_data_frame.columns.tolist()
        if ("datapointId" in column_list) & (column_list.index("datapointId") != 0):
            data_point_id_col = settings_data_frame["datapointId"]
            settings_data_frame = settings_data_frame.drop("datapointId", axis=1)
            settings_data_frame.insert(0, "datapointId", data_point_id_col)

        column_list = settings_data_frame.columns.tolist()
        if ("datapointName" in column_list) & (column_list.index("datapointName") != 1):
            data_point_name_col = settings_data_frame["datapointName"]
            settings_data_frame = settings_data_frame.drop("datapointName", axis=1)
            settings_data_frame.insert(1, "datapointName", data_point_name_col)

        settings_data_frame = settings_data_frame.where(settings_data_frame.notnull(), None)
        return settings_data_frame
    else:
        return DataFrame()


InvestmentsParam = Union[List[str], str, Dict[str, Any]]


@dataclass
class AssetFlowRequest(RequestObject):
    market_id: str
    datapoints: List[Dict[str, Any]]
    investments: Optional[InvestmentsParam]


@_decorator.typechecked
def get_asset_flow(
    market_id: str,
    data_point_settings: DataFrame,
    investments: Optional[Union[List[str], str, Dict[str, Any]]] = None,
    dry_run: Optional[bool] = False,
) -> Union[DataFrame, DryRunResults]:
    """Get asset flow data for a market of investments or specific investments within a market

    Args:
        market_id(:obj:`str`): A numeric code representing a broad market of investments. For example, the code for "US Open-end & ETF ex MM ex FoF" is "5". Use the `get_asset_flow_markets <./assetflow.html#morningstar_data.direct.get_asset_flow_markets>`_ function to retrieve a full list of codes.

        data_point_settings(:obj:`DataFrame`): A DataFrame of data points with defined settings. Each row represents a data point. Each column is a configurable setting. This DataFrame can be obtained by `retrieving asset flow data points <./assetflow.html#morningstar_data.direct.get_asset_flow_data_points>`_, or by `retrieving data point settings <./lookup.html#morningstar_data.direct.get_data_point_settings>`_.

        investments(:obj:`Union`, `required`): Defines the investments to fetch. Input can be:

            * Investment IDs (:obj:`list`, `optional`): Investment identifiers, in the format of SecId;Universe or just SecId. E.g., ["F00000YOOK;FO","FOUSA00CFV;FO"] or ["F00000YOOK","FOUSA00CFV"]. Use the `investments <./lookup.html#morningstar_data.direct.investments>`_ function to discover identifiers.
            * Investment List ID (:obj:`str`, `optional`): Saved investment list in Morningstar Direct. Use the `get_investment_lists <./lists.html#morningstar_data.direct.user_items.get_investment_lists>`_ function to discover saved lists.
            * Search Criteria  ID (:obj:`str`, `optional`): Saved search criteria in Morningstar Direct. Use the `get_search_criteria <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria>`_ function to discover saved search criteria.
            * Search Criteria Condition (:obj:`dict`, `optional`): Search criteria definition. See details in the Reference section of `get_investment_data <./investment.html#morningstar_data.direct.get_investment_data>`_ or use the `get_search_criteria_conditions <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria_conditions>`_ function to discover the definition of a saved search criteria.

        dry_run(:obj:`bool`, `optional`): When True, the query will not be executed. Instead, a DryRunResults object will be returned with details about the query's impact on daily cell limit usage. When True, the 'investments' parameter must be a list containing at least one element.

    :Returns:

        There are two return types:

        * DataFrame: A DataFrame object with asset flow data. DataFrame columns include ``investmentId`` and data point names, as provided in ``data_point_settings``.

        * DryRunResults: Returned if dry_run=True is passed

          * estimated_cells_used: Number of cells by this query
          * daily_cells_remaining_before: How many cells are remaining in your daily cell limit before running this query
          * daily_cells_remaining_after: How many cells would be remaining in your daily cell limit after running this query
          * daily_cell_limit: Your total daily cell limit

    :Examples:

    ::

        import morningstar_data as md
        import pandas

        ASSET_FLOW_DATA_POINT_SETTINGS = [
            {
                "datapointId": "TNA0M",
                "datapointName": "Total Net Assets-Market Value(Share Class)",
                "asOfDate": "2021-08-30",
                "alias": "Total Net Assets-Market Value(Share Class)",
                "startDate": None,
                "endDate": None,
                "frequency": None,
            }
        ]
        settings = pandas.DataFrame(ASSET_FLOW_DATA_POINT_SETTINGS)
        df = md.direct.get_asset_flow(
            investments=["F000010HRO"], market_id="165", data_point_settings=settings
        )
        df

    :Output:
        ============  =======================================================
        investmentId  Total Net Assets-Market Value(Share Class) - 2021-06-30
        ============  =======================================================
        F000010HRO    0.00188
        ============  =======================================================

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    if investments == {}:
        raise mdapi.BadRequestError("investments may not be an empty search criteria")

    if data_point_settings.empty:
        raise mdapi.BadRequestError("data_point_settings must not be empty")

    if dry_run and not investments:
        raise BadRequestException(_error_messages.BAD_REQUEST_ERROR_NO_INVESTMENT_IDS_FOR_DRY_RUN)

    if dry_run:
        if investments is not None:
            investment_object = Investments(investments)
            investment_id_list = investment_object.get_investment_ids()

        data_point_settings = data_point_settings.where(data_point_settings.notnull(), None)
        data_point_list = data_point_settings.to_dict(orient="records")
        estimated_cells_used = _get_data_points_total_columns(data_point_list) * len(investment_id_list)
        user_cells_quota = _get_user_cells_quota()
        dry_run_results = DryRunResults(
            estimated_cells_used=estimated_cells_used,
            daily_cells_remaining_before=user_cells_quota["daily_cell_remaining"],
            daily_cell_limit=user_cells_quota["daily_cell_limit"],
        )
        return dry_run_results

    datapoints = data_point_dataframe_to_list(data_point_settings)
    # TODO: Any extra validation? Check investment list not too long? Or do we let exception happen and catch it
    return mdapi.call_remote_function(
        "get_asset_flow",  # TODO: Make this a constant/enum
        AssetFlowRequest(
            market_id=market_id,
            datapoints=datapoints,
            investments=investments,
        ),
    )
