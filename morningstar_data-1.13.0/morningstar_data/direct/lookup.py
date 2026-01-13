import re
from enum import Enum
from typing import Any, List, Optional
from urllib.parse import quote

import pandas
from pandas import DataFrame

from .. import mdapi
from .._base import _logger
from ..direct._config import _Config
from . import _decorator, _error_messages
from ._base_api import APIBackend
from ._config_key import ALL_ASSET_FLOW_DATA_POINTS
from ._data_point import (
    _get_all_universes,
    _get_asset_flow_data_points_by_ids,
    _get_data_point_details,
)
from ._exceptions import BadRequestException, ResourceNotFoundError, ValueErrorException
from ._utils import (
    _empty_to_none,
    _extract_data,
    _filter_data_frame_column_by_setting,
    _reindex_data_frame_column,
)
from .data_type import InvestmentIdentifier, Universe

_config = _Config()


class _LookupModule(Enum):
    DATA_SET = 0
    DATA_POINT = 1
    INVESTMENT = 2


class LookupDataAPIBackend(APIBackend):
    """
    Subclass to call the Lookup Data API and handle any HTTP errors that occur.
    """

    def __init__(self, lookup_module: _LookupModule) -> None:
        super().__init__()
        self._lookup_module = lookup_module

    def _get_resource_not_found_message(self) -> str:
        if self._lookup_module == _LookupModule.DATA_SET:
            return _error_messages.RESOURCE_NOT_FOUND_ERROR_DATA_SET_LOOKUP
        elif self._lookup_module == _LookupModule.DATA_POINT:
            return _error_messages.RESOURCE_NOT_FOUND_ERROR_DATA_POINT_LOOKUP
        return _error_messages.RESOURCE_NOT_FOUND_ERROR

    def _handle_custom_http_errors(self, res: Any) -> Any:
        """
        Handle HTTP errors with custom error messages
        """
        response_message = res.json().get("message")
        if res.status_code == 404:
            _logger.debug(f"Resource Not Found Error: {res.status_code} {response_message}")
            raise ResourceNotFoundError(self._get_resource_not_found_message()) from None


_data_set_lookup_api_request = LookupDataAPIBackend(_LookupModule.DATA_SET)
_investment_lookup_api_request = LookupDataAPIBackend(_LookupModule.INVESTMENT)

_data_point_lookup_api_request = LookupDataAPIBackend(_LookupModule.DATA_POINT)


class investment_universes:
    All_Managed_Investments = Universe("All Managed Investments", "MI")
    Bonds = Universe("Bonds", "BD")
    Closed_End_Funds = Universe("Closed End Funds", "FC")
    Collective_Investment_Trusts = Universe("Collective Investment Trusts", "CZ")
    European_Pension_Life_Fund_Wrappers = Universe("European Pension/Life Fund Wrappers", "PS")
    eVestment_Hedge_Funds = Universe("eVestment Hedge Funds", "VH")
    eVestment_Separate_Accounts = Universe("eVestment Separate Accounts", "VS")
    Exchange_Traded_Funds = Universe("Exchange-Traded Funds", "FE")
    Funds_Open_End_and_Exchange_Traded_Funds = Universe("Funds (Open End and Exchange-Traded Funds)", "FX")
    France_Dedicated_Funds = Universe("France Dedicated Funds", "DF")
    Global_Restricted_Funds = Universe("Global Restricted Funds", "R1")
    Hedge_Funds = Universe("Hedge Funds", "FH")
    HFR_Hedge_Funds = Universe("HFR Hedge Funds", "H1")
    HSBC_Restricted_Funds = Universe("HSBC Restricted Funds", "IF")
    Insurance_and_Pension_Funds = Universe("Insurance and Pension Funds", "FV")
    Market_Indexes = Universe("Market Indexes", "XI")
    Models = Universe("Models", "MO")
    Money_Market_Funds = Universe("Money Market Funds", "FM")
    Open_End_Funds = Universe("Open End Funds", "FO")
    Plans_529 = Universe("529 Plans", "CP")
    Portfolios_529 = Universe("529 Portfolios", "CT")
    Private_Funds = Universe("Private Funds", "SP")
    Separate_Accounts = Universe("Separate Accounts", "SA")
    Stocks = Universe("Stocks", "ST")
    Strategies = Universe("Strategies", "MG")
    UBS_Separate_Accounts = Universe("UBS Separate Accounts", "S1")
    UK_Life_and_Pension_Funds = Universe("UK Life and Pension Funds", "V1")
    Unit_Investment_Trust = Universe("Unit Investment Trust", "FI")
    US_Variable_Annuities = Universe("US Variable Annuities", "VP")
    US_Variable_Annuity_Subaccounts = Universe("US Variable Annuity Subaccounts", "VA")
    US_Variable_Life = Universe("US Variable Life", "LP")
    US_Variable_Life_Subaccounts = Universe("US Variable Life Subaccounts", "VL")


def _search_data(datapoint: str, keyword: str, universe: str, findBy: str = "name") -> Any:
    # Encode keyword.
    _logger.debug(f"Encode the {keyword} keyword in utf-8")
    keyword_encode = quote(keyword, "utf-8")

    _logger.debug(f"Search by {findBy} in the {datapoint} data point with: keyword: {keyword}, universe: {universe}")
    url = f"{_config.securitydata_service_url()}v1/find?datapoint={datapoint}&kw={keyword_encode}&operator=namecontain&universe={universe}&findby={findBy}&status=activeonly"

    response_json = _investment_lookup_api_request.do_get_request(url)
    return response_json


@_decorator.typechecked
def get_morningstar_data_sets(universe: Optional[str] = None) -> DataFrame:
    """Returns all Morningstar pre-defined data sets.

    Args:
        universe (:obj:`str`, `optional`): Investment universe code. Example: "FO". Use `get_investment_universes <./lookup.html#morningstar_data.direct.lookup.get_investment_universes>`_ to discover possible values.

    :Returns:
        DataFrame: A DataFrame object with Morningstar data sets. DataFrame columns include:

        * datasetId
        * name

    :Examples:
        Retrieve the Morningstar data set for the open-end fund universe.

    ::

        import morningstar_data as md

        df = md.direct.lookup.get_morningstar_data_sets(universe="FO")
        df

    :Output:
        =========  ===============================
        datasetId  name
        =========  ===============================
        0026-0020  Snapshot
        0026-0447  Sustainability: ESG Risk (Fund)
        ...
        =========  ===============================

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    if universe:
        url = f"{_config.data_point_service_url()}v1/defaultviews?type=screen&universe={universe}"
    else:
        url = f"{_config.data_point_service_url()}v1/defaultviews?type=list"
    response_json = _data_set_lookup_api_request.do_get_request(url)
    result = DataFrame(response_json)
    result["datasetId"] = result["id"]
    return result[["datasetId", "name"]]


def _get_multiple_data_point_details(params: list) -> list:
    asset_flow_data_points = []
    normal_data_points = []
    params = params if params else []
    for data_point in params:
        if data_point is None:
            continue
        if data_point.get("datapointId", "").strip() in ALL_ASSET_FLOW_DATA_POINTS:
            asset_flow_data_points.append(data_point.get("datapointId", "").strip())
        else:
            normal_data_points.append(data_point)

    result = []
    if normal_data_points:
        normal_response_json = _get_data_point_details(normal_data_points)
        if isinstance(normal_response_json, list) and len(normal_response_json) > 0:
            result.extend(normal_response_json)

    if asset_flow_data_points:
        result.extend(_get_asset_flow_data_points_by_ids(asset_flow_data_points))

    return _data_points_reorder(params, result)


def _data_points_reorder(params: list, result: list) -> list:
    target = []
    for data_point in params:
        target.extend(_data_points_mapping(data_point, result))
    return target


def _data_points_mapping(data_point: Optional[dict], result: list) -> List[Any]:
    mapping_result: List[Any] = []
    if data_point is None:
        return mapping_result
    data_point_id = data_point.get("datapointId", "").strip()
    is_ts_dp = data_point.get("isTsdp", None)
    if data_point_id in ALL_ASSET_FLOW_DATA_POINTS:
        target_list = list(filter(lambda x: (data_point_id == x.get("datapointId", "").strip()), result))
        if len(target_list) > 0:
            mapping_result.append(target_list[0])
    elif is_ts_dp is not None:
        target_list = list(
            filter(
                lambda x: (data_point_id == x.get("datapointId", "").strip()) & (is_ts_dp == x.get("isTsdp", None)),
                result,
            )
        )
        if len(target_list) > 0:
            mapping_result.append(target_list[0])
    else:
        target_list_for_ts = list(
            filter(
                lambda x: (data_point_id == x.get("datapointId", "").strip()) & (x.get("isTsdp", None) is True),
                result,
            )
        )
        target_list_for_no_ts = list(
            filter(
                lambda x: (data_point_id == x.get("datapointId", "").strip()) & (x.get("isTsdp", None) is False),
                result,
            )
        )
        if len(target_list_for_ts) > 0:
            mapping_result.append(target_list_for_ts[0])
        if len(target_list_for_no_ts) > 0:
            mapping_result.append(target_list_for_no_ts[0])

    return mapping_result


@_decorator.not_null
@_decorator.typechecked
def get_data_point_settings(data_point_ids: List[str]) -> DataFrame:
    """Returns settings for a given set of data points. This settings DataFrame can be manipulated
    to reflect specific settings to be used for data retrieval.

    Args:
        data_point_ids (:obj:`list`): A list of unique identifiers for data points.
            Example: ["OS01W", "HP010"]

    :Returns:
        DataFrame: A DataFrame object with data point settings. DataFrame columns include:

        * datapointId
        * datapointName
        * displayName
        * currency
        * preEuroConversion
        * sourceId
        * frequency
        * startDate
        * endDate
        * floatStart
        * floatEnd
        * startDelay
        * endDelay
        * diffStart
        * diffEnd
        * compounding
        * calculationId
        * annualized
        * annualDays
        * benchmark
        * riskfree
        * windowType
        * windowSize
        * stepSize
        * requireContinueData
        * fit
        * scalType
        * scalValue
        * scalPercentValue
        * timehorizon

    :Examples:
        Get data point settings for data point "OS01W".

    ::

        import morningstar_data as md

        df = md.direct.get_data_point_settings(data_point_ids=["OS01W"])
        df

    :Output:
        ===========  =============  ===  ========  ========  =========
        datapointId  datapointName  ...  calcMnav  showType  transType
        ===========  =============  ===  ========  ========  =========
        OS01W	     Name           ...  None      None      None
        ===========  =============  ===  ========  ========  =========

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    params = list(map(lambda x: {"datapointId": x}, data_point_ids))
    response_json = _get_multiple_data_point_details(params)
    if len(response_json) == 0:
        raise ValueErrorException("There are no data points.")
    extracted_data = _extract_data(response_json)
    details = DataFrame(extracted_data)
    filtered_details = _filter_data_frame_column_by_setting(details)
    details_with_relocated_data_point_id = _reindex_data_frame_column(filtered_details, target_column="datapointId", new_index=0)
    details_with_relocated_data_point_name = _reindex_data_frame_column(
        details_with_relocated_data_point_id, target_column="datapointName", new_index=1
    )
    return _empty_to_none(details_with_relocated_data_point_name)


@_decorator.not_null
@_decorator.typechecked
def investments(
    keyword: str = "", investment: Optional[InvestmentIdentifier] = None, count: int = 50, only_surviving: bool = True
) -> DataFrame:
    """Returns investments that match the given keyword or :class:`~morningstar_data.direct.InvestmentIdentifier` object.

    Args:
        keyword (:obj:`str`, `optional`): Keyword to search for investments.
        investment(:obj:`InvestmentIdentifier`, `optional`): Identifies an investment by its ISIN, CUSIP, ticker, base currency, and/or exchange.
        count (:obj:`int`, `optional`): Maximum number of matching investments to return.
        only_surviving (:obj:`bool`, `optional`): Include only surviving investments.

    .. note::
        You must specify one of, but not both of ``keyword`` or ``investment``.

    :Returns:
        DataFrame: A DataFrame containing the matching investments with the following columns:


        * Name
        * SecId: Morningstar's unique identifier for this investment
        * Ticker
        * ISIN
        * CUSIP
        * Base Currency
        * Exchange: The exchange where the investment is listed
        * Country: The country where the investment is based
        * Security Type: Morningstar security type code for this investment
        * Fund Id
        * Performance Id


    :Examples:

    **1. Find all global listings for a security by its ticker symbol**

    .. code:: python

        import morningstar_data as md
        from morningstar_data.direct import InvestmentIdentifier

        # Define an investment identifier
        investment = InvestmentIdentifier(ticker="AAPL")

        # Retrieve matching investments
        df = md.direct.lookup.investments(investment=investment)
        df

    ==========  ==========  ======  ============  =========  ==============  ========  =======  =============  ========  ===========
    Name        SecId       Ticker  ISIN          CUSIP      Base Currency   Exchange  Country  Security Type  Fund Id   Performance Id
    ==========  ==========  ======  ============  =========  ==============  ========  =======  =============  ========  ===========
    Apple Inc   0P000000GY  AAPL    US0378331005  037833100  USD             EXXNAS    USA      ST             None      0P000000GY
    Apple Inc   0P0001367D  AAPL    US0378331005  037833100  USD             EXXSGO    USA      ST             None      0P0001367D
    ...         ...         ...     ...           ...        ...             ...       ...      ...            ...       ...
    Apple Inc   0P0001KOM0  AAPL    US0378331005  037833100  USD             EXXLIM    USA      ST             None      0P0001KOM0
    Apple Inc   0P0000EEPX  AAPL    US0378331005  037833100  EUR             EXXWBO    USA      ST             None      0P0000EEPX
    ==========  ==========  ======  ============  =========  ==============  ========  =======  =============  ========  ===========

    **2. Get listings using ticker symbol and base currency**

    .. code:: python

        import morningstar_data as md
        from morningstar_data.direct import InvestmentIdentifier

        # Define an investment identifier
        investment = InvestmentIdentifier(ticker="AAPL", base_currency="Canadian Dollar")

        # Retrieve matching investments
        df = md.direct.lookup.investments(investment=investment)
        df

    ================================================== === ============= ======== ======= === ==============
    Name                                               ... Base Currency Exchange Country ... Performance Id
    ================================================== === ============= ======== ======= === ==============
    Apple Inc Canadian Depository Receipt (CAD Hedged) ... CAD           EXXTSE   USA     ... 0P0001N6JJ
    Apple Inc                                          ... CAD           EXXNAS   USA     ... 0P000000GY
    ================================================== === ============= ======== ======= === ==============

    **3. Get listings using ISIN**

    .. code:: python

        import morningstar_data as md
        from morningstar_data.direct import InvestmentIdentifier

        # Define an investment identifier
        investment = InvestmentIdentifier(isin="US5949181045")

        # Retrieve matching investments
        df = md.direct.lookup.investments(investment=investment)
        df

    ============== ========== ====== ============ === =============
    Name           SecId      Ticker ISIN         ... PerformanceId
    ============== ========== ====== ============ === =============
    Microsoft Corp 0P000003MH MSFT   US5949181045 ... 0P000003MH
    Microsoft Corp 0P0000BNJR MSF    US5949181045 ... 0P0000BNJR
    Microsoft Corp 0P0000EGEI MSF    US5949181045 ... 0P0000EGEI
    Microsoft Corp 0P0001AJSV MSFT   US5949181045 ... 0P0001AJSV
    ============== ========== ====== ============ === =============
    """
    columns = [
        "Name",
        "SecId",
        "Ticker",
        "ISIN",
        "CUSIP",
        "Base Currency",
        "Exchange",
        "Country",
        "Security Type",
        "Fund Id",
        "Performance Id",
    ]

    if keyword != "" and investment is not None:
        raise ValueError("Only one of keyword or investment can be specified.")

    if count < 1:
        raise ValueError("Count must be a positive integer.")

    if keyword == "" and investment is not None:
        search_result = mdapi.search_security(investment, count, only_surviving)
        df = DataFrame(search_result.investments)
        return df[columns]

    # Encode keyword.
    keyword_encode = quote(keyword, "utf-8")
    _logger.debug(f"Search investments with:{keyword_encode}, Top {count} of investments to return.")
    url = f"{_config.securitydata_service_url()}v1/autocomplete?checkEntitlement=true&count={count}&keyword={keyword_encode}&onlySurviving={str(only_surviving).lower()}"
    response_json = _investment_lookup_api_request.do_get_request(url)
    investments = response_json["investments"]

    _logger.info(f"No investments return with keyword:{keyword}")
    if not investments:
        _logger.debug(f"No investments return with keyword:{keyword}")
        return DataFrame({"Name": [], "SecId": [], "ISIN": [], "CUSIP": [], "Base Currency": []})

    result = DataFrame(investments)
    # Remove some unused data.
    _logger.info("Remove some unused data")
    _logger.debug(f"{len(investments)} investments return. total investments found:{response_json['total']}")
    new_data_frame = result.drop(columns=["targetUniverseId", "csdccCode", "score", "itaCode"])
    # Rename and reoreder columns.
    _logger.info("Rename and reoreder columns. data")
    new_data_frame = new_data_frame.rename(
        columns={
            "name": "Name",
            "secId": "SecId",
            "ticker": "Ticker",
            "isin": "ISIN",
            "baseCurrency": "Base Currency",
            "cusip": "CUSIP",
            "exchange": "Exchange",
            "countryOrientation": "Country",
            "securityType": "Security Type",
            "fundId": "Fund Id",
            "performanceId": "Performance Id",
        }
    )
    return new_data_frame[columns]


@_decorator.not_null
@_decorator.typechecked
def get_morningstar_category(universe: str) -> DataFrame:
    """Returns Morningstar Categories for the given universe.

    Args:
        universe (:obj:`str`): Investment universe code. Example: "FO". Use `get_investment_universes <./lookup.html#morningstar_data.direct.lookup.get_investment_universes>`_ to discover possible values.

    Returns:
        DataFrame: A DataFrame object with Morningstar Category details. DataFrame columns include:

        * Morningstar Category
        * Category Code
        * Global Broad Category Id
        * Global Broad Category Group
        * Region

    Examples:
        Retrieve the Morningstar Category for the open-end fund universe.

    ::

        import morningstar_data as md

        df = md.direct.lookup.get_morningstar_category("FO")
        df

    :Output:
        ==========================  =============  ========================  ===========================  ==========
        Morningstar Category        Category Code  Global Broad Category Id  Global Broad Category Group  Region
        ==========================  =============  ========================  ===========================  ==========
        10 yr Government Bond	    INCA000008	   $BCG$FXINC	             Fixed Income	              India
        2025 Target Date Portfolio	CACA000158	   $BCG$ALLOC	             Allocation	                  Canada
        2030 Target Date Portfolio	CACA000164	   $BCG$ALLOC	             Allocation	                  Canada
        ...
        ==========================  =============  ========================  ===========================  ==========
    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    url = f"{_config.data_point_service_url()}v1/datapoints/datachoiceoptions?universe={universe}"

    _logger.debug(f"Lookup Morningstar Category from DataPoint API for universe: {universe}")

    # OF003 is datapoint 'Mornignstar Category' which identifies funds based on their actual investment styles as measured by their underlying portfolio holdings
    res = _data_point_lookup_api_request.do_post_request(url, "OF003")

    if not res or not res[0].get("options", []):
        return DataFrame(
            columns=["Morningstar Category", "Category Code", "Global Broad Category Id", "Global Broad Category Group", "Region"]
        )

    options = res[0].get("options", [])

    data = [
        {
            "Morningstar Category": x.get("name", ""),
            "Category Code": x.get("value", ""),
            "Global Broad Category Id": x.get("additional", {}).get("globalBroadCategoryId", ""),
            "Global Broad Category Group": x.get("additional", {}).get("globalBroadCategoryName", ""),
            "Region": x.get("additional", {}).get("regionName", ""),
        }
        for x in options
    ]

    return DataFrame(data)


@_decorator.not_null
@_decorator.typechecked
def firms(keyword: str, universe: Optional[str] = None) -> DataFrame:
    """Returns firms that match the given keyword.

    Args:
        keyword (:obj:`str`): Keyword to search for firms.
        universe (:obj:`str`, `optional`): Investment universe code. Example: "FO". Use `get_investment_universes <./lookup.html#morningstar_data.direct.lookup.get_investment_universes>`_ to discover possible values.

    :Returns:
        DataFrame: A DataFrame object with firms. DataFrame columns include:

        * Id
        * Firm Name
        * Universe


    :Examples:
        Get all firms that match the keyword "rock".

    ::

        import morningstar_data as md

        df = md.direct.lookup.firms("rock")
        df

    :Output:
        =============  =======================================  =============
        Id              Firm Name                                  Universe
        =============  =======================================  =============
        F00000Z17E      Amrock Capital B.V.                             FO
        F00000XC6D      BlackRock	                                    FO
        F00000SEJ6      BlackRock (Channel Islands) Limited	            FO
        ...
        =============  =======================================  =============

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    # Encode keyword.
    keyword_encode = quote(keyword, "utf-8")
    # Encode universe.
    if universe:
        univ_encode = quote(universe.upper(), "utf-8")
    else:
        univ_encode = ""
    _logger.debug(f"Search firms with:{keyword} and universe:{universe} ")
    # OF006 is Datapoint 'Firm Name', A company which offers mutual funds.
    url = f"{_config.securitydata_service_url()}v1/find?datapoint=OF006&findby=namecontain&kw={keyword_encode}&universe={univ_encode}"

    response_json = _investment_lookup_api_request.do_get_request(url)

    firms_df = pandas.json_normalize(response_json)
    if firms_df.empty:
        _logger.debug(f"Search firms without result. keyword:{keyword} and universe:{universe} ")
        return DataFrame({"Id": [], "Name": [], "Universe": []})
    firms_df = firms_df.rename(columns={"name": "Name", "additional.universe": "Universe", "id": "Id"})
    return firms_df


@_decorator.typechecked
def get_brandings(keyword: str, find_by: str, universe: Optional[str] = None) -> DataFrame:
    """Returns brandings for the given universe and keyword.

    Args:
        keyword(:obj:`str`): Keyword to search for brandings.
        find_by(:obj:`str`): Search condition to search for brandings. Valid values are ``name_begin`` and ``name_contain``.
        universe(:obj:`str`, `optional`): Investment universe code. Example: "FO". Use `get_investment_universes <./lookup.html#morningstar_data.direct.lookup.get_investment_universes>`_ to discover possible values.

    :Returns:
        DataFrame: A DataFrame object with brandings. DataFrame columns include:

        * Id
        * Name

    :Examples:
        Search brandings in the open-end fund universe, where branding name begins with the keyword "a".

    ::

        import morningstar_data as md

        df = direct.lookup.get_brandings("a", "name_begin", "FO")
        df

    :Output:
        ==========  ===============================
        Id            Name
        ==========  ===============================
        BN000007Q5    A Plus Finance
        BN000007Q6    A&G
        BN00000GBI    A1
        ...
        ==========  ===============================

    :Errors:
        BadRequestException: Raised when the parameter find_by is not a valid value.

        AccessDeniedError: Raised when the user is not authenticated.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.
    """

    if find_by not in ["name_begin", "name_contain"]:
        raise BadRequestException(message="Valid values for find_by should be `name_begin` or `name_contain`.")

    condition = find_by.replace("_", "")

    # FC001 is a global grouping of asset management companies that represents the management philosophy,
    # the firm's marketing/communication channels, and their distribution/sales efforts.
    url = f"{_config.securitydata_service_url()}v1/find?datapoint=FC001&findby={condition}&kw={keyword}&universe={universe}"

    res = _investment_lookup_api_request.do_get_request(url)

    if not res:
        return DataFrame(columns=["Id", "Name"])

    data = [{"Id": x.get("id", ""), "Name": x.get("name", "")} for x in res]

    return DataFrame(data)


@_decorator.typechecked
def portfolio_managers(keyword: str = "", universe: str = "FO") -> DataFrame:
    """
    Returns portfolio managers for the given universe and keyword.

    Args:
        keyword (:obj:`str`): String to match one or more manager names. If the keyword is empty or not passed to the function, the function will return all managers.

        universe (:obj:`str`): Investment universe code. Example: "FO". Use `get_investment_universes <./lookup.html#morningstar_data.direct.lookup.get_investment_universes>`_ to discover possible values.


    :Returns:
        DataFrame: A DataFrame object with portfolio managers. DataFrame columns include:

        * Id
        * Manager Name
        * Firm

    :Examples:
        Get portfolio managers that match the keyword "alla".

    ::

        import morningstar_data as md

        df = md.direct.lookup.portfolio_managers(keyword="alla")
        df

    :Output:
        ======  =======================  ===============
        Id   	Manager name	         Firm
        ======  =======================  ===============
        81643	Abdallah Guezour	     Schroders
        123653	Abdallah Nauphal	     TD
        213243	Adam Rizkalla	         River Canyon
        153647	Adrian Allardice	     Old Mutual
        182699	Adrian van Pallander	 Coronation
        ...
        ======  =======================  ===============

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    # FCM01 is Datapoint which name is 'Manager & Fund Company'
    _logger.debug(f"Search managers with: keyword: {keyword}, universe: {universe} in the FCM01 datapoint")
    data = _search_data("FCM01", keyword, universe)

    _logger.info(f"Total managers found: {len(data)} managers")
    result = DataFrame(data)

    _logger.debug("Rename columns")
    result.rename(columns={"id": "Id", "name": "Manager name"}, inplace=True)

    if result.empty:
        _logger.info(f"No managers return with keyword: {keyword}")
        return DataFrame({"Id": [], "Manager name": [], "Firm": []})

    _logger.debug("Select only the Id and Manager name in the DataFrame")
    portfolio_managers_result = result[["Id", "Manager name"]]

    _logger.debug("Apply the _separate_name_and_firm method in the columns to separate the manager name and the firm")
    portfolio_managers_result = portfolio_managers_result.apply(_separate_name_and_firm, axis=1)

    return portfolio_managers_result


# The name column has the manager's name and the firm name in the same column. This function looks for separating them into different columns
# INPUT: name column: "Abdallah Guezour (Schroders)"
# OUTPUT: name column: "Abdallah Guezour | firm column: "Schroders"
def _separate_name_and_firm(portfolio_manager: DataFrame) -> DataFrame:
    manager_name = portfolio_manager["Manager name"]
    # Check if the name does not contain the firm within (). If don't, the firm value will be None
    if manager_name.count("(") < 1:
        portfolio_manager["Firm"] = None
    # If the name contains the firm name within "()" we will separate them in different columns
    else:
        # Extract the firm name with an regex expression based on the name value
        portfolio_manager["Firm"] = re.findall("\((.+?)\)", manager_name)[0]
        # Remove signature name from column name value
        portfolio_manager["Manager name"] = manager_name.replace("(" + portfolio_manager["Firm"] + ")", "").strip()

    return portfolio_manager


@_decorator.typechecked
def companies(keyword: str = "", universe: str = "ST") -> DataFrame:
    """
    Returns companies for the given universe and keyword.

    Args:
        keyword (:obj:`str`): String to match one or more company names. If the keyword is empty or not passed to the function, the function will return all companies.

        universe (:obj:`str`): Investment universe code. Example: "FO". Use `get_investment_universes <./lookup.html#morningstar_data.direct.lookup.get_investment_universes>`_ to discover possible values.


    :Returns:
        DataFrame: A DataFrame object with companies. DataFrame columns include:

        * Id
        * Company name

    :Examples:
        Get companies that match the keyword "Energy".

    ::

        import morningstar_data as md

        df = md.direct.lookup.companies("Energy")
        df

    :Output:
        ==========  ========================
        Id	        Company name
        ==========  ========================
        0C00008O2A	11 Good Energy Inc
        0C000017RQ	2G Energy AG
        0C0000516I	3MV Energy Corp
        0C00000ZAZ	3Power Energy Group Inc
        0C00000QLB	3TEC Energy Corp
        ...
        ==========  ========================

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    # AA0A5 is Datapoint which name is 'Company Name'
    _logger.debug(f"Search companies with: keyword: {keyword}, universe: {universe} in the AA0A5 datapoint")
    data = _search_data("AA0A5", keyword, universe)

    _logger.info(f"Total companies found: {len(data)} companies")
    result = DataFrame(data)

    _logger.debug("Rename columns")
    result.rename(columns={"id": "Id", "name": "Company name"}, inplace=True)

    if result.empty:
        _logger.info(f"No companies return with keyword: {keyword}")
        return DataFrame({"Id": [], "Company name": []})

    _logger.debug("Select only the Id and Company name in the DataFrame")
    companies_result = result[["Id", "Company name"]]

    return companies_result


@_decorator.typechecked
def get_investment_universes(include_category_universes: bool = False) -> DataFrame:
    """Returns investment universe names and IDs.
    For example, Bonds, eVestment Hedge Funds, Global Restricted Funds.

    Args:
        include_category_universes (:obj:`bool`, `optional`): If True, the function will return all universes, including category universes. Default is False.

    :Returns:
        DataFrame: A DataFrame object with universes.  DataFrame columns include:

        * Id
        * Name

    :Examples:

    ::

        import morningstar_data as md

        df = md.direct.lookup.get_investment_universes()
        df

    :Output:
        ==== ==============
        Id   Name
        ==== ==============
        FH   Hedge Funds
        XI   Market_Indexes
        ...
        ==== ==============
    """

    universes = _get_all_universes()

    universes = universes.rename(columns={"id": "Id", "name": "Name"})

    if not include_category_universes:
        universes = universes[~universes["Id"].str.contains("CA]")]  # Drops all category universes

    return universes
