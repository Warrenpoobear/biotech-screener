import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pandas import DataFrame

from .. import mdapi
from .._utils import (
    _get_data_points_total_columns,
    _get_user_cells_quota,
    data_point_dataframe_to_list,
)
from ..mdapi import RequestObject
from . import _decorator
from ._data_objects import DataPoints, Investments
from ._data_type import DryRunResults
from .data_type import InvestmentIdentifier, TimeSeriesFormat, WarningBehavior

NUMBER_OF_INVESTMENTIDENTIFIERS_ALLOWED = 500


@dataclass
class InvestmentDataRequest(RequestObject):
    investments: Union[List[str], str, Dict[str, Any], List["InvestmentIdentifier"]]
    datapoints: Optional[Union[List[Dict[str, Any]], str, List[Any]]] = None
    time_series_format: TimeSeriesFormat = TimeSeriesFormat.WIDE
    display_name: Optional[bool] = False
    unmatched_investment_behavior: Optional[WarningBehavior] = WarningBehavior.WARNING
    preview: Optional[bool] = False

    def check_size_of_request(self) -> None:
        if isinstance(self.investments, list) and len(self.investments) > NUMBER_OF_INVESTMENTIDENTIFIERS_ALLOWED:
            if all(isinstance(item, InvestmentIdentifier) for item in self.investments):
                raise ValueError(
                    f"The size of investments list should not be greater than {NUMBER_OF_INVESTMENTIDENTIFIERS_ALLOWED} if of type List[InvestmentIdentifier]"
                )


@_decorator.typechecked
def investment_data(
    investments: Union[List[str], str, Dict[str, Any]],
    datapoints: Union[List[Dict[str, Any]], str, DataFrame],
) -> DataFrame:
    warnings.warn(
        "The investment_data function is deprecated and will be removed in the next major version. Use get_investment_data instead",
        FutureWarning,
        stacklevel=2,
    )
    return _get_investment_data(investments, datapoints, time_series_format=TimeSeriesFormat.WIDE)


# FIXME: In https://msjira.morningstar.com/browse/AL2-92, the error_handler decorator was removed from some methods
# to trigger QueryLimitException.
# There are multiple methods where the decorator was removed to trigger certain exceptions back to the caller:
#   - save_investment_list
#   - investment_data
#   - holding_dates
#   - asset_flow
# For the time being, we will leave this as-is to keep things working. Moving forward, we must re-design the
# exception handling so that we don't have these types of special cases where error_handler is only allowed on some
# methods but not others.
@_decorator.typechecked
def get_investment_data(
    investments: Union[List[str], str, Dict[str, Any], List[InvestmentIdentifier]],
    data_points: Optional[Union[List[Dict[str, Any]], str, DataFrame, List[Any]]] = None,
    display_name: bool = False,
    dry_run: Optional[bool] = False,
    time_series_format: TimeSeriesFormat = TimeSeriesFormat.WIDE,
    unmatched_investment_behavior: Optional[WarningBehavior] = WarningBehavior.WARNING,
    preview: Optional[bool] = False,
) -> Union[DataFrame, DryRunResults]:
    """Retrieve data for the specified investments and data points.

    Args:
        investments (:obj:`Union`, `required`): Defines the investments to fetch. Input can be:

            * Investment IDs (:obj:`list`, `optional`): Investment identifiers, in the format of SecId;Universe or just SecId. E.g., ["F00000YOOK;FO","FOUSA00CFV;FO"] or ["F00000YOOK","FOUSA00CFV"]. Use the `investments <./lookup.html#morningstar_data.direct.investments>`_ function to discover identifiers.
            * InvestmentIdentifiers (:obj:`list`, `optional`): A list of :class:`~morningstar_data.direct.InvestmentIdentifier` objects, allowing users to specify investments using standard identifiers such as ISIN, CUSIP, and/or ticker symbol instead of Morningstar SecIds. Introduced in version 1.11.0.

              * Multiple Matches: If multiple valid matches exist for a single InvestmentIdentifier, results will be prioritized according to the :ref:`security matching logic<Security Matching Logic>`. The highest ranked security will be used.
              * Request Limit: Supports up to 500 InvestmentIdentifier objects per request.

            * Investment List ID (:obj:`str`, `optional`): Saved investment list in Morningstar Direct. Currently, this function does not support lists that combine investments and user-created portfolios. Use the `get_investment_lists <./lists.html#morningstar_data.direct.user_items.get_investment_lists>`_ function to discover saved lists.
            * Search Criteria  ID (:obj:`str`, `optional`): Saved search criteria in Morningstar Direct. Use the `get_search_criteria <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria>`_ function to discover saved search criteria.
            * Search Criteria Condition (:obj:`dict`, `optional`): Search criteria definition. See details in the Reference section below or use the `get_search_criteria_conditions <./search_criteria.html#morningstar_data.direct.user_items.get_search_criteria_conditions>`_ function to discover the definition of a saved search criteria.

        data_points (:obj:`Union`, `optional`): Defines the data points to fetch. If not provided and investments are specified with a list ID or search criteria ID, the corresponding bound dataset will be used.

            * Data Point IDs (:obj:`List[Dict]`, `optional`): A list of dictionaries, each defining a data point and its (optional) associated settings.
            * Data Set ID (:obj:`str`, `optional`): Morningstar data set or user-created data set saved in Morningstar Direct. Use the `get_data_sets <./data_set.html#morningstar_data.direct.user_items.get_data_sets>`_ or `get_morningstar_data_sets <./data_set.html#morningstar_data.direct.get_morningstar_data_sets>`_ functions to discover saved data sets.
            * Data Point Settings (:obj:`DataFrame`, `optional`): A DataFrame of data point identifiers and their associated settings. Use the `get_data_set_details <./data_set.html#morningstar_data.direct.user_items.get_data_set_details>`_ function to discover data point settings from a saved data set.

        display_name (:obj:`bool`, `optional`): When true, the returned column names will match display names saved in the data set. Default is false.
        time_series_format (:obj:`direct.TimeSeriesFormat`, `optional`): Specifies the format of the time series data. Default is WIDE. Accepted values are:

            * ``TimeSeriesFormat.WIDE``: Data is presented in a wide format, where each row contains a different investment with values for each point in time in a different column.
            * ``TimeSeriesFormat.LONG``: Data is presented in a long format, where each row represents a single observation for a variable at a point in time.
            * ``TimeSeriesFormat.LONG_WITHOUT_NAME``: Similar to the long format but excludes the 'Name' column.

        dry_run(:obj:`bool`, `optional`): When True, the query will not be executed. Instead, a DryRunResults object will be returned with details about the query's impact on daily cell limit usage.
        unmatched_investment_behavior (:obj:`WarningBehavior`, `optional`): Determines behavior when any of the input investments can't be found.

            * ``WarningBehavior.FAIL``: A `ResourceNotFoundError` is raised if any of the investments can't be found.
            * ``WarningBehavior.WARNING`` (default): A warning is displayed listing any investments that were not found. A `ResourceNotFoundError` is raised if all investments were not found.
            * ``WarningBehavior.IGNORE``: No warning is displayed if any investments were not found. A `ResourceNotFoundError` is raised if all investments were not found.

        preview (:obj:`bool`, `optional`): Defaults to False. Setting to True allows access to data points outside of your current subscription, but limits the output to 25 rows.

    :Returns:

        There are two return types:

        * DataFrame: A DataFrame object with investment data

        * DryRunResults: Is returned if dry_run=True is passed

          * estimated_cells_used: Number of cells by this query
          * daily_cells_remaining_before: How many cells are remaining in your daily cell limit before running this query
          * daily_cells_remaining_after: How many cells would be remaining in your daily cell limit after running this query
          * daily_cell_limit: Your total daily cell limit

    :Reference:
        Constructing a Search Criteria Condition dictionary:

        For example::

                    SEARCH_CRITERIA_CONDITION = {
                            "universeId": "cz",
                            "subUniverseId": "",
                            "subUniverseName": "",
                            "securityStatus": "activeonly",
                            "useDefinedPrimary": False,
                            "criteria": [
                                {"relation": "", "field": "HU338", "operator": "=", "value": "1"},
                                {"relation": "AND", "field": "HU863", "operator": "=", "value": "1"}
                            ]
                        }

        * universeId (:obj:`string`, `required`): Universe code (e.g., "FO"). Use the `get_investment_universes <./lookup.html#morningstar_data.direct.lookup.get_investment_universes>`_ function to explore available values or `download <../_static/assets/SearchCriteriaInfo.xlsx>`_ the reference file.
        * subUniverseId (:obj:`string`, `optional`): Sub-universe code, if applicable. See the refererece file above for available values.
        * subUniverseName (:obj:`string`, `optional`): Name of sub-universe, if applicable. See the refererece file above for available values.
        * securityStatus (:obj:`string`, `optional`): Security status, can be 'activeonly' or 'activeinactive'.
        * useDefinedPrimary (:obj:`Boolean`, `optional`): If set to true, Morningstar Direct user-defined settings will be used.
        * criteria (:obj:`List[Dict[]]`, `required`): Custom search conditions. Dictionary fields described below.


          * field (:obj:`string`): Data point identifier
          * value (:obj:`string`): Value to compare against the data point.
          * criteria (:obj:`List[Dict[]]`): A nested list of additional custom search conditions.
          * relation (:obj:`string`, `required`): Boolean condition used when joining with the previous condition. Accepts an empty string (for the first condition, as no previous condition exists to join with), 'Or', or 'And'.
          * operator (:obj:`string`): Operator used to compare field value to value. Possible operators include '=', '!=', '<', '>', '<=', '>=', 'like' (data contains value), and 'not like' (data does not contain value).

            For example::

                [
                    {
                        "field": "eb919bcc-c097-4fe3-898c-470d8b89dde1"
                        "operator": "="
                        "relation": ""
                        "value": "122"
                    },
                    {
                        "criteria": [
                            {...Condition1 },
                            {...Condition2 },
                            {...Condition3 },
                        ],
                        "relation": "And"
                    },
                    {
                        "field": "LS466"
                        "operator": "="
                        "relation": "Or"
                        "value": "FH"
                    }
                ]

    ----------------------
    Usage Examples
    ----------------------

    **Get investment data for the given investments and data points.**
    ::

        import morningstar_data as md

        df = md.direct.get_investment_data(
            investments=["F0AUS05U7H", "F000010NJ5"],
            data_points=[
                {"datapointId": "OS01W"}, # Name
                {"datapointId": "LS05M"}, # Base Currency
                {
                    "datapointId": "HP010", # Monthly Return
                    "isTsdp": True,
                    "currency": "CNY",
                    "startDate": "2021-03-01",
                    "endDate": "2021-08-31",
                },
            ],
        )
        df

    :Output:
        =================  ===================================  ================= =========================  =========================
        Id                 Name                                 Base Currency     Monthly Return 2021-03-31  Monthly Return 2021-04-30
        =================  ===================================  ================= =========================  =========================
        F0AUS05U7H         Walter Scott Global Equity           Australian Dollar                  3.726094                   3.078352
        F000010NJ5         Vontobel Emerging Markets Eq U1 USD  US Dollar                         -0.417526                  -0.376890
        =================  ===================================  ================= =========================  =========================


    **Get investment data for a saved investment list and data points.**

    ::

        import morningstar_data as md

        df = md.direct.get_investment_data(
            investments="a727113a-9557-4378-924f-5d2ba553f687", # Replace with a valid List ID
            data_points=[{"datapointId": "HS793", "isTsdp": True}],
        )
        df


    ==============  =================================  =============================  =============================  =============================
    Id              Name                               Daily Return Index 2021-09-23  Daily Return Index 2021-09-24  Daily Return Index 2021-09-25
    ==============  =================================  =============================  =============================  =============================
    FOUSA00DFS;FO   BlackRock Global Allocation Inv A  129.92672                      129.56781                      129.56781
    ==============  =================================  =============================  =============================  =============================

    **Get investment data for a saved search criteria and data points.**

    ::

        import morningstar_data as md

        df = md.direct.get_investment_data(
            investments="4216254", # Replace with a valid Search Criteria ID
            data_points=[{"datapointId": "12", "isTsdp": True}]
        )
        df


    ==============  =======================  =============================
    Id              Name                     Beta 2018-10-01 to 2021-09-30
    ==============  =======================  =============================
    FOUSA06JNH;FO   DWS RREEF Real Assets A    0.654343
    ==============  =======================  =============================

    **Get investment data for a custom search criteria and data points.**

    ::

        import morningstar_data as md

        SEARCH_CRITERIA_CONDITION = {
            "universeId": "cz",
            "subUniverseId": "",
            "subUniverseName": "",
            "securityStatus": "activeonly",
            "useDefinedPrimary": False,
            "criteria": [
                {"relation": "", "field": "HU338", "operator": "=", "value": "1"},
                {"relation": "AND", "field": "HU863", "operator": "=", "value": "1"},
            ],
        }

        df = md.direct.get_investment_data(
            investments=SEARCH_CRITERIA_CONDITION,
            data_points=[{"datapointId": "HS793", "isTsdp": True}],
        )
        df


    =  ===============  =====================================  =============================  ===  =============================
    #  Id               Name                                   Daily Return Index 2022-02-18  ...  Daily Return Index 2022-03-17
    =  ===============  =====================================  =============================  ===  =============================
    0  FOUSA06UOR;CZ    Columbia Trust Stable Government Fund  None                           ...  None
    1  FOUSA06UWL;CZ    Columbia Trust Stable Income Fund      88.8333                        ...  90.7781
    =  ===============  =====================================  =============================  ===  =============================

    **Get investment data for the given investments and data points in the long format.**
    ::

        import morningstar_data as md

        df = md.direct.get_investment_data(
            investments=["F0AUS05U7H", "F000010NJ5"],
            data_points=[
                {"datapointId": "OS01W"}, # Name
                {"datapointId": "LS05M"}, # Base Currency
                {
                    "datapointId": "HP010", # Monthly Return
                    "isTsdp": True,
                    "currency": "CNY",
                    "startDate": "2021-03-01",
                    "endDate": "2021-08-31",
                },
            ],
            time_series_format=md.direct.data_type.TimeSeriesFormat.LONG
        )
        df


    =================  ===================================  ================= =========================  =========================
    Id                 Name                                 Base Currency     Date                       Monthly Return
    =================  ===================================  ================= =========================  =========================
    F0AUS05U7H         Walter Scott Global Equity           Australian Dollar                2021-04-30                   3.726094
    F0AUS05U7H         Walter Scott Global Equity           Australian Dollar                2021-05-31                   3.078352
    F0AUS05U7H         Walter Scott Global Equity           Australian Dollar                2021-06-30                   3.078352
    F000010NJ5         Vontobel Emerging Markets Eq U1 USD  US Dollar                        2021-04-30                  -0.417526
    F000010NJ5         Vontobel Emerging Markets Eq U1 USD  US Dollar                        2021-05-31                  -0.376890
    F000010NJ5         Vontobel Emerging Markets Eq U1 USD  US Dollar                        2021-06-30                  3.078352
    =================  ===================================  ================= =========================  =========================

    **Using Ticker to retrieve data for Apple (AAPL).**
    ::

        import morningstar_data as md
        from morningstar_data.direct import InvestmentIdentifier

        # Define the data point IDs to retrieve
        data_point_ids = [
            {"datapointId": "LS05M"}, # Base Currency
            {"datapointId": "LS01Z"}, # Exchange by Name
            {"datapointId": "OS00F"},  # Inception Date
            {"datapointId": "LS017"}   # Domicile
        ]

       # Initialize an InvestmentIdentifier object for the ticker symbol
       investments = [InvestmentIdentifier(ticker="AAPL")]

       # Retrieve investment data using the defined ticker and data points
       df = md.direct.get_investment_data(investments, data_point_ids)

       df

    ============ =========== =============== ====================== ================ ==============
    Id           Name        Base Currency   Exchange               Inception Date   Domicile
    ============ =========== =============== ====================== ================ ==============
    0P000000GY   Apple Inc   US Dollar       Nasdaq - All Markets   1980-12-12       United States
    ============ =========== =============== ====================== ================ ==============


    **Using Ticker and Currency to retrieve data for Apple (AAPL).**
    ::

        import morningstar_data as md
        from morningstar_data.direct import InvestmentIdentifier

        # Define the data point IDs to retrieve
        data_point_ids = [
            {"datapointId": "LS05M"}, # Base Currency
            {"datapointId": "LS01Z"}, # Exchange by Name
            {"datapointId": "OS00F"},  # Inception Date
            {"datapointId": "LS017"}   # Domicile
        ]

        # Example: Using Ticker and Base Currency
        investments = [InvestmentIdentifier(ticker="AAPL", base_currency="Euro")]

        # Retrieve the data via ticker
        df = md.direct.get_investment_data(investments, data_point_ids)

        df

    ============ =========== =============== ==================== ================ ==============
    Id           Name        Base Currency   Exchange             Inception Date   Domicile
    ============ =========== =============== ==================== ================ ==============
    0P0000EEPX   Apple Inc   Euro            Wiener Boerse AG     2017-05-23       United States
    ============ =========== =============== ==================== ================ ==============


    **Retrieving Investment Data Using Ticker and Exchange.**
    ::

        import morningstar_data as md
        from morningstar_data.direct import InvestmentIdentifier

        # Define the data point IDs to retrieve
        data_point_ids = [
            {"datapointId": "LS05M"}, # Base Currency
            {"datapointId": "LS01Z"}, # Exchange by Name
            {"datapointId": "OS00F"},  # Inception Date
            {"datapointId": "LS017"}   # Domicile
        ]

        # Example: Using Ticker and Exchange
        investments = [InvestmentIdentifier(ticker="MSFT", exchange="Toronto Stock Exchange")]

        # Retrieve the data via Ticker and Exchange
        df = md.direct.get_investment_data(investments, data_point_ids)

        df

    ============ ============================================ ================== ======================== ================ ===============
    Id           Name                                         Base Currency      Exchange                 Inception Date   Domicile
    ============ ============================================ ================== ======================== ================ ===============
    0P0001NHUC   Microsoft Corp Canadian Depository Receipt   Canadian Dollar	   Toronto Stock Exchange   2021-10-05       United States
    ============ ============================================ ================== ======================== ================ ===============

    **Retrieving Investment Data Using ISINs and CUSIPs.**
    ::

        import morningstar_data as md
        from morningstar_data.direct import InvestmentIdentifier

        # Defining Datapoints
        data_point_ids = [
            {"datapointId": "LS05M"},  # Base Currency
            {"datapointId": "LS01Z"},  # Exchange by Name
            {"datapointId": "LF035"},  # Global Broad Category Group
            {"datapointId": "ZZ006"},  # Annual Expense Ratio
        ]

        # Example: Using ISINs and CUSIPs
        investments = [
            InvestmentIdentifier(isin="US6176971074"),  # ISIN for the Morningstar US Equity Fund
            InvestmentIdentifier(cusip="453320103"),    # CUSIP for the Income Fund of America
        ]

        # Retrieve the data via ISIN and CUSIP
        df = md.direct.get_investment_data(investments, data_point_ids)

        df

    === ============= ======================================= =============== ======================= ============================= ======================================
    #   Id            Name                                    Base Currency   Exchange                Global Broad Category Group   Annual Report Adjusted Expense Ratio
    === ============= ======================================= =============== ======================= ============================= ======================================
    1   F00000YOOK    Morningstar US Equity                   US Dollar       Nasdaq - All Markets    Equity                        0.84
    2   FOUSA00D6E    American Funds Income Fund of Amer A    US Dollar       Nasdaq - All Markets    Allocation                    0.58
    === ============= ======================================= =============== ======================= ============================= ======================================


    **Bulk Processing via Identifier Lists.**
    ::

        import morningstar_data as md
        from morningstar_data.direct import InvestmentIdentifier
        import pandas as pd

        # Defining Data Points
        data_point_ids = [
            {"datapointId": "LS05M"},  # Base Currency
            {"datapointId": "LS01Z"},  # Exchange by Name
            {"datapointId": "LF035"},  # Global Broad Category Group
            {"datapointId": "ZZ006"},  # Annual Expense Ratio
        ]

        # Read the list of ISINs from securities.csv
        df_securities = pd.read_csv("securities.csv")

        # Extract the list of ISINs from the DataFrame
        isin_list = df_securities['ISIN'].tolist()

        # Create InvestmentIdentifier objects using the ISINs list
        investments_isins = [InvestmentIdentifier(isin=isin) for isin in isin_list]

        # Retrieve the securities and the associated data_point_ids simply using ISINs
        df = md.direct.get_investment_data(
            investments=investments_isins,
            data_points=data_point_ids
        )

        df

    === ============= ===================================== =============== ======================= ============================= ======================================
    #   Id            Name                                  Base Currency   Exchange                Global Broad Category Group   Annual Report Adjusted Expense Ratio
    === ============= ===================================== =============== ======================= ============================= ======================================
    1   FOUSA00K3V    AB Concentrated Growth Advisor        US Dollar       Nasdaq - All Markets    Equity                        0.75
    2   FOUSA00CAQ    AB Growth A                           US Dollar       Nasdaq - All Markets    Equity                        1.09
    3   FOUSA00B92    AB Large Cap Growth A                 US Dollar       Nasdaq - All Markets    Equity                        0.81
    4   F00000YDG4    AB Sustainable US Thematic Advisor    US Dollar       Nasdaq - All Markets    Equity                        0.65
    5   FOUSA00LBE    abrdn US Sustainable Leaders A        US Dollar       Nasdaq - All Markets    Equity                        1.19
    === ============= ===================================== =============== ======================= ============================= ======================================



    .. Note::
        ``InvestmentSearchWarning:`` This warning is raised when the get_investment_data function is unable to find investments for all provided fields in InvestmentIdentifier objects. If you encounter this warning, review your identifiers to ensure they are valid and correctly specified.

    """

    if len(investments) == 0:
        raise ValueError("Please specify at least one investment")

    result = _get_investment_data(
        investments=investments,
        data_points=data_points,
        display_name=display_name,
        time_series_format=time_series_format,
        dry_run=dry_run,
        unmatched_investment_behavior=unmatched_investment_behavior,
        preview=preview,
    )

    return result


@_decorator.typechecked
def _get_investment_data(
    investments: Union[List[str], str, Dict[str, Any], List[InvestmentIdentifier]],
    data_points: Optional[Union[List[Dict[str, Any]], str, DataFrame, List[Any]]] = None,
    display_name: bool = False,
    dry_run: Optional[bool] = False,
    time_series_format: TimeSeriesFormat = TimeSeriesFormat.WIDE,
    unmatched_investment_behavior: Optional[WarningBehavior] = WarningBehavior.WARNING,
    preview: Optional[bool] = False,
) -> Union[DataFrame, DryRunResults]:
    if dry_run:
        investment_param = Investments(investments)
        data_point_param = DataPoints(data_points)
        investment_id_list = investment_param.get_investment_ids()
        data_point_settings = data_point_param.get_data_points()
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

    if isinstance(data_points, DataFrame):
        data_points = data_point_dataframe_to_list(data_points)

    investment_data_request = InvestmentDataRequest(
        investments=investments,
        datapoints=data_points,
        time_series_format=time_series_format,
        display_name=display_name,
        unmatched_investment_behavior=unmatched_investment_behavior,
        preview=preview,
    )

    investment_data_request.check_size_of_request()

    return mdapi.call_remote_function(
        "get_investment_data",
        investment_data_request,
    )
