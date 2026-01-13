from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Dict

import simplejson as json
from pandas import DataFrame

from ._api import _config, _direct_api_request
from ._error_messages import (
    BAD_REQUEST_ERROR_PORTFOLIO_DATA_SET,
    RESOURCE_NOT_FOUND_PORTFOLIO_DATA_SET,
)
from ._exceptions import ApiResponseException, BadRequestException, ResourceNotFoundError


def _annual_return_year(year: int = 0) -> str:
    current_date = datetime.date.today()
    new_year = current_date.year - year
    return str(new_year)


def _update_annual_return_alias(data_point_id: str) -> str:
    maps = {
        "AR002": 1,
        "AR003": 2,
        "AR004": 3,
        "AR005": 4,
        "AR006": 5,
        "AR007": 6,
        "AR008": 7,
        "AR009": 8,
        "AR00A": 9,
        "AR00B": 10,
        "AR00C": 11,
    }
    map_value = maps.get(data_point_id, None)
    if map_value is not None:
        return _annual_return_year(map_value)
    else:
        return ""


def _get_data_sets() -> DataFrame:
    url = f"{_config.asset_service_url()}/portfolio/datasets"
    response_json = _direct_api_request("get", url)
    return DataFrame(response_json)


def _get_data_set_with_id(data_set_id: str) -> Any:
    try:
        if data_set_id is not None:
            url = f"{_config.asset_service_url()}portfolio/dataset/{data_set_id}"
            response_json = _direct_api_request("get", url)
    except ApiResponseException as e:
        if e.status_code == 404:
            raise ResourceNotFoundError(RESOURCE_NOT_FOUND_PORTFOLIO_DATA_SET) from None
        elif e.status_code == 400:
            raise BadRequestException(BAD_REQUEST_ERROR_PORTFOLIO_DATA_SET) from None
        raise
    response_json = _update_calendar_year_data_points_name(response_json)
    return response_json


def _update_calendar_year_data_points_name(response_json: Any) -> Any:
    data_set_id = response_json.get("id", None)
    if data_set_id is not None and PortfolioDataSet.get(str(data_set_id)) == PortfolioDataSet.Returns_Calendar_Year:
        data_points = response_json.get("datapoints", [])
        for data_point in data_points:
            new_alias = f"{data_point.get('alias', '').strip()} {_update_annual_return_alias(data_point.get('datapointId'))}"
            data_point["alias"] = new_alias
    return response_json


def _get_data_points_default_settings(data_points: list) -> Dict[str, dict]:
    url = f"{_config.asset_service_url()}portfolio/datapoints"
    post_body = list(map(lambda x: {"datapointId": x}, data_points))
    response_json = _direct_api_request("post", url, json.dumps(post_body, ignore_nan=True))
    defined_data_points = {x.get("datapointId"): x for x in response_json}
    return defined_data_points


class DataSetType(Enum):
    Current = 1
    TimeSeries = 2
    Current_Or_TimeSeries = 3
    Custom_Calculation = 4
    No_Map = 0


class PortfolioDataSet(Enum):
    def __init__(self, data_set_id: str, data_set_type: DataSetType) -> None:
        self.data_set_type = data_set_type
        self.data_set_id = data_set_id

    @classmethod
    def get(cls, data_set_id: str) -> Any:
        for member in cls._member_map_.values():
            if member._value_[0] == data_set_id:
                return member
        return PortfolioDataSet["No_Map"]

    @classmethod
    def get_data_set_type(cls, data_set_id: str) -> Any:
        for member in cls._member_map_.values():
            if member._value_[0] == data_set_id:
                return member._value_[1]
        return DataSetType["No_Map"]

    No_Map = ("no_map", DataSetType.No_Map)
    Asset_Allocation = ("99", DataSetType.Current_Or_TimeSeries)
    Custom_Calculation = ("8", DataSetType.Custom_Calculation)
    Daily_Gross_Return_Index = ("7", DataSetType.TimeSeries)
    Daily_Return_Index = ("6", DataSetType.TimeSeries)
    Equity_Country_Exposure = ("15", DataSetType.Current_Or_TimeSeries)
    Equity_Market_Capitalization = ("25", DataSetType.Current_Or_TimeSeries)
    Equity_Port_Stats_Long = ("27", DataSetType.Current_Or_TimeSeries)
    Equity_Port_Stats_Short = ("28", DataSetType.Current_Or_TimeSeries)
    Equity_Regional_Exposure = ("21", DataSetType.Current_Or_TimeSeries)
    Equity_Sector_Exposure = ("17", DataSetType.Current_Or_TimeSeries)
    Equity_Style_Analysis = ("24", DataSetType.Current_Or_TimeSeries)
    Equity_Style_Capitalization = ("22", DataSetType.Current_Or_TimeSeries)
    Equity_Style_Valuation = ("23", DataSetType.Current_Or_TimeSeries)
    Equity_Type = ("19", DataSetType.TimeSeries)
    ESG_Carbon = ("12", DataSetType.Current_Or_TimeSeries)
    ESG_Product_Involvement = ("13", DataSetType.Current_Or_TimeSeries)
    ESG_Sustainability = ("14", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_ABS_Collateral_Type_Detail = ("34", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Bond_Insurer = ("71", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Call_Type = ("35", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_CMO_Issuer_Type = ("36", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_CMO_Payment_Type = ("37", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_CMO_Tranche_Type = ("38", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Convertible_Type = ("43", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Country_Exposure = ("29", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Coupon_Type = ("44", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Credit_Enhancement_Type = ("67", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Credit_Grd_And_Credit_Rtg_Breakdown_Detail = ("77", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_CreditQualityBreakdown = ("68", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Credit_Rtg = ("76", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Current_Yield_And_Current_Yield_Breakdown_Detail = ("85", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Default_Reason = ("61", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Default_Type = ("62", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_DTW_Breakdown_Super = ("82", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_DTW_And_DTW_Breakdown_Detail = ("75", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Duration_Effective = ("39", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Duration_Modified_to_maturity = ("40", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Eff_Convxty_breakdown_Super = ("84", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_EFF_Convxty_And_Convxty_Breakdown_Detail = ("73", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Eff_Dua_Breakdown_Super = ("78", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Effective_Maturity = ("41", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Eff_Mty_Breakdown_Super = ("80", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Geographic_Exposure = ("93", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Guarantee_Type = ("63", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Guarantor_Type = ("64", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Interest_Rate_Type_Detail = ("45", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Issuer_Type = ("46", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Issuer_Type_Breakdown_Super = ("90", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_LOC_Type = ("65", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Maturity_Type = ("47", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_MBS_Agency = ("48", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Mbs_Agency_Breakdown_Super = ("87", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_MBS_Collateral_Type = ("49", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Mod_Dur_Breakdown_Super = ("79", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Morningstar_Sectors = ("94", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Municipal_Security_Type_Breakdown = ("92", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Municipal_Security_Type_Breakdown_Super = ("88", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Next_Call_And_Next_Call_Breakdown_Detail = ("86", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_OAS_Breakdown_Super = ("83", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_OAS_And_OAS_Breakdown_Detail = ("72", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_PIK_Type = ("50", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Portfolio_Statistics = ("33", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Pre_refunded_ETM_Type = ("66", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Primary_Sector_Breakdown = ("32", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Put_Type = ("51", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Put_Type_Detail = ("52", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Secondary_Sector_Breakdown = ("30", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_SEC_Registration_Type = ("53", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Secured_by_Collateral_Type = ("54", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Securitization_Collateral_Type = ("69", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Securitized_Mortgage_Type = ("55", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Security_Rank = ("56", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Security_Rank_Breakdown_Super = ("91", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Sinking_Fund_Type = ("57", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_State_Taxation = ("58", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Structured_Securitized_Type = ("59", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Super_Sector_Breakdown = ("31", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Use_of_Proceeds = ("70", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Use_Of_Proceeds_Breakdown_Super = ("89", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_US_Federal_Taxation = ("60", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_Yield_to_Maturity_YTM = ("42", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_YTM_Breakdown_Super = ("81", DataSetType.Current_Or_TimeSeries)
    Fixed_Income_YTW_And_YTW_Breakdown_Detail = ("74", DataSetType.Current_Or_TimeSeries)
    GICS_Equity_Industry = ("20", DataSetType.Current_Or_TimeSeries)
    GICS_Equity_Sector_Exposure = ("18", DataSetType.Current_Or_TimeSeries)
    MSCI_Equity_Country_Exposure = ("16", DataSetType.Current_Or_TimeSeries)
    Post_tax_Returns_Month_end = ("9", DataSetType.Current_Or_TimeSeries)
    Regional_Asset_Class_Market_Cap = ("95", DataSetType.Current)
    Representative_Cost = ("96", DataSetType.Current_Or_TimeSeries)
    Returns_Calendar_Year = ("5", DataSetType.Current)
    Returns_Daily = ("2", DataSetType.Current)
    Returns_Month_End = ("3", DataSetType.Current)
    Returns_Quarter_End = ("4", DataSetType.Current)
    Risk_Total_ReturnMonth_End = ("10", DataSetType.Current_Or_TimeSeries)
    Risk_Total_ReturnQtr_End = ("11", DataSetType.Current_Or_TimeSeries)
    Snapshot = ("1", DataSetType.Current)
    South_Africa_Asset_Allocation = ("101", DataSetType.Current_Or_TimeSeries)
    Stock_Intersection = ("26", DataSetType.Current_Or_TimeSeries)
    Surveyed_Asset_Allocation_AUS = ("97", DataSetType.Current_Or_TimeSeries)
    Surveyed_Asset_Allocation_CAN = ("98", DataSetType.Current_Or_TimeSeries)
    Surveyed_Asset_Allocation_NZ = ("100", DataSetType.Current_Or_TimeSeries)
