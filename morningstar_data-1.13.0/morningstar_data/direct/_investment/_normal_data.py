import functools
import os
from typing import Any, Dict, List, Optional

import simplejson as json
from pandas import DataFrame

from ...direct._api import _direct_api_request
from .._config import _Config
from .._config_key import ALL_ASSET_FLOW_DATA_POINTS
from .._data_objects import DataPoints, Investments
from .._exceptions import BadRequestException
from ._common import _get_data_points, _get_investment_ids
from ._data import Column, InvestmentDataRequest, InvestmentDataResults

_config = _Config()


class NormalDataProvider:
    @staticmethod
    def build_request(
        investment_object: Investments, data_point_object: DataPoints, display_name: bool = False
    ) -> InvestmentDataRequest:
        investment_id_list = _get_investment_ids(investment_object)
        data_points = _get_data_points(investment_object, data_point_object, display_name)
        normal_data_points = _filter_out_data_points_by_id(data_points, filter_out_list=ALL_ASSET_FLOW_DATA_POINTS)
        return InvestmentDataRequest(investment_id_list, normal_data_points)

    @staticmethod
    def run_request(req: InvestmentDataRequest) -> InvestmentDataResults:
        raw_data = _get_normal_data(req.investment_ids, req.data_points)
        return _parse_raw_normal_data_values(raw_data)


def _filter_out_data_points_by_id(df: DataFrame, filter_out_list: list) -> DataFrame:
    return df[~df["datapointId"].isin(filter_out_list)].reset_index().drop(["index"], axis=1)


def _get_normal_data(investment_ids: list, normal_data_point_settings: DataFrame) -> dict:
    normal_resp: dict = dict()
    if normal_data_point_settings is not None and not normal_data_point_settings.empty:
        if not investment_ids:
            raise BadRequestException("No investments.")
        data_point_list = normal_data_point_settings.to_dict(orient="records")
        postbody = {
            "datapoints": data_point_list,
            "investments": list(map(lambda x: {"id": x}, investment_ids)),
        }
        resp = _request_investment_data(postbody)
        if resp and isinstance(resp, dict):
            normal_resp.update(resp)

        updated_data_point_list = []

        for data_point, normal_data_point in zip(data_point_list, normal_resp["datapoints"]):
            use_display_name = "displayName" in normal_data_point_settings.columns and data_point["displayName"]

            if use_display_name:
                name = data_point["displayName"]
            else:
                name = normal_data_point["name"]

            data_point_entry = {**normal_data_point, "name": name}
            updated_data_point_list.append(data_point_entry)

        normal_resp["datapoints"] = updated_data_point_list

    return normal_resp


def _request_investment_data(params: dict) -> Dict[str, Any]:
    if os.getenv("FEED_ID"):
        url = _config.investment_data_service_url()
    else:
        url = f"{_config.securitydata_service_url()}v1/data?includeAdditionalDps=false"

    response_json: Dict[str, Any] = _direct_api_request("POST", url, data=json.dumps(params, ignore_nan=True))
    return response_json


def _parse_raw_normal_data_values(response_json: dict) -> InvestmentDataResults:
    investment_results = InvestmentDataResults()
    data_points = response_json.get("datapoints", [])
    for inv in response_json.get("investments", []):
        investment_id = inv.get("id", "")
        if _with_entitled(inv) is False:
            investment_results.add_meta_data(investment_id, "entitled", False)
            continue
        value_map = dict(list(map(lambda x: (x.get("alias", ""), x), inv.get("values", []))))
        for dp in data_points:
            alias = dp.get("alias", "")
            value = value_map.get(alias, None)
            investment_results.add_meta_data(investment_id, alias, dp)
            add_column_data = functools.partial(investment_results.add_column_data, investment_id, alias)

            if _with_entitled(value) is False:
                add_column_data(_convert_data_point_to_column(dp, None))
                continue

            if dp.get("isTsdp", False):
                add_column_data(_convert_ts_data_point_to_columns(dp, value))
                continue

            if "MULTIPLE" == dp.get("nonstandardDisplayType", ""):
                add_column_data(_convert_multiple_display_type_data_point_to_column(dp, value))
            else:  # current data_point
                add_column_data(_convert_data_point_to_column(dp, value))
    return investment_results


def _convert_data_point_to_column(data_point: dict, value: Optional[dict]) -> list:
    # { "alias": "HS05A_4", "value": "3", "text": "Large Growth" }
    data_point_name = data_point.get("name", "")
    if value is None or not isinstance(value, dict):
        return [Column(name=data_point_name, value=value)]
    if _is_mstar_ip_data_point(data_point):
        return [
            Column(name=data_point_name, value=value.get("value")),
            Column(name=_get_mstar_ip_data_point_text_column_name(data_point), value=value.get("text")),
        ]
    else:
        return [Column(name=data_point_name, value=value.get("text", value.get("value")))]


def _convert_ts_data_point_to_columns(data_point: dict, value: dict) -> list:
    # { "alias": "HS05A_3", "value": [ { "index": 0, "text": "Large Growth", "value": "3" }, { "index": 1, "text": "Large Growth", "value": "3" } ] }
    column_data: list = list()
    value_map = dict(list(map(lambda x: (x.get("index", ""), x), value.get("value", [])))) if isinstance(value, dict) else dict()
    for i, tp in enumerate(data_point.get("timePeriods", [])):
        value_dict: dict = value_map.get(i, dict())
        if _with_entitled(value_dict) is False:
            value_dict = dict()
        if _is_mstar_ip_data_point(data_point):
            column_data.append(Column(name=_get_ts_column_name(data_point, _get_date(tp)), value=value_dict.get("value")))
            column_data.append(
                Column(name=_get_mstar_ip_data_point_text_column_name(data_point, _get_date(tp)), value=value_dict.get("text"))
            )
        else:
            column_data.append(
                Column(name=_get_ts_column_name(data_point, _get_date(tp)), value=value_dict.get("text", value_dict.get("value")))
            )
    return column_data


def _convert_multiple_display_type_data_point_to_column(data_point: dict, value: dict) -> list:
    #  { "alias": "HS05A_3", "value": [{ "value": "[1996-11-01 -- ] William C. Nygren" },{ "value": "[2000-03-21 -- 2012-08-01] Henry Berghoef" }] }
    data_point_name = data_point.get("name", "")
    val: Optional[List[Dict[str, Any]]] = value.get("value") if isinstance(value, dict) else value
    if isinstance(val, list):
        v = [x["value"] for x in val]
        values: Optional[List[Any]] = v if len(v) > 0 else None
        return [Column(name=data_point_name, value=values)]
    else:
        return [Column(name=data_point_name, value=val)]


def _is_mstar_ip_data_point(data_point: dict) -> bool:
    mstar_ip_type = data_point.get("mstarIpType", "") if data_point is not None else ""
    return mstar_ip_type is not None and len(mstar_ip_type) > 0


def _get_date(timeperiod: dict) -> str:
    start_date = timeperiod.get("startDate", "")
    end_date = timeperiod.get("endDate", "")
    return f"{start_date} to {end_date}" if len(start_date) > 0 else f"{end_date}"


def _get_ts_column_name(data_point: dict, date: str) -> str:
    data_point_name = _get_display_name(data_point)
    return f"{data_point_name} {date}"


def _get_mstar_ip_data_point_text_column_name(data_point: dict, date: Optional[str] = None) -> Any:
    resp = _get_display_name(data_point) + " - display text"
    if date is not None and len(date) > 0:
        resp = resp + " " + date
    return resp


def _get_display_name(data_point: dict) -> str:
    if data_point.get("displayName") is not None:
        return str(data_point.get("displayName"))
    return data_point.get("name", "")


def _with_entitled(data: dict) -> bool:
    # NBI-523: Rollback NBI-433
    # if data:
    #     entitled = data.get("entitled", True)
    #     is_allowed_to_be_exported = data.get("isAllowedToBeExported", True)
    #     if entitled is False or is_allowed_to_be_exported is False:
    #         return False
    return True
