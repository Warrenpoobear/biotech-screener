import simplejson as json
from pandas import DataFrame

from .._config import _Config
from .._config_key import ALL_ASSET_FLOW_DATA_POINTS
from .._data_objects import DataPoints, Investments
from ..asset_flow import _asset_flow_api_request
from ._common import _get_data_points, _get_investment_ids
from ._data import Column, InvestmentDataRequest, InvestmentDataResults

_config = _Config()


class AssetFlowProvider:
    @staticmethod
    def build_request(
        investment_object: Investments, data_point_object: DataPoints, display_name: bool = False
    ) -> InvestmentDataRequest:
        investment_id_list = _get_investment_ids(investment_object)
        data_points = _get_data_points(investment_object, data_point_object, display_name)
        asset_flow_data_points = _filter_data_points_by_id(data_points, filter_list=ALL_ASSET_FLOW_DATA_POINTS)
        return InvestmentDataRequest(investment_id_list, asset_flow_data_points)

    @staticmethod
    def run_request(req: InvestmentDataRequest) -> InvestmentDataResults:
        return _get_asset_flow_data(req.investment_ids, req.data_points)


def _filter_data_points_by_id(df: DataFrame, filter_list: list) -> DataFrame:
    return df[df["datapointId"].isin(filter_list)].reset_index().drop(["index"], axis=1)


def _get_asset_flow_data(investment_ids: list, asset_flow_data_point_settings: DataFrame) -> InvestmentDataResults:
    investment_results = InvestmentDataResults()
    if asset_flow_data_point_settings is not None and not asset_flow_data_point_settings.empty:
        if "marketId" not in asset_flow_data_point_settings.columns:
            asset_flow_data_point_settings["marketId"] = None
        data_point_list = asset_flow_data_point_settings.to_dict(orient="records")
        market_data_point_dict = _group_by_market(data_point_list)

        for market_id, data_points in market_data_point_dict.items():
            url = f"{_config.securitydata_service_url()}v1/assetflow/data"
            postbody = {"marketId": market_id, "datapoints": data_points}
            if investment_ids:
                postbody["investments"] = list(map(lambda x: {"id": _get_sec_id(x)}, investment_ids))
            sub_investment_results = _parse_raw_asset_flow_data_values(
                investment_ids,
                data_points,
                _asset_flow_api_request.do_post_request(url, json.dumps(postbody, ignore_nan=True)),
            )
            investment_results.merge_with(sub_investment_results, in_place=True)
    return investment_results


def _group_by_market(data_point_list: list) -> dict:
    market_data_point_dict: dict = dict()
    for data_point in data_point_list:
        market_id = data_point.get("marketId", None)
        if market_id is not None and len(market_id.strip()) > 0:
            market_data_points = market_data_point_dict.get(market_id, [])
            market_data_points.append(data_point)
            market_data_point_dict[market_id] = market_data_points
    return market_data_point_dict


def _parse_raw_asset_flow_data_values(investment_ids: list, data_points: list, response_json: list) -> InvestmentDataResults:
    sub_investment_results = InvestmentDataResults()
    if not response_json or not isinstance(response_json, list):
        return sub_investment_results
    sec_id_to_investment_id = {_get_sec_id(x): x for x in investment_ids}
    alias_name_dict = {x.get("alias", ""): x.get("datapointName", "") for x in data_points}
    data_points_dict = {x.get("alias", ""): {"datapointName": x.get("datapointName", ""), "isTsdp": True} for x in data_points}

    for investment in response_json:
        sec_id = investment.get("id", "")
        investment_id = sec_id_to_investment_id[sec_id]
        values = investment.get("values", [])
        for data_point_value in values:
            alias = data_point_value.get("alias", "").strip()
            value_list = data_point_value.get("value", [])
            column_data = [
                Column(name=_concat_str(alias_name_dict.get(alias, alias), x.get("date", "")), value=x.get("value", None))
                for x in value_list
            ]
            data_point_meta = data_points_dict.get(alias)
            sub_investment_results.add_column_data(investment_id, alias, column_data)
            sub_investment_results.add_meta_data(investment_id, alias, data_point_meta)
    return sub_investment_results


def _concat_str(s1: str, s2: str) -> str:
    return s1 + " - " + s2


def _get_sec_id(investment_id: str) -> str:
    assert isinstance(investment_id, str)
    sec_id, *_ = investment_id.split(";")
    return sec_id
