from dataclasses import dataclass
from typing import Any, List, Optional

import simplejson as json
from pandas import DataFrame

from ..._base import _logger
from .._base_api import APIBackend
from .._config import _Config
from .._data_objects import DataPoints, Investments
from .._error_messages import BAD_REQUEST_ERROR_INVALID_PERCENTILES
from .._exceptions import (
    BadRequestException,
    InternalServerError,
    ResourceNotFoundError,
)
from ..data_type import Order, PeerGroupMethodology

_config = _Config()


class PeerGroupAPIBackend(APIBackend):
    """
    Subclass to call the Peer Group Data API and handle any HTTP errors that occur.
    """

    def __init__(self) -> None:
        super().__init__()

    def _handle_custom_http_errors(self, res: Any) -> Any:
        """
        Handle HTTP errors with custom error messages
        """
        response_message = res.json().get("message")
        if res.status_code == 404:
            _logger.debug(f"Resource Not Found Error: {res.status_code} {response_message}")
            raise ResourceNotFoundError from None

        if res.status_code == 500:
            _logger.debug(f"Something went wrong: {res.status_code} {response_message}")
            raise InternalServerError(response_message) from None


_peer_group_api_request = PeerGroupAPIBackend()


@dataclass(frozen=True)
class PeerGroupRequest:
    investment_source: dict
    data_points: DataFrame
    order: Order
    percentiles: Optional[List[int]] = None
    methodology: Optional[PeerGroupMethodology] = None


@dataclass(frozen=True)
class PeerGroupResponse:
    req: PeerGroupRequest
    breakpoints: list


class PeerGroupProvider:
    def build_request(
        self,
        investment_object: Investments,
        data_point_object: DataPoints,
        order: Order,
        percentiles: Optional[List[int]] = None,
        methodology: Optional[PeerGroupMethodology] = None,
    ) -> PeerGroupRequest:
        # do not support dataset id
        data_point_object.data_set_id = None
        methodology = _get_methodology_value(methodology)
        _validate_percentiles(percentiles)

        investment_source = investment_object.generate_investment_source()
        data_point_data_frame = _get_data_points(data_point_object)

        return PeerGroupRequest(investment_source, data_point_data_frame, order, percentiles, methodology)

    def run_request(self, req: PeerGroupRequest) -> PeerGroupResponse:
        breakpoints = _get_breakpoints(
            investment_source=req.investment_source,
            data_point_data_frame=req.data_points,
            order=req.order.value,
            methodology=req.methodology,
            percentiles=req.percentiles,
        )
        return PeerGroupResponse(req, breakpoints)

    def build_data_frame(self, resp: PeerGroupResponse) -> DataFrame:
        return _convert_to_breakpoints_data_frame(resp.breakpoints)


def _get_breakpoints(
    investment_source: dict,
    data_point_data_frame: DataFrame,
    order: Any,
    methodology: Optional[str] = None,
    percentiles: Optional[list] = None,
) -> list:
    breakpoints = []
    postbody = {
        "investmentSource": investment_source,
        "datapoints": data_point_data_frame.to_dict(orient="records"),
        "percentiles": percentiles,
        "order": order,
    }
    url = f"{_config.peergroup_service_url()}peergroupservice/v1/breakpoints"
    if methodology is not None and len(methodology.strip()) > 0:
        url = f"{url}?methodology={methodology}"
    resp = _peer_group_api_request.do_post_request(url, json.dumps(postbody, ignore_nan=True))
    if resp and isinstance(resp, dict):
        breakpoints = resp.get("breakpoints", [])
    return breakpoints


def _convert_to_breakpoints_data_frame(breakpoints: list) -> DataFrame:
    breakpoint_list = []
    for breakpoint in breakpoints:
        breakpoint_dict = dict()
        breakpoint_dict["Alias"] = breakpoint.get("alias", None)
        time_period = breakpoint.get("timePeriod", dict())
        breakpoint_dict["StartDate"] = time_period.get("startDate", dict())
        breakpoint_dict["EndDate"] = time_period.get("endDate", dict())
        values = breakpoint.get("values", list())
        for value in values:
            breakpoint_dict[value.get("percentile", None)] = value.get("value", None)
        breakpoint_list.append(breakpoint_dict)
    return DataFrame(breakpoint_list)


def _get_methodology_value(methodology: Optional[PeerGroupMethodology] = None) -> Any:
    if methodology is not None and isinstance(methodology, PeerGroupMethodology):
        return methodology.value
    return None


def _validate_percentiles(percentiles: Optional[list] = None) -> None:
    if not percentiles:
        return

    for per in percentiles:
        if per is None or not isinstance(per, int) or per < 1 or per > 100:
            raise BadRequestException(BAD_REQUEST_ERROR_INVALID_PERCENTILES)


def _get_data_points(data_point_object: DataPoints) -> DataFrame:
    data_point_object._validate_data_point_ids_and_alias()
    return data_point_object.get_peer_group_data_points()
