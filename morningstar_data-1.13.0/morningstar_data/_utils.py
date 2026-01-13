import inspect
import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import simplejson as json
from pandas import DataFrame

from .direct._api import _direct_api_request
from .direct._config import _Config

_config = _Config()
_logger = logging.getLogger(__name__)


def get_parameter(key: str) -> Optional[Any]:
    """Fetch a parameter from query string with a fallback to an environment variable

    Voila supports query string parameters (available via QUERY_STRING environment variable).
    This function helps to prioritize query string parameters over environment variables with matching names.
    """

    try:
        from os import environ as _env
        from urllib.parse import parse_qs

        # Voila supports
        query_string = _env.get("QUERY_STRING", "")
        parameters: Dict[str, Any] = parse_qs(query_string)
        if key in parameters:
            return parameters[key][0]
        if key in _env:
            return _env.get(key)
        return None
    except BaseException:
        _logger.warn(f"Something is wrong with key={key}")
        return None


def load_remote(target_folder: str, remote_endpoint: str) -> Any:
    from os import path as _path
    from subprocess import run

    try:
        result = run(
            [
                "sh",
                _path.join(_path.dirname(__file__), "remote.sh"),
                target_folder,
                remote_endpoint,
            ],
            shell=False,
            capture_output=True,
        )
        return result
    except Exception as _ex:
        _logger.warn(_ex)


def format_analytics_logs(
    function: str, params: str, component: str = "morningstar_data.direct", action: str = "FUNCTION_RUN"
) -> str:
    request_id = str(uuid.uuid1())
    result = {
        "object_type": "MD Package",
        "object_id": function,
        "application": "Analytics Lab",
        "component": component,
        "action": action,
        "session_id": os.getenv("ANALYTICSLAB_SESSION_ID"),
        "user_id": os.getenv("UIM_USER_ID"),
        "details": params,
        "event_id": request_id,
    }

    return f"{json.dumps(result, ignore_nan=True)}"


def get_log_flag(func: Callable) -> bool:
    # log_flag allows to know if a function should be logged for analytics or not
    log_flag = True

    # Return a list of frame records for the caller function’s stack. The first entry in the returned list represents the caller; the last entry represents the outermost call on the stack.
    stack = inspect.stack()

    for stk in stack:
        # <module> represents it is for function called by function execution) - should return true
        if stk.function == "<module>":
            return log_flag
        # Checks if caller is any internal function or not get_log_flag (self), sets flag to false
        elif stk.function not in ["get_log_flag", "wrapper"] or stk.function.startswith("_"):
            log_flag = False
            return log_flag
        else:
            continue

    return log_flag


def _get_user_cells_quota() -> Dict:
    url = f"{_config.asset_service_url()}/auto/analyticslab/log"
    response_json = _direct_api_request("get", url)
    user_cells_quota = {"daily_cell_limit": response_json["max"], "daily_cell_remaining": response_json["remains"]}
    return user_cells_quota


def _get_data_points_total_columns(data_point_details: list) -> int:
    url = f"{_config.data_point_service_url()}v1/datapoints/columns"
    columns_response_json: List[Any] = _direct_api_request("POST", url, json.dumps(data_point_details, ignore_nan=True))
    total_columns = sum(int(item.get("columns")) for item in columns_response_json)
    return total_columns


def join_url(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts)


def data_point_dataframe_to_list(dp_dataframe: DataFrame) -> List[Dict[str, Any]]:
    dp_dict = dp_dataframe.replace(np.nan, None).to_dict(orient="records")

    # Avoid sending dp settings with None values, the API will consider them invalid
    return [{k: v for k, v in dp.items() if v is not None} for dp in dp_dict]
