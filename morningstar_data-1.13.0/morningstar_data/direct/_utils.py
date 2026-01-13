import datetime
import time
from typing import Any, Callable, Dict, List
from uuid import UUID

import requests
from pandas import DataFrame

from .._base import _logger
from ._config_key import REQUIRED_DATA_POINT_SETTINGS, VISIBLE_DATA_POINT_SETTINGS
from ._error_messages import BAD_REQUEST_ERROR_INVALID_DATE_FORMAT
from ._exceptions import BadRequestException

FORMAT_DATE = "%Y-%m-%d"


def _reduce_list_data(lists: List[Dict[str, Any]], props_dict: Dict[str, str]) -> List[Dict[str, Any]]:
    filtered_lists, id_for, name = [], props_dict["id"], props_dict["name"]
    for item in lists:
        data: Dict[str, Any] = {"id": item[id_for], "name": item[name]}
        filtered_lists.append(data)

    return filtered_lists


def _extract_data(data: list) -> List[Any]:
    values = []

    for item in data:
        value = {}
        for k, v in item.items():
            if isinstance(v, dict):
                value.update(v)
            else:
                value[k] = v
        values.append(value)

    return values


def _empty_to_none(df: DataFrame) -> DataFrame:
    return df.where((df.notnull()) & (df != ""), None)


def _get_iso_today() -> str:
    d: datetime.date = datetime.date.today()
    format_iso: str = d.isoformat()
    return format_iso


def _mapper_data_frame_return_list(df: DataFrame, get_values: Callable) -> List:
    data = [entry for entry in df.to_dict(orient="index").values()]
    items: List = [get_values(key, value, row) for row in data for key, value in row.items()]
    return [item for item in items if item is not None]


def _mapper_data_frame_return_dict(df: DataFrame, get_values: Callable) -> Dict:
    result: Dict = {}

    def get_values_for_list(column: str, value: Any, row: Dict[str, Any]) -> None:
        get_values(result, column, value, row)
        return None

    _mapper_data_frame_return_list(df, get_values_for_list)

    return result


def _rename_data_frame_column(df: DataFrame, source: str, target: str) -> DataFrame:
    return df.rename(columns={source: target})


def _filter_data_frame_column_by_setting(df: DataFrame) -> DataFrame:
    data_point_settings = REQUIRED_DATA_POINT_SETTINGS + VISIBLE_DATA_POINT_SETTINGS
    return df.filter(items=data_point_settings)


def _reindex_data_frame_column(df: DataFrame, target_column: str, new_index: int) -> DataFrame:
    column_list = df.columns.tolist()

    if (target_column in column_list) and (column_list.index(target_column) != new_index):
        data_point_id_col = df[target_column]
        df.drop(target_column, axis=1, inplace=True)
        df.insert(new_index, target_column, data_point_id_col)

    return df


def _is_uuid(uuid_str: str, version: int = 4) -> bool:
    try:
        uuid_obj = UUID(uuid_str, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_str


def _format_date(date: str) -> str:
    try:
        return time.strftime(FORMAT_DATE, time.strptime(date, FORMAT_DATE))
    except Exception as e:
        _logger.debug(f"Date format error: {e}")
        raise BadRequestException(BAD_REQUEST_ERROR_INVALID_DATE_FORMAT) from None


def get_bytes(url: str) -> requests.Response:
    """
    Downloads bytes from S3 presigned url
    """
    res = requests.get(url)
    res.raise_for_status()
    return res
