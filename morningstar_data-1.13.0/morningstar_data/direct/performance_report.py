import shutil
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import requests
from pandas import DataFrame

from .._base import _logger
from . import _decorator
from ._base_api import APIBackend
from ._config import _Config
from ._error_messages import RESOURCE_NOT_FOUND_ERROR_PERFORMANCE_REPORT
from ._exceptions import (
    BadRequestException,
    NetworkExceptionError,
    ResourceNotFoundError,
    ValueErrorException,
)

_config = _Config()


class PerformanceReportAPIBackend(APIBackend):
    """
    Subclass to call the Performance Report API and handle any HTTP errors that occur.
    """

    def __init__(self) -> None:
        super().__init__()

    def _handle_custom_http_errors(self, res: Any) -> Any:
        """
        Handle HTTP errors with custom error messages
        """
        response_message = res.json().get("message", "")
        if res.status_code == 404:
            _logger.debug(f"Resource Not Found Error: {res.status_code} {response_message}")
            raise ResourceNotFoundError(RESOURCE_NOT_FOUND_ERROR_PERFORMANCE_REPORT) from None


_performance_report_api_request = PerformanceReportAPIBackend()


@_decorator.typechecked
def get_reports() -> DataFrame:
    """Returns all performance reports saved or shared to a user in Morningstar Direct.

    :Returns:
        DataFrame: A DataFrame object with all performance reports. DataFrame columns include:

        * reportId
        * name
        * permission
        * ownerId
        * ownerName
        * shared
        * createdOn
        * lastCalculatedOn
        * folderId
        * hasInvestmentSource

    :Examples:

    ::

        import morningstar_data as md

        df = md.direct.performance_report.get_reports()
        df

    :Output:
        ========  ======  ==========  =======  ======  ====================  ================  ========
        reportId  name    permission  ownerId  shared  createdOn             lastCalculatedOn  folderId
        ========  ======  ==========  =======  ======  ====================  ================  ========
        4940775   sample  READ_WRITE  XXX      False   2017-08-17T09:39:00Z                    1
        ...
        ========  ======  ==========  =======  ======  ====================  ================  ========

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

    """
    url = f"{_config.performancereport_service_url()}v1/reports"
    response_json: Dict[str, Any] = _performance_report_api_request.do_get_request(url)
    return DataFrame(response_json["reports"])


@_decorator.not_null
@_decorator.typechecked
def get_report(report_id: str) -> DataFrame:
    """Returns performance report data for the specified report ID.

    Args:
        report_id (:obj:`str`): Unique identifier of a saved performance report from the Performance Reporting
            module in Morningstar Direct, e.g., "7782164". Use `get_reports <#morningstar_data.direct.get_reports>`_ to discover possible values.

    :Returns:
        DataFrame: A DataFrame object with calculated performance report data. Columns include all columns
        that the user configured in the performance report.

    :Examples:
        Get performance report data.

    ::

        import morningstar_data as md

        df = md.direct.performance_report.get_report(report_id="2463866") # Replace with a valid report ID
        df

    :Output:
        ================  ===========  =============  =========================  ======
        Group/Investment  Object Type  Display Group  Peer Group                 Ticker
        ================  ===========  =============  =========================  ======
        sample            investments  Unclassified   Peer Group: Display Group  XXX
        ================  ===========  =============  =========================  ======

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.
    """
    url = f"{_config.performancereport_service_url()}v1/reports/{report_id}"
    response_json: Dict[str, Any] = _performance_report_api_request.do_get_request(url)
    return _merge_datas(response_json)


def _get_all_columns(columns: List[Dict[str, Any]], name_master: Optional[str] = None) -> List[Dict[str, Any]]:
    all_columns = []
    for column in columns:
        if "hidden" in column and column["hidden"] is True:
            continue
        elif "children" in column:
            all_columns += _get_all_columns(column["children"], column["name"])
        elif name_master is None and column["name"] == "Name":
            column["fullName"] = "Group/Investment"
            all_columns.append(column)
            all_columns.append({"fullName": "Object Type", "id": "ObjectType"})
            all_columns.append({"fullName": "Display Group", "id": "DisplayGroup"})
            all_columns.append({"fullName": "Peer Group", "id": "PeerGroup"})
        else:
            column["fullName"] = name_master + " " + column["name"] if name_master is not None else column["name"]
            if column.get("mstarIpType", "") == "":
                all_columns.append(column)
            else:
                all_columns.append({"fullName": column["fullName"], "id": column["id"]})
                all_columns.append(
                    {
                        "fullName": column["fullName"] + " - display text",
                        "id": column["id"],
                        "ismstarip": True,
                    }
                )

    return all_columns


def _get_all_columns_name(columns: List[Dict[str, Any]]) -> List[str]:
    all_columns = []
    for c in columns:
        all_columns.append(c["fullName"])
    return all_columns


def _filter_grouy_by_type(groups: List[Dict[str, Any]], type: str) -> Dict[str, Any]:
    new_groups = [group for group in groups if group["type"] == type]
    groups_group_id = [group["id"] for group in new_groups]
    groups_group_dict = dict(zip(groups_group_id, new_groups))
    return groups_group_dict


def _get_data_dict(datas: Dict[str, Any], name: str) -> Dict[str, Any]:
    # Ignoring type below -- no idea what any of data, datas, data_dict, or datas_id are
    datas_id = [data[name] for data in datas]  # type: ignore
    return dict(zip(datas_id, datas))


def _set_data_rows(values: Dict[str, Any], columns: List[Any], switch: Dict[str, Any]) -> List[Any]:
    set_data = []
    for col in columns:
        col_id = col["id"]
        to_add = switch.get(col_id, None)
        if to_add is not None:
            set_data.append(to_add)
        elif col_id in values:
            if col.get("ismstarip", False):
                set_data.append(values[col_id].get("text", ""))
            else:
                set_data.append(values[col_id].get("value", ""))
        else:
            set_data.append("")

    return set_data


def _get_data_rows(
    investments: List[Dict[str, Any]],
    groups_group: Dict[str, Any],
    new_columns: List[Dict[str, Any]],
    ObjectType: Any,
    PeerGroup: Any,
) -> List[Any]:
    output_datas = []
    for inv in investments:
        group_id = inv["groupId"]
        values_dict = _get_data_dict(inv["values"], "alias")
        switch = {
            "ObjectType": ObjectType,
            "DisplayGroup": groups_group[group_id]["name"],
            "PeerGroup": PeerGroup,
        }
        output_datas.append(_set_data_rows(values_dict, new_columns, switch))
    return output_datas


def _filter_datas_by_group_id(datas: List[Dict[str, Any]], group_id: str) -> List[Dict[str, Any]]:
    return [data for data in datas if data["groupId"] == group_id]


def _get_sub_group(groups_data: List[Dict[str, Any]], group_id: str) -> Dict[str, Any]:
    new_groups = [group for group in groups_data if "groupId" in group and group["groupId"] == group_id]
    groups_group_id = [group["id"] for group in new_groups]
    return dict(zip(groups_group_id, new_groups))


def _get_group_data_rows(
    datas: List[Dict[str, Any]],
    groups_group: Any,
    new_columns: List[Dict[str, Any]],
    ObjectType: Any,
    group_name: Any,
) -> List[Any]:
    output_datas = []
    switch = {
        "ObjectType": ObjectType,
        "DisplayGroup": group_name,
        "PeerGroup": "",
    }

    for inv in datas:
        values_dict = _get_data_dict(inv["values"], "alias")
        output_datas.append(_set_data_rows(values_dict, new_columns, switch))

    return output_datas


def _get_peer_group_name_by_group_id(group_peer_groups: Union[Dict[Any, Any], Any]) -> Any:
    if (
        group_peer_groups
        and group_peer_groups[0]
        and group_peer_groups[0]["values"]
        and group_peer_groups[0]["values"][0]
        and group_peer_groups[0]["values"][0]["value"]
    ):
        return group_peer_groups[0]["values"][0]["value"]

    return ""


def _merge_datas(performance_report_data: Dict[str, Any]) -> DataFrame:
    metadata = performance_report_data["metaData"]
    view = metadata["view"]
    if view and view["id"] == "5":
        raise BadRequestException("This view not supported: " + view["name"])

    subtype_name = {
        "PeerGroup": "Peer Group Statistics",
        "DisplayGroup": "Display Group Statistics",
    }

    columns = performance_report_data["columns"]
    investments = performance_report_data["investments"]
    benchmarks = performance_report_data["benchmarks"]
    groups_data = performance_report_data["groups"]
    peer_groups = performance_report_data["peerGroups"]
    ranks = performance_report_data["ranks"]
    summary_statistics = performance_report_data["summaryStatistics"]

    all_data_row = []
    new_columns = _get_all_columns(columns)
    new_columns_name = _get_all_columns_name(new_columns)

    new_groups = [group for group in groups_data if group["type"] == "Group"]
    groups_group = _filter_grouy_by_type(groups_data, "Group")
    for group in new_groups:
        group_id = group["id"]
        group_name = group["name"]
        sub_groups = _get_sub_group(groups_data, group_id)
        group_investments = _filter_datas_by_group_id(investments, group_id)
        group_benchmarks = _filter_datas_by_group_id(benchmarks, group_id)
        group_peer_groups = _filter_datas_by_group_id(peer_groups, group_id)
        group_ranks = _filter_datas_by_group_id(ranks, group_id)
        peer_group_name = _get_peer_group_name_by_group_id(group_peer_groups)
        all_data_row.extend(
            _get_data_rows(
                group_investments,
                groups_group,
                new_columns,
                "investments",
                peer_group_name,
            )
        )
        all_data_row.extend(_get_data_rows(group_benchmarks, groups_group, new_columns, "Display Benchmark", ""))
        all_data_row.extend(_get_group_data_rows(group_peer_groups, groups_group, new_columns, "PeerGroup", group_name))
        all_data_row.extend(_get_group_data_rows(group_ranks, groups_group, new_columns, "Ranks", group_name))

        for key in sub_groups:
            subtype = sub_groups[key]["type"]

            if subtype in subtype_name:
                sub_peer_groups = _filter_datas_by_group_id(summary_statistics, sub_groups[key]["id"])
                all_data_row.extend(
                    _get_group_data_rows(
                        sub_peer_groups,
                        sub_peer_groups,
                        new_columns,
                        subtype_name[subtype],
                        group_name,
                    )
                )

    lssilss = _filter_datas_by_group_id(summary_statistics, "lssilss")
    if lssilss:
        all_data_row.extend(_get_group_data_rows(lssilss, lssilss, new_columns, "List Summary Statistics", ""))

    data_frame = DataFrame(all_data_row)
    data_frame.columns = new_columns_name
    return data_frame


@_decorator.not_null
@_decorator.typechecked
def calculate_report(report_id: str, timezone_offset: Optional[int] = None) -> DataFrame:
    """Initiates re-calculation of a performance report.

    Args:
        report_id (:obj:`str`): Unique identifier of a saved performance report from the Performance Reporting
            module in Morningstar Direct, e.g., "7782164". Use `get_reports <#morningstar_data.direct.get_reports>`_ to discover possible values.

    :Returns:
        DataFrame: A DataFrame object confirming that report calculation was triggered. DataFrame columns include:

        * reportId
        * success

    Raises:
        ValueErrorException: Raised when the ``report_id`` parameter is invalid.

    :Examples:
        Calculate the performance report.

    ::

        import morningstar_data as md

        df = md.direct.performance_report.calculate_report(report_id="7128568") # Replace with a valid report ID
        df

    :Output:
        ========  =======
        reportId  success
        ========  =======
        7128568   True
        ========  =======

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.
    """
    df = get_report_status(report_id)
    if df["status"].values[0] == "Ready" or df["status"].values[0] == "Failed":
        if timezone_offset:
            url = f"{_config.performancereport_service_url()}v1/reports/calc/{report_id}?timeZoneOffset={timezone_offset}"
            warnings.warn("timezone_offset is a deprecated parameter and will be removed soon.", FutureWarning, stacklevel=2)
        else:
            url = f"{_config.performancereport_service_url()}v1/reports/calc/{report_id}"

        response_json = _performance_report_api_request.do_post_request(url)
        return DataFrame(response_json)
    else:
        raise ValueErrorException(f"report {report_id} is calculating.")


@_decorator.not_null
@_decorator.typechecked
def get_report_status(report_id: str) -> DataFrame:
    """Returns the current calculation status of the specified performance report. Possible statuses are:

    * Failed
    * Queued
    * Calculating
    * Generating Report
    * Merging Excel
    * Downloading
    * Ready

    Args:
        report_id (:obj:`str`): Unique identifier of a saved performance report from the Performance Reporting
            module in Morningstar Direct, e.g., "7782164". Use `get_reports <#morningstar_data.direct.get_reports>`_ to discover possible values.

    :Returns:
        DataFrame: A DataFrame object with calculation status. DataFrame columns include:

        * reportId
        * status

    :Examples:
        Get performance report calculation status.

    ::

        import morningstar_data as md

        df = md.direct.performance_report.get_report_status(report_id="2463866") # Replace with a valid report ID
        df

    :Output:
        ========  ======
        reportId  status
        ========  ======
        2463866   Ready
        ========  ======

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.
    """
    url = f"{_config.performancereport_service_url()}v1/reports/status/{report_id}"
    response_json = _performance_report_api_request.do_get_request(url)
    return DataFrame(response_json)


@_decorator.not_null
@_decorator.typechecked
def export_to_excel(report_id: str, file_name: Optional[str] = None) -> str:
    """Export a performance report to Excel.

    Args:
        report_id (:obj:`str`): Unique identifier of a saved performance report from the Performance Reporting
            module in Morningstar Direct, e.g., "7782164". Use `get_reports <#morningstar_data.direct.get_reports>`_ to discover possible values.
        file_name (:obj:`str`, `optional`): Custom file name for the exported Excel file. If not provided, a default name with the format "report_<report_id>" will be used, e.g., "report_7782164.xlsx".

    Returns:
        str: Excel file name of the exported performance report.


    Examples:
        Export the given performance report.

    ::

        import morningstar_data as md

        resp = md.direct.performance_report.export_to_excel(report_id="2442678") # Replace with a valid report ID
        resp

    :Output:
        report_2442678.xlsx


    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    report_id_parse = quote(report_id, "utf-8")
    url = f"{_config.performancereport_export_service_url()}v1/reports/export/{report_id_parse}"
    presigned_url_response = _performance_report_api_request.do_get_request(url=url)

    current_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file_suffix = ".xlsx"
    if file_name is None or file_name == "":
        file_name = f"performance_report_{report_id}_{current_date_time}.xlsx"
    elif not file_name.endswith(excel_file_suffix):
        file_name = file_name + excel_file_suffix
    try:
        url = presigned_url_response["url"]
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(file_name, "wb") as excel_file:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, excel_file)
        return file_name
    # Catching HTTPErrors as this is presigned URL from bucket no do_get_request is used
    except requests.ConnectionError as e:
        _logger.error(e)
        raise NetworkExceptionError from None
    except requests.HTTPError as e:
        _performance_report_api_request._handle_custom_http_errors(response)
        _performance_report_api_request._handle_default_http_errors(response)
        raise e
