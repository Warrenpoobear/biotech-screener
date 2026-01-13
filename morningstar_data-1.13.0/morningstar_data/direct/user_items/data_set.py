import re
import warnings
from typing import Any, Dict, Optional

from pandas import DataFrame

from ..._base import _logger
from .. import _decorator, _error_messages, _utils
from .._base_api import APIBackend
from .._config import _Config
from .._exceptions import BadRequestException, ResourceNotFoundError

_config = _Config()


class DataSetAPIBackend(APIBackend):
    """
    Subclass to call the Data Set API and handle any HTTP errors that occur.
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
            raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_PORTFOLIO_DATA_SET) from None
        else:
            pass


_data_set_api = DataSetAPIBackend()


@_decorator.not_null
def _get_morningstar_data_set_data_points(data_set_id: str) -> DataFrame:
    response_json = _get_defaultview_data_points(data_set_id)
    if not response_json.get("content", None):
        raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_PORTFOLIO_DATA_SET)
    content = response_json["content"]
    extracted_content = _utils._extract_data(content)
    details = DataFrame(extracted_content)
    details_with_renamed_column = _utils._rename_data_frame_column(details, source="id", target="datapointId")
    return _utils._filter_data_frame_column_by_setting(details_with_renamed_column)


@_decorator.typechecked
def get_data_sets() -> DataFrame:
    """Returns all data sets saved by or shared to a user in Morningstar Direct.

    :Returns:
        DataFrame: A DataFrame object with all data sets. DataFrame columns include:

        * datasetId
        * name
        * source
        * shared


    :Examples:

    ::

        import morningstar_data as md

        df = md.direct.user_items.get_data_sets()
        df

    :Output:
        ============  =====================  ============  ========
        datasetId     name                   source        shared
        ============  =====================  ============  ========
        5447361       alpha                  DESKTOP       False
        5386429       Michael's Search Ds	 DESKTOP       False
        ...
        ============  =====================  ============  ========

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    url = f"{_config.dataset_service_url()}v1/datasets"
    response_json: Dict[str, Any] = _data_set_api.do_get_request(url)
    if not response_json.get("datasets", None):
        raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_ERROR_NO_DATA_SETS)

    result = DataFrame(response_json["datasets"])
    return result[["datasetId", "name", "source", "shared"]]


@_decorator.not_null
@_decorator.typechecked
def get_data_set(data_set_id: str) -> DataFrame:
    warnings.warn(
        "The get_data_set function is deprecated and will be removed in the next major version. Use get_data_set_details instead",
        FutureWarning,
        stacklevel=2,
    )
    return get_data_set_details(data_set_id)


@_decorator.not_null
@_decorator.typechecked
def get_data_set_details(data_set_id: str) -> DataFrame:
    """Returns all data points for a given saved data set.

    Args:
        data_set_id (:obj:`str`): Unique identifier of a Morningstar or user-created data set saved
            in Morningstar Direct, e.g., "6102286". Use the `get_data_sets <#morningstar_data.direct.user_items.get_data_sets>`_ or `get_morningstar_data_sets <#morningstar_data.direct.get_morningstar_data_sets>`_ functions to discover saved data sets.

    :Returns:
        DataFrame: A DataFrame object with data points. DataFrame columns include:

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
        Get data points contained in data set "0218-0450".

    ::

        import morningstar_data as md

        df = md.direct.user_items.get_data_set_details(data_set_id="0218-0450")
        df

    :Output:
        ===========  =====  ====  ========  =============  ===  ================================
        datapointId  alias  type  universe  datapointName  ...  calcIsApplyIsraelsenModification
        ===========  =====  ====  ========  =============  ===  ================================
        OS01W        Z0	    text            Name           ...  NaN
        ...
        ===========  =====  ====  ========  =============  ===  ================================

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    if re.match(r"^\d+$", data_set_id, re.M) is not None:
        return _get_user_data_set_data_points(data_set_id)
    elif re.match(r"^\d{4}-\d{4}$", data_set_id, re.M) is not None:
        return _get_morningstar_data_set_data_points(data_set_id)
    else:
        raise BadRequestException(_error_messages.BAD_REQUEST_ERROR_DATA_SET)


def _get_user_data_set_data_points(data_set_id: str) -> DataFrame:
    response_json = _get_user_data_set(data_set_id)
    if not response_json.get("content", None):
        raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_PORTFOLIO_DATA_SET)
    content = response_json["content"]
    extracted_data = _utils._extract_data(content)
    details = DataFrame(extracted_data)
    return _utils._filter_data_frame_column_by_setting(details)


def _get_user_data_set(data_set_id: Optional[str]) -> Dict[str, Any]:
    url = f"{_config.dataset_service_url()}v1/datasets/{data_set_id}"
    response_json: Dict[str, Any] = _data_set_api.do_get_request(url)
    return response_json


def _get_defaultview_data_points(view_id: str) -> Dict[str, Any]:
    url = f"{_config.data_point_service_url()}v1/defaultviews/{view_id}/datapoints"
    response_json: Dict[str, Any] = _data_set_api.do_get_request(url)
    return response_json


def _get_binding_data_set(user_object_id: str, id_type: str) -> Dict[str, Any]:
    url = f"{_config.dataset_service_url()}v1/datasets/{id_type}/{user_object_id}"
    response_json: Dict[str, Any] = _data_set_api.do_get_request(url)
    return response_json
