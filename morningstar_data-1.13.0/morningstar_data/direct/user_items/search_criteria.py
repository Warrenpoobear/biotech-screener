import warnings
from typing import Any, Dict, List
from urllib.parse import quote

import simplejson as json
from pandas import DataFrame

from ..._base import _logger
from .. import _decorator, _error_messages, investment
from .._base_api import APIBackend
from .._config import _Config
from .._exceptions import BadRequestException, ResourceNotFoundError
from .._utils import _reduce_list_data
from . import _utils

_config = _Config()


class SearchCriteriaAPIBackend(APIBackend):
    """
    Subclass to call the Search Criteria API and handle any HTTP errors that occur.
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
            raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_ERROR_SEARCH_CRITERIA) from None
        else:
            pass


_search_criteria_api = SearchCriteriaAPIBackend()


@_decorator.not_null
@_decorator.typechecked
def get_search_results(search_criteria_id: str) -> DataFrame:
    """Returns all investments matching the specified search criteria.

    Args:
        search_criteria_id (:obj:`str`): Unique identifier of a saved search criteria in Morningstar Direct, e.g., "9009". Use `get_search_criteria <#morningstar_data.direct.user_items.get_search_criteria>`_ to discover possible values.

    :Returns:
        DataFrame: A DataFrame object with investments that match the search criteria. DataFrame columns include:

        * secid
        * masterportfolioid
        * tradingsymbol
        * name
        * securitytype
        * exchangeid
        * category

    :Examples:
        Get investments that match the given search criteria.

    ::

        import morningstar_data as md

        df = md.direct.user_items.get_search_results(search_criteria_id="4237053") # Replace with a valid search criteria id
        df

    :Output:
        ==========  =================  =============  =======================  ============  ==========  ================
        secid       masterportfolioid  tradingsymbol  name                     securitytype  exchangeid  category
        ==========  =================  =============  =======================  ============  ==========  ================
        FOUSA06JNH  210311             AAAAX          DWS RREEF Real Assets A  FO            EXXNAS      World Allocation
        ...
        ==========  =================  =============  =======================  ============  ==========  ================

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    try:
        search_df = DataFrame(columns=_utils.investments_list_default_columns)
        _logger.debug(f"Getting Direct SearchCriteria: {search_criteria_id}")

        search_criteria_id_encode = quote(search_criteria_id, "utf-8")
        url = f"{_config.searches_service_url()}v1/searches/{search_criteria_id_encode}"

        response_json = _search_criteria_api.do_get_request(url)
        search = response_json["content"]

        _logger.debug(f"Executing Direct SearchCriteria: {search_criteria_id}")
        investments = execute_search(search)

        # [{"id": "1;a"}, {"id": "2;x"}] => "'1','2'"
        ids = list(map(lambda x: f"{x['id']}", investments))

        if not ids:
            _logger.debug("No ids returned in Direct Searches API response, creating empty DataFrame.")
            search_df = DataFrame(columns=_utils.investments_list_default_columns)
        else:
            search_df = investment.get_investment_data(investments=ids, data_points=_utils.investments_list_data_points)
            search_df = search_df.drop(["Id"], axis=1)
            search_df.columns = _utils.investments_list_default_columns

        if search_df.empty:
            _logger.debug("Returning empty DataFrame.")
        _logger.debug(f"Returning DataFrame for SearchCriteria: {search_criteria_id}")
        return search_df

    except KeyError:
        raise BadRequestException(_error_messages.BAD_REQUEST_ERROR) from None


@_decorator.typechecked
def get_search_criterias() -> DataFrame:
    warnings.warn(
        "The get_search_criterias function is deprecated and will be removed in the next major version. Use get_search_criteria instead",
        FutureWarning,
        stacklevel=2,
    )
    return get_search_criteria()


@_decorator.typechecked
def get_search_criteria() -> DataFrame:
    """Returns all search criteria saved or shared to a user in Morningstar Direct

    :Returns:
        DataFrame: A DataFrame object with all search criteria names and identifiers. DataFrame columns include:

        * id
        * name

    :Examples:

    ::

        import morningstar_data as md

        df = md.direct.user_items.get_search_criteria()
        df

    :Output:
        =======  ======
        id       name
        =======  ======
        4175985  sample
        5022284  others
        ...
        =======  ======

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """
    try:
        _logger.debug("Getting all Direct SearchCriteria")

        url = f"{_config.searches_service_url()}v1/searches"

        response_json = _search_criteria_api.do_get_request(url)
        props_dict: Dict[str, str] = {"id": "searchId", "name": "name"}
        all_search_criterias = _reduce_list_data(response_json["search"], props_dict)
        return DataFrame(all_search_criterias)
    except Exception as e:
        if isinstance(e, ResourceNotFoundError):
            raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_ERROR_NO_SEARCH_CRITERIA) from None
        raise e from None


@_decorator.not_null
@_decorator.typechecked
def get_search_criteria_condition(search_criteria_id: str) -> Any:
    warnings.warn(
        "The get_search_criteria_condition function is deprecated and will be removed in the next major version. Use get_search_criteria_conditions instead",
        FutureWarning,
        stacklevel=2,
    )
    return get_search_criteria_conditions(search_criteria_id)


@_decorator.not_null
@_decorator.typechecked
def get_search_criteria_conditions(search_criteria_id: str) -> Dict[str, Any]:
    """Returns the detailed definition of a saved search criteria.

    Args:
        search_criteria_id (:obj:`str`): Unique identifier of a saved search criteria in Morningstar Direct, e.g., "9009". Use `get_search_criteria <#morningstar_data.direct.user_items.get_search_criteria>`_ to discover possible values.

    :Returns:
        Dict:

    :Examples:

    ::

        import morningstar_data as md

        df = md.direct.user_items.get_search_criteria_conditions(search_criteria_id="9009") # Replace with a valid search criteria id
        df

    :Output:

        ::

            {
                "universeId": "FO",
                "subUniverseId": "",
                "subUniverseName": "",
                "securityStatus": "activeonly",
                "useDefinedPrimary": False,
                "criteria": [
                    {
                        "relation": "",
                        "field": "OS001",
                        "operator": "=",
                        "value": "AAAAX",
                        "id": "FOUSA06JNH",
                        "name": "AAAAX"
                    }
                ]
            }

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

    """

    try:
        search_criteria_id_encode = quote(search_criteria_id, "utf-8")
        url = f"{_config.searches_service_url()}v1/searches/{search_criteria_id_encode}"
        response_json = _search_criteria_api.do_get_request(url)
        condition: Dict[str, Any] = response_json.get("content", {})
        return condition
    except Exception as e:
        if isinstance(e, ResourceNotFoundError):
            raise ResourceNotFoundError(
                _error_messages.RESOURCE_NOT_FOUND_ERROR_SEARCH_CRITERIA_ID.format(search_criteria_id)
            ) from None
        raise e from None


def execute_search(search: dict) -> List[Dict[str, Any]]:
    request_body = json.dumps({"search": search}, ignore_nan=True)
    url = f"{_config.securitydata_service_url()}v1/searchresults"

    response_json: Dict[Any, Any] = _search_criteria_api.do_post_request(url, data=request_body)
    investments: List[Dict[str, Any]] = response_json.get("investments", [])
    if "investments" not in response_json:
        _logger.error(f"No investments in search results: {response_json}")
        raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_ERROR_SEARCH_CRITERIA_NO_INVESTMENTS) from None
    return investments
