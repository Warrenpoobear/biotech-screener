from enum import Enum
from typing import Any, Dict, Optional

from pandas import DataFrame, concat

from ._base import _logger
from .direct._base_api import APIBackend
from .direct._config import _Config
from .direct._error_messages import (
    RESOURCE_NOT_FOUND_ERROR_LOOKUP_KEYWORD,
    RESOURCE_NOT_FOUND_ERROR_LOOKUP_KEYWORD_UNIVERSE,
)
from .direct._exceptions import ResourceNotFoundError

_config = _Config()


class _LookupParameter(Enum):
    KEYWORD = 0
    KEYWORD_UNIVERSE = 1


class LookupAPIBackend(APIBackend):
    """
    Subclass to call the Holding API and handle any HTTP errors that occur.
    """

    def __init__(self, lookup_parameter: _LookupParameter) -> None:
        self._lookup_parameter = lookup_parameter
        super().__init__()

    def _get_resource_not_found_message(self) -> str:
        if self._lookup_parameter == _LookupParameter.KEYWORD:
            return RESOURCE_NOT_FOUND_ERROR_LOOKUP_KEYWORD
        return RESOURCE_NOT_FOUND_ERROR_LOOKUP_KEYWORD_UNIVERSE

    def _handle_custom_http_errors(self, res: Any) -> Any:
        """
        Handle HTTP errors with custom error messages
        """
        response_message = res.json().get("message")
        if res.status_code == 404:
            _logger.debug(f"Resource Not Found Error: {res.status_code} {response_message}")
            raise ResourceNotFoundError(self._get_resource_not_found_message()) from None


_keyword_api_request = LookupAPIBackend(_LookupParameter.KEYWORD)
_keyword_universe_api_request = LookupAPIBackend(_LookupParameter.KEYWORD_UNIVERSE)


def _get_data_point_options(data_point: str, universe: Optional[str]) -> dict:
    url = f"{_config.data_point_service_url()}v1/datapoints/{data_point}/searchoperatorsandoptions"
    _lookup_api_request = _keyword_api_request
    if universe:
        _logger.info("Universe is specified, so searching within this universe only.")
        # Some data points need to specify universe, for example OF003(Morningstar Category).
        url = url + f"?universe={universe}"
        _lookup_api_request = _keyword_universe_api_request
    response_json: Dict[Any, Any] = _lookup_api_request.do_get_request(url)
    return response_json


def _search_option_data(data_point: str, keyword: Optional[str], universe: Optional[str] = None) -> DataFrame:
    # get the available values for special data point.
    data_point_options = _get_data_point_options(data_point, universe)
    df = DataFrame({"id": [], "name": []})
    if data_point_options:
        result = DataFrame(data=data_point_options[0].get("options"))
        result.rename(columns={"value": "id"}, inplace=True)
        df = result[["id", "name"]]

    return _filter_data_frame(df, keyword) if keyword else df


def _filter_data_frame(df: DataFrame, keyword: str) -> DataFrame:
    filtered_df = concat([df[df["id"].str.contains(keyword, case=False)], df[df["name"].str.contains(keyword, case=False)]])
    return filtered_df.drop_duplicates(subset=["id", "name"], keep="first")


def currency_codes(keyword: Optional[str] = None) -> DataFrame:
    """Returns currency codes and currency name that match the given keyword. If no keyword is provided, the function returns all currency codes and currency names.

    Args:
        keyword (Optional[str], optional): A string used to lookup currency codes. Example: "USD". Returns matching currency code for the keyword 'USD'.

    :Returns:
        DataFrame: Returns a DataFrame. The DataFrame columns include:

        * currency_code
        * currency_name

    :Examples:

        Search currency codes based on keyword "Afgh"
    ::

        import morningstar_data as md
        df = md.lookup.currency_codes(
            keyword="Afgh"
        )
        df

    :Output:
        ==============  ===============
        currency_code      currency_name
        ==============  ===============
        AFN                Afghani
        ==============  ===============

    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        ForbiddenError: Raised when the user does not have permission to access the requested resource.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        ResourceNotFoundError: Raised when a currency code matching the given keyword does not exist.
    """

    # LS05M is a datapoint which name is currency, and its options include all available currency value.
    currency_df = _search_option_data(data_point="LS05M", keyword=keyword)
    currency_df.rename(columns={"id": "currency_code", "name": "currency_name"}, inplace=True)
    return currency_df
