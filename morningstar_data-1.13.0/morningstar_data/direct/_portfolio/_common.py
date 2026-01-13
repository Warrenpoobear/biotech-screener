from typing import Any, Optional

from simplejson import JSONEncoder

from ..._base import _logger
from .. import _error_messages
from .._base_api import APIBackend
from .._error_messages import RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_List


class PortfolioJSONEncoder(JSONEncoder):
    def default(self, obj: Any) -> Any:
        return obj.to_json()


class PortfolioNotFoundError(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        if not message:
            super().__init__(_error_messages.RESOURCE_NOT_FOUND_ERROR)
        else:
            super().__init__(message)


class PortfolioDataApiBackend(APIBackend):
    """
    Subclass to call the Portfolio Data API and handle any HTTP errors that occur.
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
            raise PortfolioNotFoundError(RESOURCE_NOT_FOUND_ERROR_PORTFOLIO_List) from None
