from typing import Any

from ..._base import _logger
from .._base_api import APIBackend
from .._config import _Config
from .._error_messages import RESOURCE_NOT_FOUND_ERROR_HOLDING
from .._exceptions import (
    ResourceNotFoundError,
)

_config = _Config()


class HoldingAPIBackend(APIBackend):
    """
    Subclass to call the Holding API and handle any HTTP errors that occur.
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
            raise ResourceNotFoundError(RESOURCE_NOT_FOUND_ERROR_HOLDING) from None
