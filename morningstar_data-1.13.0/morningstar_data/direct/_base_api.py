import json
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import requests

# NOTE! We are not able to verify the SSL cert on Direct Search API since
#       the VPC endpoint domain name is not on the server's SSL cert.
#       So with a heavy heart, we disable the InsecureRequestWarning
from urllib3.exceptions import InsecureRequestWarning

from .._base import _logger
from . import _error_messages
from ._config import _Config
from ._core_api import make_request
from ._exceptions import (
    AccessDeniedError,
    BadRequestException,
    ClientError,
    ForbiddenError,
    InternalServerError,
    NetworkExceptionError,
    TimeoutError,
)

_config = _Config()

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


class APIBackend(ABC):
    """
    Abstract class for all API requests
    """

    def do_get_request(self, url: str) -> Any:
        """
        Makes a GET request
        """
        return self._make_request(url, "GET")

    def do_post_request(self, url: str, data: Optional[Union[Dict[Any, Any], str]] = None) -> Any:
        """
        Makes a POST request
        """
        return self._make_request(url, "POST", data)

    def do_put_request(self, url: str, data: Optional[Union[Dict[Any, Any], str]] = None) -> Any:
        """
        Makes a PUT request
        """
        return self._make_request(url, "PUT", data)

    def get_headers(self) -> Dict[str, str]:
        return _config.get_headers()

    def _make_request(self, url: str, method: str, data: Optional[Union[Dict[Any, Any], str]] = None) -> Any:
        """
        Request and handle OK (status_code = 200) and network error responses.
        """
        _logger.info(f"Requesting: {method} {url}")
        try:
            res = make_request(method, url, headers=self.get_headers(), verify=False, data=data)
            res.raise_for_status()
            if "warning_messages" in res.headers:
                warning_messages = json.loads(res.headers.get("warning_messages", []))
                for warn_text in warning_messages:
                    warnings.warn(warn_text, Warning)
            return res.json()
        except requests.ConnectionError:
            raise NetworkExceptionError from None
        except requests.HTTPError:
            self._handle_custom_http_errors(res)
            self._handle_default_http_errors(res)

    def _handle_default_http_errors(self, res: Any) -> Any:
        """
        Handle HTTP errors with standard messages
        """
        try:
            response_message = res.json().get("message")
        except KeyError:
            response_message = res.json()
            _logger.debug("DO Request Error")
        if res.status_code == 401:
            _logger.debug(f"Access Denied Error: {res.status_code} {response_message}")
            raise AccessDeniedError from None
        elif res.status_code == 403:
            _logger.debug(f"Forbidden Error: {res.status_code} {response_message}")
            raise ForbiddenError from None
        elif res.status_code == 408:
            _logger.debug(f"Timeout Error: {res.status_code} {response_message}")
            raise TimeoutError from None
        elif res.status_code == 400:
            _logger.debug(f"Bad Request: {res.status_code} {response_message}")
            raise BadRequestException(_error_messages.BAD_REQUEST_ERROR) from None
        elif res.status_code > 400 and res.status_code < 500:
            _logger.debug(f"Client Error: {res.status_code} {response_message}")
            raise ClientError(_error_messages.CLIENT_ERROR) from None
        else:
            _logger.debug(f"Internal Server Error: {res.status_code} {response_message}")
            raise InternalServerError(_error_messages.INTERNAL_SERVER_ERROR) from None

    @abstractmethod
    def _handle_custom_http_errors(self, res: Any) -> None:
        """
        Method to handle special HTTP errors with custom error messages
        """
        pass
