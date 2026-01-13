import logging
import os
import uuid
from typing import Any, Dict, Optional, Union

import requests

from ...direct._base_api import APIBackend
from ...direct._exceptions import (
    BadRequestException,
    NetworkExceptionError,
)
from .._core_api import make_request

_logger = logging.getLogger(__name__)


class DeliveryAPIBackend(APIBackend):
    def do_post_request(self, url: str, data: Optional[Union[Dict[Any, Any], str]] = None) -> Any:
        """
        Makes a POST request
        """
        return self._make_request(url, "POST", data)

    def _get_request_id(self) -> str:
        """Returns the request id for use in POST requests"""
        request_id = os.getenv("FEED_RUN_ID", str(uuid.uuid4()))
        _logger.info(f"RequestId: {request_id} passed to POST requests")
        return request_id

    def _make_request(self, url: str, method: str, data: Optional[Union[Dict[Any, Any], str]] = None) -> Any:
        _logger.info(f"Requesting: {method} {url}")
        try:
            headers = {**self.get_headers(), "X-API-RequestId": self._get_request_id()}
            res = make_request(method, url, headers=headers, verify=False, data=data)
            res.raise_for_status()
            return res.json()
        except requests.ConnectionError:
            raise NetworkExceptionError from None
        except requests.HTTPError:
            self._handle_custom_http_errors(res)
            self._handle_default_http_errors(res)

    def _handle_custom_http_errors(self, res: Any) -> Any:
        try:
            response_message = res.json().get("status")
        except requests.JSONDecodeError:
            response_message = res.text

        if res.status_code == 400:
            _logger.debug(f"Bad Request: {res.status_code} {response_message}")
            raise BadRequestException(message=response_message) from None
