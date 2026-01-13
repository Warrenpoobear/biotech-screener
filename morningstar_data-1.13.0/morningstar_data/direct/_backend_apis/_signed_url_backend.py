import json
from typing import Any, Dict, Optional, Union

import requests

from ..._base import _logger
from ...datalake._data_objects.csvfile import CSVFile
from ...datalake._exceptions import (
    InvalidQueryException,
    UnauthorizedDataLakeAccessError,
)
from ...direct._base_api import APIBackend
from ...direct._core_api import make_request
from ...direct._error_messages import RESOURCE_NOT_FOUND_ERROR_HOLDING
from ...direct._exceptions import NetworkExceptionError, ResourceNotFoundError


class SignedUrlBackend(APIBackend):
    """
    Subclass to call a Signed url and handle any HTTP errors that occur.
    Temporarily implements do_get_request becuase the request does not need headers and the response need not be JSON
    """

    def __init__(self) -> None:
        super().__init__()

    def do_post_request(self, url: str, data: Optional[Union[Dict[Any, Any], str]] = None) -> Any:
        """
        Makes a POST request
        """
        return self._make_request(url=url, method="POST", data=data)

    def do_put_request(self, url: str, data: Optional[Union[Dict[Any, Any], str]] = None) -> Any:
        """
        Makes a PUT request
        """
        return self._make_request(url=url, method="PUT", data=data)

    def do_get_request(self, url: str) -> Any:
        """
        Makes a GET request
        """
        return self._make_request(url, "GET")

    def put_csv_file(self, csv: CSVFile, signed_upload_url: str) -> Any:
        """
        Uploads a CSV file to S3 presigned url
        """
        return requests.put(url=signed_upload_url, data=csv.csv_buffer.getvalue())

    def get_bytes(self, url: str) -> Any:
        """
        Downloads bytes from S3 presigned url
        """
        res = requests.get(url)
        res.raise_for_status()
        return res

    def _make_request(self, url: str, method: str, data: Optional[Union[Dict[Any, Any], str]] = None) -> Any:
        """
        Request and handle OK (status_code = 200) and network error responses.
        """
        _logger.info(f"Requesting: {method} {url}")
        try:
            res = make_request(method=method, url=url, headers=self.get_headers(), verify=False, data=json.dumps(data))
            res.raise_for_status()
            return res.text
        except requests.ConnectionError:
            raise NetworkExceptionError from None
        except requests.HTTPError:
            self._handle_custom_http_errors(res)
            self._handle_default_http_errors(res)

    def _handle_custom_http_errors(self, res: Any) -> Any:
        """
        Handle HTTP errors with custom error messages
        """
        try:
            response_message = res.json().get("message")
        except requests.JSONDecodeError:
            response_message = res.text

        if res.status_code == 400:
            _logger.debug(f"Bad Request Error: {res.status_code} {response_message}")
            raise InvalidQueryException(response_message) from None
        if res.status_code == 401:
            _logger.debug(f"Unauthorized Error: {res.status_code} {response_message}")
            raise UnauthorizedDataLakeAccessError from None
        if res.status_code == 404:
            _logger.debug(f"Resource Not Found Error: {res.status_code} {response_message}")
            raise ResourceNotFoundError(RESOURCE_NOT_FOUND_ERROR_HOLDING) from None
