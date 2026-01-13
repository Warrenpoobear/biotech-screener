import os
import uuid
from typing import Any, Dict, Optional, Union

import polling2
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .._base import _logger
from .._version import __version__
from ..direct._error_messages import INTERNAL_SERVER_ERROR
from ._config import _Config
from ._exceptions import ApiRequestException, ApiResponseException, InternalServerError
from ._utils import get_bytes

_config = _Config()

POLL_STEP = 1  # seconds
POLL_TIMEOUT = 900  # seconds (15 minutes)
HEADER_STATUS = "proxy_response_status"


def get_poll_step_from_headers(headers: Dict[str, Any]) -> int:
    return int(headers["POLL_STEP"]) if "POLL_STEP" in headers else POLL_STEP


def get_poll_timeout_from_headers(headers: Dict[str, Any]) -> int:
    return int(headers["POLL_TIMEOUT"]) if "POLL_TIMEOUT" in headers else POLL_TIMEOUT


def is_task_complete(response: Any) -> bool:
    if "proxy_response_status" not in response.headers:
        return False
    if (
        response.headers["proxy_response_status"].upper() == "SUCCESS"
        or response.headers["proxy_response_status"].upper() == "FAILURE"
    ):
        return True
    else:
        return False


@retry(reraise=True, stop=stop_after_attempt(8), wait=wait_exponential(multiplier=2, min=1, max=120))  # type: ignore
def request_with_retry(
    method: str, url: str, headers: Dict[str, Any], verify: bool, data: Optional[Union[Dict[Any, Any], str]] = None
) -> requests.Response:
    STATUS_FORCE_LIST = [500, 501, 502, 503, 504]
    MAX_REDIRECTS = 5
    REDIRECT_CODES = [301, 307]
    try:
        res = requests.request(method, url, headers=headers, verify=verify, data=data, allow_redirects=False)

        redirect_count = 0

        while res.status_code in REDIRECT_CODES and redirect_count <= MAX_REDIRECTS:
            redirected_url = res.headers["Location"]
            _logger.info(f"Redirection to {redirected_url} for {method} request to {url}")

            res = requests.request(method, redirected_url, headers=headers, verify=verify, data=data, allow_redirects=False)

            redirect_count += 1

    except Exception as e:
        _logger.error(f"Retry Error: Exception making request. method: {method},url: {url},headers: {headers} , error: {e}")
        if "proxy_request" in url:
            raise ApiRequestException(f"Failed connecting to {url}, error: {e}")
        else:
            raise ApiResponseException(f"Failed connecting to {url}, error: {e}")

    if res.status_code in STATUS_FORCE_LIST:
        # Build a message from the response if we can
        try:
            res_json = res.json()
        except requests.JSONDecodeError:
            raise InternalServerError(
                f"{INTERNAL_SERVER_ERROR}. Got status code: {res.status_code} for method: {method} to {url}."
            )

        error_msg = f"{INTERNAL_SERVER_ERROR}. Got status code: {res.status_code} for method: {method} to {url}."

        try:
            hint = res_json["_meta"]["hint"]
            error_msg = error_msg + f" hint: {hint}"
        except KeyError:
            pass

        try:
            message = res_json["message"]
            error_msg = error_msg + f" message: {message}"
        except KeyError:
            pass

        raise InternalServerError(error_msg)

    return res


# This function checks whether the response from a function is a presigned URL, and if it is, it downloads the content from that URL and returns the response object.
# If the response is not a presigned URL, it returns the response object.
def download_if_signed_url(response: requests.Response) -> requests.Response:
    if response.headers.get("download_presigned_url") == "True":
        download_url = response.content
        _logger.info(f"Downloading from s3 at the url {download_url}")
        res_s3_byte_object = get_bytes(download_url)
        # Note that _content is a private attribute, so we should not normally modify it. However, in this case, we need to modify it to change the value of the content attribute.
        response._content = res_s3_byte_object.content
        return response
    else:
        return response


def make_request(
    method: str, url: str, headers: Dict[str, Any], verify: bool, data: Optional[Union[Dict[Any, Any], str]] = None
) -> requests.Response:
    request_id = None
    poll_step = POLL_STEP
    poll_timeout = POLL_TIMEOUT
    try:
        _logger.info("Queueing request")
        _logger.debug(f"Queueing request: {method} {url}")
        headers["md-package-version"] = __version__

        if "X-API-RequestId" not in headers.keys() and "x-api-requestid" not in headers.keys():
            headers["X-API-RequestId"] = str(uuid.uuid4())

        if os.getenv("REQUEST_ID", "") != "":
            headers["X-API-CorrelationId"] = os.getenv("REQUEST_ID")

        if os.getenv("FEED_ID"):
            headers["x-feed-id"] = os.getenv("FEED_ID")

        request_uuid = headers.get("X-API-RequestId", headers.get("x-api-requestid"))
        _logger.debug(f"Updated Request ID: {request_uuid}")
        res = request_with_retry(method, url, headers=headers, verify=verify, data=data)
        res.raise_for_status()

        _logger.info("Got valid response from queueing request")
        _logger.debug(f"Got {res.status_code} response from queueing request.")

        _logger.debug(f"Response headers: {res.headers}")

        proxy_status = res.headers[HEADER_STATUS].upper()
        if proxy_status == "SUCCESS":
            _logger.debug("Request is already completed. Returning the response.")
            return res
        elif proxy_status == "QUEUED":
            _logger.debug("Request is still queued. Getting the request id")
            request_id = res.json()["id"]
            _logger.debug(f"Client Request ID {request_uuid}. MD-API Task ID {request_id}")
            _logger.info(f"Request id is {request_id}")
            poll_step = get_poll_step_from_headers(res.headers)
            poll_timeout = get_poll_timeout_from_headers(res.headers)

    except requests.ConnectionError as e:
        _logger.error(repr(e))
        raise ApiRequestException(f"Failed connecting to {url}")
    except requests.HTTPError as e:
        # If the request is already completed, return the response.
        # The response is considered successful if the proxy_response_status in the header is "completed"
        # Every response from MD API has a proxy_response_status header
        _logger.info("Got HTTPError from queueing request")
        # Access the value of "proxy_response_status" in res.headers dictionary, with a default value of an empty string ("") if the key is not found
        # if res.headers.get("proxy_response_status", "").upper() == "SUCCESS":
        #     _logger.debug("Request is already completed. Returning the response.")
        #     return res

        _logger.error(repr(e))
        raise ApiResponseException("Got HTTPError from queueing request", e.response.status_code)

    response_url = f"{_config._MD_API}proxy_response/{request_id}"
    _logger.debug(f"Response URL is {response_url}")

    # Poll the response URL until the request is completed
    # Return the response object as it is as exception handling is done in the calling function
    # Response is considered successful if the proxy_response_status in the header is "completed"

    # time.sleep(5)

    # return requests.get(response_url, headers=headers, verify=verify)

    response = polling2.poll(
        lambda: request_with_retry("GET", response_url, headers=headers, verify=verify),
        step=poll_step,
        timeout=poll_timeout,
        check_success=lambda response: is_task_complete(response),
    )
    return download_if_signed_url(response=response)
