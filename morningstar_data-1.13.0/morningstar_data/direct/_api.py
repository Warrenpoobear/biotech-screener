from typing import Any, Dict, Optional, Union

import requests

# NOTE! We are not able to verify the SSL cert on Direct Search API since
#       the VPC endpoint domain name is not on the server's SSL cert.
#       So with a heavy heart, we disable the InsecureRequestWarning
from urllib3.exceptions import InsecureRequestWarning

from .._base import _logger
from ._config import _Config
from ._core_api import make_request
from ._exceptions import ApiRequestException, ApiResponseException, QueryLimitException

_config = _Config()

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


def _direct_api_request(method: str, url: str, data: Optional[Union[Dict[Any, Any], str]] = None) -> Any:
    """
    Generic request method for the Direct API which covers headers and aspects
    of response handling that are consistent across API calls

    :method: Request method, eg. GET or POST
    :url: Request URL
    :data: (optional) POST message body, if any
    """
    headers = _config.get_headers()
    _logger.debug(f"Requesting: {method} {url} with headers: {headers}")
    try:
        res = make_request(method, url, headers=headers, verify=False, data=data)
        res.raise_for_status()

        return res.json()
    except requests.ConnectionError as e:
        _logger.error(repr(e))
        raise ApiRequestException(f"Failed connecting to {url}")
    except requests.HTTPError:
        message = f"Received {res.status_code} error from {url}"
        json_response = res.json()
        if res.status_code == 403 and "Exceed query limitation." in json_response["message"]:
            query_limit = _get_security_data_query_limit()
            raise QueryLimitException(query_limit) from None
        else:
            if "_meta" in json_response and "hint" in json_response["_meta"]:
                message += f" ({json_response['_meta']['hint']})"
            elif "message" in json_response:
                message += f" ({json_response['message']})"
            raise ApiResponseException(message, res.status_code)


def _get_security_data_query_limit() -> str:
    try:
        headers = _config.get_headers()
        url = f"{_config.securitydata_service_url()}v1/limitation/summary"

        res = make_request("GET", url, headers=headers, verify=False)
        json_response = res.json()
        return str(json_response["limitationTotal"])
    except Exception:
        return ""
