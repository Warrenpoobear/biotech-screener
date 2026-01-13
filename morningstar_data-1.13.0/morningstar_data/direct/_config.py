import base64
import os
import re
from dataclasses import dataclass
from os import environ
from typing import Dict

import simplejson as json

from ._exceptions import (
    CredentialsException,
    EAMSTokenDoesNotExistError,
    MalformedJWTError,
)


@dataclass
class _Config:
    def __init__(self) -> None:
        self._MD_API: str = os.getenv("MD_API", "https://www.us-api.morningstar.com/md-api/")

    def get_uim_token(self) -> str:
        token = os.getenv("MD_AUTH_TOKEN", os.getenv("UIM_JWT", ""))
        if token == "":
            raise CredentialsException("Please set up MD_AUTH_TOKEN environment variable to access the morningstar_data package")
        return self._filter_invalid_jwt(token)

    def get_headers(self, add_headers: Dict[str, str] = {}) -> Dict[str, str]:
        headers = {
            "X-API-ComponentId": os.getenv("DO_API_REQUEST_ORIGIN", "analyticslab"),
            "X-API-Sourceapp": os.getenv("DO_API_REQUEST_ORIGIN", "morningstar-data"),
            "X-API-ProductId": "Direct",
            "Content-Type": "application/json",
            **add_headers,
        }

        # This will allow us to make DO API requests on behalf of another user.
        # EAMS impersonation is going away. We are using this temporarily.
        if environ.get("USE_EAMS_IMPERSONATION"):
            # EAMS_JWT is always on the environment in Analytics Lab.
            # We only want to use it in DNA Lab api when USE_EAMS_IMPERSONATION is set to True
            if "EAMS_JWT" in environ:
                headers["X-API-JsonWebToken"] = environ["EAMS_JWT"]
            else:
                raise EAMSTokenDoesNotExistError("EAMS_JWT does not exist on the environment.")
        else:
            headers["authorization"] = f"Bearer {self.get_uim_token()}"

        return headers

    def object_service_url(self) -> str:
        return f"{self._MD_API}proxy_request/object_service/"

    def custom_data_points_service_url(self) -> str:
        return f"{self._MD_API}proxy_request/custom_data_points_service/"

    def securitydata_service_url(self) -> str:
        return f"{self._MD_API}proxy_request/security_data_service/"

    def searches_service_url(self) -> str:
        return f"{self._MD_API}proxy_request/searches_service/"

    def investment_data_service_url(self) -> str:
        return f"{self._MD_API}v0/investment/investment_data"

    def portfolio_service_url(self) -> str:
        return f"{self._MD_API}proxy_request/portfolio_service/"

    def performancereport_service_url(self) -> str:
        return f"{self._MD_API}proxy_request/performancereport_service/"

    def performancereport_export_service_url(self) -> str:
        return f"{self._MD_API}proxy_request/performancereport_export_service/"

    def dataset_service_url(self) -> str:
        return f"{self._MD_API}proxy_request/dataset_service/"

    def data_point_service_url(self) -> str:
        return f"{self._MD_API}proxy_request/data_point_service/"

    def asset_service_url(self) -> str:
        return f"{self._MD_API}proxy_request/asset_service/"

    def peergroup_service_url(self) -> str:
        return f"{self._MD_API}proxy_request/peergroup_service/"

    def fof_api_url(self) -> str:
        return f"{self._MD_API}proxy_request/fof_service/"

    def al_proxy_api_url(self) -> str:
        return f"{self._MD_API}proxy_request/al_proxy_api/"

    def mds_api_url(self) -> str:
        return f"{self._MD_API}proxy_request/mds_api/"

    def temp_table_url(self) -> str:
        return f"{self._MD_API}proxy_request/temp_table_service/"

    def query_api_url(self) -> str:
        return f"{self._MD_API}proxy_request/lakehouse_query_service/"

    def _filter_invalid_jwt(self, jwt: str) -> str:
        # Define the regular expression pattern to match a valid JWT
        pattern = r"^[A-Za-z0-9_-]*\.?[A-Za-z0-9_-]*\.?[A-Za-z0-9_-]*$"

        # Check if the JWT matches the valid pattern
        if not re.match(pattern, jwt):
            # Replace invalid characters with an empty string
            jwt = re.sub(r"[^\w\.-]", "", jwt)

        # jwt = jwt.strip()
        # # Use regex to replace
        # jwt = re.sub(pattern, "", jwt)

        # Check that the JWT has 3 parts
        if len(jwt.split(".")) != 3:
            raise MalformedJWTError

        # Split the JWT into header, payload, and signature
        header, payload, signature = jwt.split(".")

        # Filter out any invalid characters from the header and payload
        decoded_header = base64.urlsafe_b64decode(header + "=" * (4 - len(header) % 4)).decode("utf-8", "ignore")
        decoded_payload = base64.urlsafe_b64decode(payload + "=" * (4 - len(payload) % 4)).decode("utf-8", "ignore")

        # Check that the filtered header and payload are valid JSON objects
        try:
            json.loads(decoded_header)
            json.loads(decoded_payload)
        except ValueError:
            raise MalformedJWTError from None

        # If we got here, the JWT is valid and filtered
        return ".".join([header, payload, signature])
