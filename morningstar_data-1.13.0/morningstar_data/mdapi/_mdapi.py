"""
Here lie methods to interact with the Morningstar Data API. These methods are not intended to be used directly by
users of the library. They will be used by the public methods in the morningstar_data package.
"""

import dataclasses
import io
import os
import time
import uuid
import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests
import urllib3

from .._utils import join_url
from .._version import __version__
from ..direct import _decorator
from ..direct._config import _Config
from ..direct._data_type import DryRunResults
from ..direct.data_type import (
    EnrichmentType,
    Frequency,
    HoldingsView,
    InvestmentIdentifier,
    PortfolioDepth,
)
from ._exceptions import (
    MdApiRequestException,
    MdApiTaskException,
    MdApiTaskTimeoutException,
    MdBaseException,
    exception_by_name,
)
from ._types import HoldingDatesResult, InvestmentLookupResult, MdapiTask, RequestObject, TaskResult

_config = _Config()

# Timeouts will generally be managed in the API, but just in case, let's have a fallback super-long timeout here
POLL_TIMEOUT = int(os.environ.get("MD_TASK_TIMEOUT_SECONDS", 10_000))

MD_API_V1_BASE = join_url(os.environ.get("MD_API", "https://www.us-api-proxy.morningstar.com/md-api"), "/v1")
MD_API_REQUEST_TIMEOUT = 5  # seconds
S3_CONNECTION_TIMEOUT = 5  # seconds
S3_READ_TIMEOUT = 60  # seconds

SEARCH_SECURITY_ENDPOINT = "search_security"
HOLDING_DATE_ENDPOINT = "get_holding_dates"
HOLDINGS_TASK_ENDPOINT = "get_holdings_task"
LOOKTHROUGH_HOLDINGS_TASK_ENDPOINT = "get_lookthrough_holdings_task"

NUMBER_OF_INVESTMENTIDENTIFIERS_ALLOWED = 500


@dataclasses.dataclass
class AsyncHoldingsRequest(RequestObject):
    investments: Optional[Union[List[str], str, Dict[str, Any], List[InvestmentIdentifier]]]
    date: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    data_points: Optional[Union[List[Dict[str, Any]], str, pd.DataFrame, List[Any]]]
    dates: Optional[List[str]]
    frequency: Optional[str] = Frequency.monthly.abbr
    holdings_view: Union[int, HoldingsView] = HoldingsView.FULL
    preview: Optional[bool] = False
    enrichment_type: EnrichmentType = EnrichmentType.POINT_IN_TIME
    suppression_client_id: Optional[str] = None
    delta_start_time: Optional[str] = None
    delta_include_unchanged_data: Optional[bool] = None
    portfolio_depth: PortfolioDepth = PortfolioDepth.BASE_LEVEL

    def check_size_of_request(self) -> None:
        if (
            self.investments is not None
            and isinstance(self.investments, list)
            and len(self.investments) > NUMBER_OF_INVESTMENTIDENTIFIERS_ALLOWED
        ):
            if all(isinstance(item, InvestmentIdentifier) for item in self.investments):
                raise ValueError(
                    f"The size of investments list should not be greater than {NUMBER_OF_INVESTMENTIDENTIFIERS_ALLOWED} if of type List[InvestmentIdentifier]"
                )


def _get_headers() -> Dict[str, str]:
    request_id = str(uuid.uuid4())
    auth_token = _config.get_uim_token()

    headers = {
        "authorization": f"Bearer {auth_token}",
        "md-correlation-id": request_id,
        "md-request-id": request_id,
        "md-package-version": __version__,
        "md-direct-source-app": os.getenv("DO_API_REQUEST_ORIGIN", "morningstar-data"),
    }
    if os.getenv("FEED_RUN_ID"):
        headers["md-feed-run-id"] = os.getenv("FEED_RUN_ID")

    if os.getenv("REQUEST_ID"):  # When running on al-api pods, this env var will be present
        headers["md-feed-run-attempt-id"] = os.getenv("REQUEST_ID")
        headers["md-correlation-id"] = os.getenv("REQUEST_ID")

    if os.getenv("FEED_ID"):
        headers["md-feed-id"] = os.getenv("FEED_ID")

    return headers


def _read_feather_from_s3(url: str) -> pd.DataFrame:
    if url == "":
        return pd.DataFrame()
    download_response = requests.get(url, timeout=(S3_CONNECTION_TIMEOUT, S3_READ_TIMEOUT))
    download_response.raise_for_status()
    feather_content = io.BytesIO(download_response.content)
    return pd.read_feather(feather_content)


def _wait_for_completed_task_response(task: MdapiTask) -> MdapiTask:
    task_status_request_session = requests.Session()
    task_status_request_session.mount(
        "https://",
        requests.adapters.HTTPAdapter(
            max_retries=urllib3.util.Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        ),
    )

    task_start_time = int(time.time())
    while not task.is_complete():
        if time.time() - task_start_time > POLL_TIMEOUT:
            raise MdApiTaskTimeoutException(task_id=task.id, timeout_time=POLL_TIMEOUT)
        time.sleep(task.next_poll_delay_milliseconds / 1000.0)
        task = _check_task_status(task.poll_url, task_start_time, task_status_request_session)
    return task


def _check_task_status(polling_url: str, task_start_time: float, task_status_request_session: requests.Session) -> MdapiTask:
    request_headers = _get_headers()
    try:
        response = task_status_request_session.get(
            polling_url,
            headers=request_headers,
            timeout=MD_API_REQUEST_TIMEOUT,
            params={"task_start_time": task_start_time},
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise MdApiRequestException(request_id=request_headers["md-request-id"], url=polling_url, detail=str(e)) from e

    return MdapiTask(**response.json())


def _create_task(url: str, request_object: RequestObject) -> MdapiTask:
    response = _request_md_api("POST", url, request_object)
    return MdapiTask(**response)


def _request_md_api(
    http_method: str, url: str, request_object: Optional[RequestObject] = None, params: Optional[dict] = None
) -> dict:
    request_headers = _get_headers()
    try:
        if http_method == "POST":
            if request_object:
                response = requests.post(
                    url, json=dataclasses.asdict(request_object), headers=request_headers, timeout=MD_API_REQUEST_TIMEOUT
                )
            else:
                raise NotImplementedError  # Implement if we need to support POST with an empty body
        else:
            params = params or {}
            response = requests.get(url, params=params, headers=request_headers, timeout=MD_API_REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.HTTPError:
        request_id = request_headers["md-request-id"]
        try:
            response_json = response.json()
        except requests.exceptions.JSONDecodeError:
            message = f"Unexpected API error format. Response body: '{response.text}'"
            raise MdApiRequestException(request_id=request_id, url=url, detail=message) from None
        raise exception_by_name(
            response_json.get("error_class", ""), request_id=request_id, message=response_json.get("error_message", "")
        ) from None
    except requests.RequestException as e:
        raise MdApiRequestException(request_id=request_headers["md-request-id"], url=url, detail=str(e)) from e

    return dict(response.json())


def call_remote_function(method: str, request_object: RequestObject) -> pd.DataFrame:
    url = join_url(MD_API_V1_BASE, method)

    task = _create_task(url, request_object)
    try:
        response = _wait_for_completed_task_response(task)
        # The API sets the dataframe_file_url if a dataframe is empty
        if response.is_successful():
            # response.result should always be present, but check for mypy's sake
            task_result = response.result or TaskResult(dataframe_file_url="")
            df = _read_feather_from_s3(task_result.dataframe_file_url)
            for alias in task_result.columns_with_list_values:
                # Maintaining some legacy behavior here. If the column has only one value, we'll convert it to a singular value.
                # We may want to consider changing this as a breaking change in the future.
                df[alias] = df[alias].apply(lambda x: x[0] if len(x) == 1 else list(x))
            if task_result.warning_messages:
                for warn_text in task_result.warning_messages:
                    warnings.warn(warn_text, Warning)

            return df

        if response.failed():
            raise exception_by_name(str(response.error_class), task_id=task.id, message=str(response.error_message)) from None

    # Let's catch any unhandled exceptions and attach the task id so it can be shared for troubleshooting help.
    # We'll ignore our custom exceptions since they already have all the detail we would want.
    except MdBaseException:
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        raise MdApiTaskException(task_id=str(task.id), detail=repr(e)) from e


def search_security(investment_identifier: InvestmentIdentifier, count: int, only_surviving: bool) -> InvestmentLookupResult:
    url = join_url(MD_API_V1_BASE, SEARCH_SECURITY_ENDPOINT)

    params = dataclasses.asdict(investment_identifier)
    params.update({"count": count, "only_surviving": only_surviving})

    response = _request_md_api("GET", url, params=params)
    return InvestmentLookupResult(**response)


def get_holding_dates(holding_date_request: RequestObject) -> HoldingDatesResult:
    url = join_url(MD_API_V1_BASE, HOLDING_DATE_ENDPOINT)

    response = _request_md_api("POST", url, request_object=holding_date_request)

    result = HoldingDatesResult(holdings_dates=response.get("holdings_dates", []))

    return result


def init_holdings_async(async_holdings_request: AsyncHoldingsRequest) -> MdapiTask:
    url = join_url(MD_API_V1_BASE, HOLDINGS_TASK_ENDPOINT)
    response = _request_md_api("POST", url, request_object=async_holdings_request)
    return MdapiTask(**response)


def init_lookthrough_holdings_async(async_holdings_request: AsyncHoldingsRequest) -> MdapiTask:
    url = join_url(MD_API_V1_BASE, LOOKTHROUGH_HOLDINGS_TASK_ENDPOINT)
    response = _request_md_api("POST", url, request_object=async_holdings_request)
    return MdapiTask(**response)


TASK_STATUS_ENDPOINT = "task"


@_decorator.typechecked
def get_task_status(job_id: str) -> MdapiTask:
    """
    Retrieves the status of an asynchronous task given its ID.

    Args:
        job_id (str): The unique identifier of the asynchronous task.

    :Returns:
        MdapiTask: An object representing the status and details of the asynchronous task.
            The MdapiTask object contains the following attributes:

            - id: Unique identifier for the task
            - poll_url: URL endpoint for polling the task status
            - status: Current status of the task (SUCCESS, PENDING, or FAILURE)
            - result (Optional[TaskResult]): Contains the task results. This field is an object of type TaskResult, which contain below attributes.
              - dataframe_file_url
              - columns_with_list_values
              - warning_messages
              - pages: A list of Page objects representing paginated data. Each Page object contains below fields.
                - url: Pass this URL to `get_holdings_task_result <./holdings_async.html#morningstar_data.direct.get_holdings_task_result>`_ to fetch a dataframe for this page. The URL expires after a certain period (see expiry_time below).
                - id: Page number
                - expiry_time: Expiry time of the URL. After this time, the URL will no longer work.
            - error_class: Exception class name if the task failed
            - error_message: Error message if the task failed
            - task_timeout_seconds: Maximum time to wait for task completion

    :Example::
        status = md.mdapi.get_task_status('10000000-0000-0000-0000-000000000000')
        status

    :Output:
        MdapiTask object with task status and details::

            MdapiTask(
                id='10000000-0000-0000-0000-000000000000',
                poll_url='https://datalab.morningstar.com/v1/task/10000000-0000-0000-0000-000000000000',
                status='success',
                result=TaskResult(
                    dataframe_file_url='',
                    columns_with_list_values=[],
                    warning_messages=None,
                    pages=[
                            Page(
                                url='https://....',
                                id=1,
                                expiry_time='2025-10-17T16:48:07.932212814'
                            )
                        ]
                ),
                error_class=None,
                error_message=None,
                next_poll_delay_milliseconds=1,
                task_timeout_seconds=900
            )

    """
    url = join_url(MD_API_V1_BASE, TASK_STATUS_ENDPOINT, job_id)
    response = _request_md_api("GET", url)
    return MdapiTask(**response)


@_decorator.typechecked
def get_holding_dry_run_results(async_holdings_request: AsyncHoldingsRequest) -> DryRunResults:
    """
    This function is used to estimate the number of cells that will be used by a holdings request.
    It does not return any data, only an estimate of the number of cells that would be used.
    """
    url = join_url(MD_API_V1_BASE, "get_holdings_dry_run")
    response = _request_md_api("POST", url, request_object=async_holdings_request)

    response.pop("daily_cells_remaining_after", None)
    return DryRunResults(**response)
