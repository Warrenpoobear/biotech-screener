import logging
import os
from typing import Any, Dict, List, Union

import polling2
import simplejson as json
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

from ..direct import _decorator
from ..direct._backend_apis import DeliveryAPIBackend
from ..direct._error_messages import FAILED_DELIVERY_ERROR
from ..direct._exceptions import FailedDeliveryError
from ._delivery_config import DeliveryConfig, DeliveryType
from ._helpers import (
    _create_feed_body,
    _create_output_file_name,
    _encrypt_ftp_password,
    _get_bucket_name,
    _get_job_name,
    _get_mds_base_url,
    _get_mds_bucket,
    _get_user_id,
)

_logger = logging.getLogger(__name__)
_api_backend = DeliveryAPIBackend()


def _create_feed_id() -> int:
    """Calls the MDS feeds api to generate a new feed id.

    This feed id is needed when calling the MDS deliver api. In general, when a notebook is running
    in a schedule it will be passed in this feed id. But for an ad-hoc delivery this feed id needs
    to be created before the MDS delivery api can be called.
    """
    _logger.info("Creating a feed id using MDS api")
    url = f"{_get_mds_base_url()}v1/feeds?type=notebook"
    data = _create_feed_body()

    try:
        _logger.info(f"Calling MDS /feeds endpoint with: {data}")
        response_json = _api_backend.do_post_request(url, data=json.dumps(data, ignore_nan=True))
        feed_id: int = response_json["feedId"]
        return feed_id
    except Exception:
        _logger.error("Could not create a feed id using MDS api")
        raise


def _upload_to_s3(file_path: str, output_file_name: str) -> None:
    """Uploads a file to the MDS s3 bucket"""
    try:
        config = TransferConfig(
            multipart_threshold=5 * 1024 * 1024 * 1024,  # 5GB
            max_concurrency=10,
            multipart_chunksize=500 * 1024 * 1024,  # 500MB
            use_threads=True,
        )

        output_bucket = _get_mds_bucket()
        output_bucket.upload_file(file_path, output_file_name, Config=config)
    except FileNotFoundError:
        _logger.error(f"File not found: {file_path}, make sure the file you are trying to upload exists")
        raise
    except ClientError:
        _logger.error("Could not upload to s3 bucket due to AWS error")
        raise
    except Exception:
        _logger.error("Could not upload to s3 bucket")
        raise


def _process_configs(configs: List[DeliveryConfig]) -> Dict[str, List]:
    unique_files = set()
    unique_emails = set()
    unique_ftps = set()
    unique_profiles = set()

    file_mappings = []

    # Processing each config
    for config in configs:
        if config.method == DeliveryType.EMAIL:
            unique_emails.add(config.email_address)
        elif config.method == DeliveryType.DELIVERY_PROFILE:
            unique_profiles.add(config.delivery_profile_id)
        else:
            unique_ftps.add((config.user_name, config.password, config.server, config.folder))
        unique_files.add((config.file_path, config.delivered_file_name, config.include_timestamp))

    # Converting sets to lists to maintain order and for indexing
    unique_files_list = list(unique_files)
    unique_emails_list = list(unique_emails)
    unique_ftps_list = list(unique_ftps)
    unique_profiles_list = list(unique_profiles)

    # Creating mappings
    for config in configs:
        file_index = unique_files_list.index((config.file_path, config.delivered_file_name, config.include_timestamp))
        if config.method == DeliveryType.EMAIL:
            email_index = unique_emails_list.index(config.email_address)
            mapping = {"fileIndex": f"{file_index}", "deliveryInfo": f"emails.{email_index}"}
        elif config.method == DeliveryType.DELIVERY_PROFILE:
            profile_index = unique_profiles_list.index(config.delivery_profile_id)
            mapping = {"fileIndex": f"{file_index}", "deliveryInfo": f"profiles.{profile_index}"}
        else:
            ftp_index = unique_ftps_list.index((config.user_name, config.password, config.server, config.folder))
            mapping = {"fileIndex": f"{file_index}", "deliveryInfo": f"ftps.{ftp_index}"}

        file_mappings.append(mapping)

    # Remove duplicate mappings
    file_mappings = [dict(t) for t in {tuple(d.items()) for d in file_mappings}]

    files_list = [
        {
            "key": file[0],
            "fileName": file[1],
            "bucket": _get_bucket_name(),
            "includeTimestamp": str(file[2]),
        }
        for file in unique_files_list
    ]
    emails_list = [{"address": email} for email in unique_emails_list]
    profiles_list = [{"deliveryProfileId": profile} for profile in unique_profiles_list]
    ftps_list = [
        {"username": ftp[0], "password": _encrypt_ftp_password(ftp[1]), "server": ftp[2], "folder": ftp[3]}
        for ftp in unique_ftps_list
    ]

    return {
        "files": files_list,
        "emails": emails_list,
        "profiles": profiles_list,
        "ftps": ftps_list,
        "delivery_details": file_mappings,
    }


def _create_deliver_body(configs: List[DeliveryConfig]) -> Dict:
    """Creates the post body for calling  MDS delivery API with single or multiple files/delivery destinations."""

    # convert list of configs to mds post body format
    config_mapping = _process_configs(configs=configs)

    body = {
        "userId": _get_user_id(),
        "userName": _get_user_id(),
        "jobName": _get_job_name(),
        "files": config_mapping["files"],
        "deliveryInfos": {
            "emails": config_mapping["emails"],
            "profiles": config_mapping["profiles"],
            "ftps": config_mapping["ftps"],
        },
        "deliveryDetails": config_mapping["delivery_details"],
    }
    return body


def _mds_deliver_file(config: List[DeliveryConfig], feed_id: int) -> str:
    """Calls the MDS delivery api with the needed information to deliver the uploaded file and returns job_id"""
    url = f"{_get_mds_base_url()}v1/delivery/{feed_id}"
    data = _create_deliver_body(config)
    _logger.info(f"Calling MDS /delivery/{feed_id} endpoint with: {data}")
    response_json = _api_backend.do_post_request(url, data=json.dumps(data, ignore_nan=True))
    _logger.info(f"Delivery response: {response_json}")
    job_id: str = response_json["jobId"]
    return job_id


def _process_file_paths(delivery_configs: List[DeliveryConfig]) -> Dict:
    """Get list of unique files/file paths to upload in S3"""
    processed_file_paths = {}
    for config in delivery_configs:
        file_path = config.file_path
        if file_path not in processed_file_paths:
            # create output filename to upload in s3
            output_file_name = _create_output_file_name(file_path)
            processed_file_paths[file_path] = output_file_name
        else:
            output_file_name = processed_file_paths[file_path]
        config.file_path = output_file_name
    return processed_file_paths


@_decorator.typechecked
def get_delivery_profile() -> Dict:
    """Fetch ftp and S3 delivery profiles using MDS API
    Returns:
        :obj:`dict`: FTP and S3 delivery profiles for the user.
    """
    url = f"{_get_mds_base_url()}v1/delivery-profile"
    _logger.info("Calling MDS /delivery-profile endpoint. ")
    mds_delivery_profiles: Dict[Any, Any] = _api_backend.do_get_request(url)
    _logger.info(f"MDS delivery profile response: {mds_delivery_profiles}")
    delivery_profiles = {"ftps": mds_delivery_profiles.get("ftps", []), "s3s": mds_delivery_profiles.get("s3s", [])}
    return delivery_profiles


@_decorator.typechecked
def delivery_status(job_id: str, poll: bool = True) -> str:
    """Polls the MDS delivery api with the job_id to check the delivery status.
    Args:
        job_id (:obj:`str`): The job id that we get from MDS endpoint
        poll (:obj:`bool`): Flag to poll the endpoint or not (default is set as True,
        as we want to poll the endpoint to get the status)

    Returns:
        :obj:`str`: Delivery status as given by the endpoint.
    """
    url = f"{_get_mds_base_url()}v1/jobs/{job_id}/status"
    _logger.info(f"Calling MDS /delivery/{job_id}/status endpoint")

    # polling the endpoint till it gets status response; raise exception error if occurred
    def _delivery_finished(res: dict) -> bool:
        try:
            return res["jobStatus"].lower() in ["done", "failed"]
        except KeyError as e:
            _logger.error(f"Could not find delivery status. Exception as {e}")
            raise

    if poll:
        _logger.info("Polling the request to get the delivery status.")
        try:
            polling2.poll(
                lambda: _api_backend.do_get_request(url),
                step=5,
                timeout=60,
                check_success=lambda response: _delivery_finished(response),
            )
        except polling2.TimeoutException as te:
            _logger.info(f"Timeout for the request. Exception as {te}.")

    response_json = _api_backend.do_get_request(url)
    _logger.info(f"Delivery status: {response_json}")

    return str(response_json["jobStatus"])  # "DONE" or "FAILED" or "RUNNING"


@_decorator.typechecked
def deliver(
    config: Union[DeliveryConfig, List[DeliveryConfig]],
    wait_for_delivery: bool = False,
) -> dict:
    """

    Delivers a file from a notebook to an email address, FTP location, or Delivery Profile.

    Args:
        config (:obj:`md.utils.DeliveryConfig`): An object that holds the delivery configuration information.

            All delivery methods (email, FTP, Delivery Profile) contains
                * file_path (:obj: `str`, `required`): the path to the file that will be delivered, including file name and extension.
                * delivered_file_name (:obj: `str`, `optional`): parameter for the user to specify the file name to be used when
                    delivered. No file extension should be provided. If it's omitted, the file_path value will use as default filename.
                *  include_timestamp (:obj: `bool`, `optional`): optional parameter for the user append a timestamp to the file name. Default value will be False. The format will be YYYYMMDDHHMMSS.

            For an email, it contains
                * method (:obj:`DeliveryType`): ``DeliveryType.EMAIL`` in this case.
                * email_address (:obj:`str`): email_address to send the file to.
            For FTP, it contains
                * method (:obj:`DeliveryType`): ``DeliveryType.FTP`` in this case.
                * user_name (:obj:`str`): the user_name to login to the ftp server.
                * password (:obj:`str`): the password to login to the ftp server.
                * server (:obj:`str`): the ftp server to use.
                * folder (:obj:`str`): the folder on the ftp server to upload the file to.
            For a Delivery Profile, it contains
                * method (:obj:`DeliveryType`): ``DeliveryType.DELIVERY_PROFILE`` in this case.
                * delivery_profile_id (:obj:`str`): delivery_profile_id that was setup by MDS. Can be retrieved with get_delivery_profile()
        wait_for_delivery (:obj:`bool`): Flag to poll the api to check the delivery status or not. (True=show status)

    :Returns:
        :obj:`dict`: A dictionary with the 'job_id', 'message', 'delivery_status' keys containing information about the delivery status

    :Examples:

    Deliver to an email address.

    ::

        import morningstar_data as md
        import pandas as pd

        df = pd.DataFrame({'a':[1], 'b':[2]})
        df.to_csv("test_export.csv")

        # Email example
            * For single file
                delivery_config = md.utils.DeliveryConfig(method=md.utils.DeliveryType.EMAIL, email_address="test@email.com", file_path="test_export.csv", delivered_file_name="Delivered_File_123", include_timestamp=True)
                md.utils.deliver(config=delivery_config)

            * For multiple files
                df = pd.DataFrame({'a':[1], 'b':[2]})
                df.to_csv("test_export2.csv")

                delivery_config1 = md.utils.DeliveryConfig(file_path="test_export.csv", method=md.utils.DeliveryType.EMAIL, email_address="test1@email.com", delivered_file_name="Delivered_File_123", include_timestamp=True)
                delivery_config2 = md.utils.DeliveryConfig(file_path="test_export2.csv", method=md.utils.DeliveryType.EMAIL, email_address="test2@email.com", delivered_file_name="Delivered_File_123", include_timestamp=True)
                md.utils.deliver([delivery_config1, delivery_config2])
    Deliver to a ftp server.

    ::

        import morningstar_data as md
        import pandas as pd

        df = pd.DataFrame({'a':[1], 'b':[2]})
        df.to_csv("test_export.csv")

        # FTP example
        delivery_config = md.utils.DeliveryConfig(method=md.utils.DeliveryType.FTP, user_name="test", password="test", server="ts.ftp.com", folder="data/", file_path="test_export.csv", delivered_file_name="Delivered_File_123")
        md.utils.deliver(config=delivery_config)

    Deliver to a Delivery Profile

    ::

        import morningstar_data as md
        import pandas as pd

        df = pd.DataFrame({'a':[1], 'b':[2]})
        df.to_csv("test_export.csv")

        # Delivery Profile example
        delivery_config = md.utils.DeliveryConfig(method=md.utils.DeliveryType.DELIVERY_PROFILE, delivery_profile_id="1234", file_path="test_export.csv", delivered_file_name="Delivered_File_123")
        md.utils.deliver(config=delivery_config)

    Multiple delivery within a single notebook.

    ::

        import morningstar_data as md
        import pandas as pd

        df = pd.DataFrame({'a':[1], 'b':[2]})
        df.to_csv("test_export.csv")

        ftp_delivery_config = md.utils.DeliveryConfig(file_path="test_export.csv", method=md.utils.DeliveryType.FTP, user_name="test", password="test", server="ts.ftp.com", folder="data/", delivered_file_name="Delivered_File_123")
        email_delivery_config = md.utils.DeliveryConfig(file_path="test_export.csv", method=md.utils.DeliveryType.EMAIL, email_address="test@email.com", delivered_file_name="Delivered_File_123", include_timestamp=False)
        md.utils.deliver([ftp_delivery_config, email_delivery_config])

    Multiple delivery for multiple files.

    ::

        import morningstar_data as md
        import pandas as pd

        df = pd.DataFrame({'a':[1], 'b':[2]})
        df.to_csv("test_export.csv")

        df = pd.DataFrame({'a':[1], 'b':[2]})
        df.to_csv("test_export2.csv")

        ftp_delivery_config = md.utils.DeliveryConfig(file_path="test_export.csv", method=md.utils.DeliveryType.FTP, user_name="test", password="test", server="ts.ftp.com", folder="data/", delivered_file_name="Delivered_File_123", include_timestamp=False)
        email_delivery_config = md.utils.DeliveryConfig(file_path="test_export2.csv", method=md.utils.DeliveryType.EMAIL, email_address="test@email.com", delivered_file_name="Delivered_File_123", include_timestamp=True)
        md.utils.deliver([ftp_delivery_config, email_delivery_config])

    Deliver with a specific filename.

    ::

        import morningstar_data as md
        import pandas as pd

        df = pd.DataFrame({'a':[1], 'b':[2]})
        df.to_csv("test_export.csv")

        # Email example
        delivery_config = md.utils.DeliveryConfig(method=md.utils.DeliveryType.EMAIL, email_address="test@email.com", file_path="test_export.csv", delivered_file_name="Delivered_Dataframe_123", include_timestamp=True)
        md.utils.deliver(config=delivery_config)
        # As we set include_timestamp to True, the file delivered will be named "Delivered_Dataframe_123_2024022819184484.csv"

        delivery_config = md.utils.DeliveryConfig(method=md.utils.DeliveryType.EMAIL, email_address="test@email.com", file_path="test_export.csv", delivered_file_name="Delivered_Dataframe_123")
        md.utils.deliver(config=delivery_config)
        # If include_timestamp set to False or if omitted, the file delivered will be named "Delivered_Dataframe_123.csv"

    Deliver without the delivered_file_name property

    ::

        import morningstar_data as md
        import pandas as pd

        df = pd.DataFrame({'a':[1], 'b':[2]})
        df.to_csv("test_export.csv")

        delivery_config = md.utils.DeliveryConfig(method=md.utils.DeliveryType.EMAIL, email_address="test@email.com", file_path="test_export.csv", include_timestamp=True)
        md.utils.deliver(config=delivery_config)
        # In this example we omitted the delivered_file_name property, so the file delivered will use the file_path and be named "test_export_2024022819184484.csv"


    :Errors:
        AccessDeniedError: Raised when the user is not authenticated.

        BadRequestError: Raised when the user does not provide a properly formatted request.

        InternalServerError: Raised when the server encounters an unhandled error.

        NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

        FileNotFoundError: Raised when the file path specified to be delivered does not exist
    """
    # Check for a feed_id, create one if not specified
    try:
        feed_id = int(os.environ["FEED_ID"])
    except KeyError:
        feed_id = _create_feed_id()

    # if single config is passed and it is not in list, convert into list
    if not isinstance(config, list):
        config = [config]

    # get list of unique files from delivery configs
    processed_file_paths = _process_file_paths(config)

    # upload files to s3
    for file_path, output_file_name in processed_file_paths.items():
        _upload_to_s3(file_path, output_file_name)

    _logger.info(f"Final config for creating post body: {config}")
    job_id: str = _mds_deliver_file(config, feed_id)
    res = {"job_id": job_id, "message": "Delivery has been submitted."}

    # if flag is set True, poll the api, return the delivery status
    res["delivery_status"] = delivery_status(job_id, poll=wait_for_delivery)
    if str(res["delivery_status"]).lower() == "failed":
        raise FailedDeliveryError((FAILED_DELIVERY_ERROR).format(res["job_id"])) from None
    return res
