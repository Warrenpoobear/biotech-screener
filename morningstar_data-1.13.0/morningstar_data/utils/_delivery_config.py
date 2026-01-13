import dataclasses
import os
import re
from enum import Enum
from pathlib import Path

from ..direct._exceptions import BadRequestException, ValueErrorException


class DeliveryConfigErrorMessage(Enum):
    EMAIL = """
        Invalid Email Address.
        Example:
        - email_address = "analyticslab.user@morningstar.com"
    """
    FTP = """
        Invalid FTP configuration.
        The username and password must not be empty
        The server must not include any protocal, like "ftp://".
        The folder must not include a root path '/'.
        Example:
        - user_name = "analyticslab_user"
        - password = "password123"
        - server = "ftp.morningstar.com"
        - folder = "folder_1/folder_2/"
    """
    DELIVERY_PROFILE = """
        Invalid delivery profile Id.
        The delivery profile id must not be empty
        The delivery profile id must be a digit string
        Example:
        - delivery_profile_id = "12345"
    """


class DeliveryType(Enum):
    EMAIL = 1
    FTP = 2
    DELIVERY_PROFILE = 3


@dataclasses.dataclass
class DeliveryConfig:
    """Container for delivery configuration settings
    Args:
        method (:obj:`DeliveryType`): Required, the type of delivery, see DeliveryType for supported options
        email_address (:obj:`str`): a valid email_address to send the file to, needed if using "email" delivery option
        user_name (:obj:`str`): the user_name to login to the ftp server, needed if using "ftp" method
        password (:obj:`str`): the password to login to the ftp server, needed if using "ftp" method
        server (:obj:`str`): the ftp server to use, needed if using "ftp" method. The server address should not include "ftp://"
        folder (:obj:`str`): the folder on the ftp server to upload the file to, needed if using "ftp" method. The folder should not include a folder path like "/folder"
        file_path (:obj:`str`, `required`): the path to the file that will be delivered, including file name and extension.
        delivered_file_name (:obj:`str`, `optional`): the parameter for the user to specify the file name to be used when
            delivered. No file extension should be provided. If it's omitted, the file_path value will use as default filename. Valid characters are `., -, _, (, ), , a-z, A-Z, 0-9`. If the file
            name includes non-valid characters, an exception will be raised. If the property is omitted and the file_path is "data/folder/test_export.csv"
            the delivered_file_name will be "test_export.csv" or "test_export_2024022819184484.csv" based on the value of include_timestamp property.
        include_timestamp (:obj: `bool`, `optional`): optional parameter for the user append a timestamp to the file name. Default value will be False. The format will be YYYYMMDDHHMMSS.

    :Examples:

    Valid delivery config

    ::
        email_address = "test@morningstar.com"
        delivery_config_email = md.utils.DeliveryConfig(method=md.utils.DeliveryType.EMAIL, email_address=email_address, file_path="test_export.csv", delivered_file_name="Delivered_File_123")

        ftp_username = "test"
        ftp_password = "test_pwd"
        ftp_server = "ts.ftp.com"
        ftp_folder = "data/"
        delivery_config_ftp = md.utils.DeliveryConfig(method=md.utils.DeliveryType.FTP, user_name=ftp_username, password=ftp_password, server=ftp_server, folder=ftp_folder, file_path="test_export.csv", delivered_file_name="Delivered_File_123", include_timestamp=True)
        # As include_timestamp is set to True, the final file name will be Delivered_File_123_2024022819184484.csv

        delivery_profile_id = "12345"
        delivery_config_email = md.utils.DeliveryConfig(method=md.utils.DeliveryType.DELIVERY_PROFILE, delivery_profile_id=delivery_profile_id, file_path="test_export.csv", delivered_file_name="Delivered_File_123", include_timestamp=True)
        # As include_timestamp is set to True, the final file name will be Delivered_File_123_2024022819184484.csv

    Valid delivery config if doing multiple files or destination deliveries (Specify file_path and delivered_file_name)

    ::
        delivery_config_email = md.utils.DeliveryConfig(file_path="test_export.csv", method=md.utils.DeliveryType.EMAIL, email_address=email_address, delivered_file_name="Delivered_Dataframe_123", include_timestamp=False)

    Deliver without the delivered_file_name property

    ::

        import morningstar_data as md
        import pandas as pd

        df = pd.DataFrame({'a':[1], 'b':[2]})
        df.to_csv("test_export.csv")

        delivery_config = md.utils.DeliveryConfig(method=md.utils.DeliveryType.EMAIL, email_address="test@email.com", file_path="test_file.csv", include_timestamp=True)
        md.utils.deliver(config=delivery_config)
        # In this example we omitted the delivered_file_name property, so the file delivered will use the file_path and be named "test_file_2024022819184484.csv"

        delivery_config = md.utils.DeliveryConfig(method=md.utils.DeliveryType.EMAIL, email_address="test@email.com", file_path="test_file.csv", include_timestamp=False)
        md.utils.deliver(config=delivery_config)
        # In this example we omitted the delivered_file_name property and the include_timestamp property is false,
        # so the file delivered will use the file_path and be named "test_file.csv"

    Invalid delivery config

    ::
        email_address = "@invalid_email_address@morningstar.com"
        delivery_config_email = md.utils.DeliveryConfig(method=md.utils.DeliveryType.EMAIL, email_address=email_address, delivered_file_name="Delivered_Dataframe_123")
        >>> ValueErrorException: Invalid Email Address

        ftp_username = "test"
        ftp_password = "test_pwd"
        ftp_server = "ftp://ts.ftp.com"
        ftp_folder = "/data/"
        delivery_config_ftp = md.utils.DeliveryConfig(method=md.utils.DeliveryType.FTP, user_name=ftp_username, password=ftp_password, server=ftp_server, folder=ftp_folder, delivered_file_name="Delivered_Dataframe_123")
        >>> ValueErrorException: Invalid FTP configuration

        delivery_profile_id = "12ABC"
        delivery_config_email = md.utils.DeliveryConfig(method=md.utils.DeliveryType.DELIVERY_PROFILE, delivery_profile_id=delivery_profile_id, delivered_file_name="Delivered_Dataframe_123", include_timestamp=True)
        >>> ValueErrorException: Invalid delivery profile Id

        delivery_profile_id = "12ABC"
        delivery_config_email = md.utils.DeliveryConfig(method=md.utils.DeliveryType.DELIVERY_PROFILE, delivery_profile_id=delivery_profile_id)
        >>> BadRequestException: Delivered File Name can not be empty. Please provide a valid file name.
    """

    method: DeliveryType
    delivered_file_name: str = ""
    email_address: str = ""
    delivery_profile_id: str = ""
    user_name: str = ""
    password: str = ""
    server: str = ""
    folder: str = ""
    file_path: str = ""
    include_timestamp: bool = False

    def __post_init__(self) -> None:
        if not self.is_delivery_type_valid():
            raise ValueErrorException(DeliveryConfigErrorMessage[self.method.name].value)

        if not self._is_file_path_valid():
            error_message = (
                f"The specified file '{self.file_path}' does not exist or the path is invalid. Please provide a valid file path."
            )
            raise BadRequestException(error_message)

        if not self.delivered_file_name:
            self.delivered_file_name = Path(self.file_path).stem

    def _is_email_valid(self) -> bool:
        """Check whether the email config provided is valid"""
        email_regex = re.compile(
            r"""\b             # a word boundary
                [a-z0-9._%+-]+ # the email prefix part, any character in the list
                @              # the @ part
                [a-z0-9.-]+    # the email domain name part, any character in the list
                \.             # the dot part
                [a-z]{2,}      # the top-level domain part, any alphabetical character and length >= 2
                \b             # a word boundary
            """,
            re.X,
        )
        return re.fullmatch(email_regex, self.email_address.lower()) is not None

    def _is_delivery_profile_valid(self) -> bool:
        """Checks if the provided delivery profile ID is valid.
        At present, the MDS delivery endpoint allows any user to deliver to any profile ID.
        In the future, MDS need to add validation based on the user's company.
        """
        return self.delivery_profile_id.isdigit()

    def _is_ftp_valid(self) -> bool:
        """Check whether the ftp config provided is valid"""
        if self.user_name == "" or self.password == "":
            return False

        server_regex = re.compile(
            r"""\b          # a word boundary
                [a-z0-9.-]+ # the domain name part, any character in the list
                \.          # the dot part
                [a-z]{2,}   # the top-level domain part, any alphabetical character and length >= 2
                \b          # a word boundary
            """,
            re.X,
        )
        if not re.fullmatch(server_regex, self.server.lower()):
            return False

        folder_regex = re.compile("^(\.$|[^/].*\/)$")
        # (\.$|: Matches a single dot (.).
        # [^/]: Ensures the string does not start with a /.
        # .*: Allows any sequence of characters (including spaces, +, -, _, %).
        # \/: Ensures the string ends with a /.

        if not re.fullmatch(folder_regex, self.folder):
            return False

        return True

    def _is_file_path_valid(self) -> bool:
        """Check whether the file path provided is valid"""
        return os.path.isfile(self.file_path)

    def is_delivery_type_valid(self) -> bool:
        is_email_valid = self.method == DeliveryType.EMAIL and self._is_email_valid()
        is_ftp_valid = self.method == DeliveryType.FTP and self._is_ftp_valid()
        is_delivery_profile_valid = self.method == DeliveryType.DELIVERY_PROFILE and self._is_delivery_profile_valid()
        return is_email_valid or is_ftp_valid or is_delivery_profile_valid
