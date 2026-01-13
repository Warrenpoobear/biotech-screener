from typing import Optional

from . import _error_messages


class UnauthorizedDataLakeAccessError(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        if not message:
            super().__init__(_error_messages.UNAUTHORIZED_DATALAKE_REQUEST_ERROR)
        else:
            super().__init__(message)


class InvalidQueryException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnavailableExternally(Exception):
    def __init__(self) -> None:
        super().__init__(_error_messages.UNAVAILABLE_EXTERNALLY)


class DataNotFoundException(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        super().__init__(_error_messages.DATA_NOT_FOUND)


class TempTableNameNotFoundException(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        super().__init__(_error_messages.TEMP_TABLE_NAME_NOT_FOUND)
