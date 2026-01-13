from string import Template
from typing import Optional

from . import _error_messages


class AccessDeniedError(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        if not message:
            super().__init__(_error_messages.ACCESS_DENIED_ERROR)
        else:
            super().__init__(message)


class ApiRequestException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ApiResponseException(Exception):
    def __init__(self, message: str, status: int = 404) -> None:
        self.status_code = status
        super().__init__(message)


class BadRequestException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class CredentialsException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ConnectionException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ClientError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ForbiddenError(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        if not message:
            super().__init__(_error_messages.FORBIDDEN_ERROR)
        else:
            super().__init__(message)


class InvalidQueryException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InternalServerError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class NetworkExceptionError(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        if not message:
            super().__init__(_error_messages.NETWORK_ERROR)
        else:
            super().__init__(message)


class QueryLimitException(Exception):
    def __init__(self, query_limit: Optional[str] = None, status: int = 403) -> None:
        self.status_code = status
        if query_limit is not None:
            tpl = Template(_error_messages.QUERY_LIMIT_ERROR_SHOW_LIMIT)
            super().__init__(tpl.substitute(query=query_limit))
        else:
            super().__init__(_error_messages.QUERY_LIMIT_ERROR)


class ResourceNotFoundError(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        if not message:
            super().__init__(_error_messages.RESOURCE_NOT_FOUND_ERROR)
        else:
            super().__init__(message)


class TimeoutError(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        if not message:
            super().__init__(_error_messages.TIMEOUT_ERROR)
        else:
            super().__init__(message)


class ValueErrorException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class EAMSTokenDoesNotExistError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MalformedJWTError(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        if not message:
            super().__init__(_error_messages.MALFORMED_JWT_ERROR)
        else:
            super().__init__(message)


class FailedDeliveryError(Exception):
    def __init__(self, message: Optional[str] = None) -> None:
        if not message:
            super().__init__(_error_messages.FAILED_DELIVERY_ERROR)
        else:
            super().__init__(message)
