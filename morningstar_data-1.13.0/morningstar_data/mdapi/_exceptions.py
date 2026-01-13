from dataclasses import dataclass
from typing import Optional


class MdBaseException(Exception):
    pass


@dataclass
class MdApiRequestException(MdBaseException):
    request_id: str
    url: str
    detail: str

    def __str__(self) -> str:
        output = f"MdApiRequestException\nURL: {self.url}\nRequest ID: {self.request_id}"
        if self.detail:
            output += f"\nDetail: {self.detail}"
        return output


@dataclass
class MdApiTaskException(MdBaseException):
    task_id: str
    detail: Optional[str]

    def __str__(self) -> str:
        output = f"\nTask ID: {self.task_id}"
        if self.detail:
            output += f"\nDetail: {self.detail}"
        return output


@dataclass
class MdApiTaskTimeoutException(MdBaseException):
    task_id: str
    timeout_time: int

    def __str__(self) -> str:
        return f"Task timed out with timeout of {self.timeout_time} seconds.\nTask ID: {self.task_id}"


@dataclass
class MdApiTaskFailure(MdBaseException):
    message: str
    request_id: Optional[str] = None
    task_id: Optional[str] = None

    def __str__(self) -> str:
        if self.task_id:
            output = f"\nTask ID: {self.task_id}"
        else:
            output = f"\nRequest ID: {self.request_id}"
        if self.message:
            output += f"\nDetail: {self.message}"
        return output


class BadRequestError(MdApiTaskFailure):
    pass


class ResourceNotFoundError(MdApiTaskFailure):
    pass


class AccessDeniedError(MdApiTaskFailure):
    pass


# We may get exceptions passed to us from the API in string (json) format. At this point we will have
# the exception class name, the error message, and either a task id for request id.
# This function is used to create an exception object based on these values.
def exception_by_name(
    name: str, message: str = "", task_id: Optional[str] = None, request_id: Optional[str] = None
) -> MdApiTaskFailure:
    # Let's see if the exception is a subclass of MdApiTaskFailure.
    # If it is, we'll use that class to create the exception object. Otherwise default to the base MdApiTaskFailure class.
    throwable_exceptions = MdApiTaskFailure.__subclasses__()
    exc_class = next((exc for exc in throwable_exceptions if exc.__name__ == name), MdApiTaskFailure)

    return exc_class(task_id=task_id, request_id=request_id, message=message)
