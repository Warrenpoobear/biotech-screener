import sys
import traceback
from functools import wraps
from typing import Any, Callable

import typeguard

from ._exceptions import (
    ApiResponseException,
    BadRequestException,
    QueryLimitException,
    ValueErrorException,
)

typechecked: Callable = typeguard.typechecked


def not_null(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for arg in args:
            if not arg:
                raise BadRequestException("Null or empty parameter is not allowed.")

        return func(*args, **kwargs)

    return wrapper


def error_handler(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (BadRequestException, ValueErrorException) as e:
            wrappers = [x.name for x in traceback.extract_stack() if x.name == "error_handler"]
            if len(wrappers) > 1:
                raise ValueErrorException(str(e)) from None
            else:
                sys.stderr.write(str(e))
        except ApiResponseException as e:
            if e.status_code == 404:
                sys.stderr.write("Specified data not Found. ")
            else:
                raise ValueErrorException(str(e)) from None
        except QueryLimitException:
            # Some exceptions need to be propagated back to the caller as-is
            raise
        except Exception as e:
            raise ValueErrorException(str(e)) from None

    return wrapper
