import re
from datetime import datetime
from typing import Callable, Union

from .._exceptions import ValueErrorException

TypeValue = Union[str, int, float, None]


class ValidCustomDataPointTypes:
    def __init__(self, data_point_definition: dict):
        function_name = re.sub(r"(?<!^)(?=[A-Z])", "_", data_point_definition["columnType"]).lower()
        self.is_valid: Callable = getattr(ValidCustomDataPointTypes, function_name, self.raise_value_error)

    def raise_value_error(self, val: TypeValue) -> None:
        raise ValueErrorException("Column type is not supported.")

    @staticmethod
    def extended_text(val: TypeValue) -> TypeValue:
        if isinstance(val, str) and len(val) <= 250:
            return val
        raise ValueErrorException("Only allows a string data type with 250 characters or less")

    @staticmethod
    def numeric(val: TypeValue) -> TypeValue:
        if isinstance(val, int) or isinstance(val, float):
            return val
        raise ValueErrorException("Only allows numbers")

    @staticmethod
    def free_text(val: TypeValue) -> TypeValue:
        if isinstance(val, str) and len(val) <= 50:
            return val
        raise ValueErrorException("Only allows a string data type with 50 characters or less")

    @staticmethod
    def date(val: TypeValue) -> TypeValue:
        format_Data = "%Y-%m-%d"
        if isinstance(val, str):
            datetime.strptime(val, format_Data)
            return val
        raise ValueErrorException("Only allows Date data type with the format yyyy-mm-dd")

    @staticmethod
    def category(val: TypeValue) -> TypeValue:
        return ValidCustomDataPointTypes.free_text(val)

    @staticmethod
    def indicator(val: TypeValue) -> TypeValue:
        return ValidCustomDataPointTypes.numeric(val)

    @staticmethod
    def t_s_return(val: TypeValue) -> TypeValue:
        return ValidCustomDataPointTypes.numeric(val)

    @staticmethod
    def t_s_price(val: TypeValue) -> TypeValue:
        return ValidCustomDataPointTypes.numeric(val)

    @staticmethod
    def t_s_numeric(val: TypeValue) -> TypeValue:
        return ValidCustomDataPointTypes.numeric(val)

    @staticmethod
    def t_s_free_text(val: TypeValue) -> TypeValue:
        return ValidCustomDataPointTypes.free_text(val)
