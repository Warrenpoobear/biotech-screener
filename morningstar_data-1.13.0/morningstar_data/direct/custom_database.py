import warnings
from abc import ABC
from typing import Any, Dict, List, Optional, Union

import simplejson as json
from pandas import DataFrame

from .._base import _logger
from . import _decorator, _error_messages
from ._base_api import APIBackend
from ._config import _Config
from ._data_objects import TypeValue, ValidCustomDataPointTypes
from ._data_type import DatabaseCD
from ._exceptions import ResourceNotFoundError, ValueErrorException
from ._utils import _mapper_data_frame_return_dict, _mapper_data_frame_return_list
from .data_type import Blank

_config = _Config()


class CustomDatabaseAPIBackend(APIBackend):
    """
    Subclass to call the custom database API and handle any HTTP errors that occur.
    """

    def __init__(self) -> None:
        super().__init__()

    def _handle_custom_http_errors(self, res: Any) -> Any:
        """
        Handle HTTP errors with custom error messages
        """
        response_message = res.json().get("message")
        if res.status_code == 404:
            _logger.debug(f"Resource Not Found Error: {res.status_code} {response_message}")
            raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_ERROR_CUSTOM_DATABASE) from None


ResultType = Dict[str, Union[bool, str]]
ListDatapointType = List[Dict[str, Any]]
DictValues = Dict[str, TypeValue]
RelValuesWithDatapoint = Dict[str, Union[TypeValue, List[DictValues]]]


class _CustomDatabase(ABC):
    def __init__(self, database_type: DatabaseCD, custom_database_api_request: CustomDatabaseAPIBackend):
        self.database_type: DatabaseCD = database_type
        self.all_data_points: ListDatapointType = []
        self._custom_database_api_request = custom_database_api_request

    def get_data_point_from_name(self, data_point_name: str, time_series: bool = False) -> Dict[str, str]:
        data_points = self.get_all_data_points()
        data_point = next(
            (item for item in data_points if item["name"] == data_point_name),
            None,
        )

        if data_point is None or "columnType" not in data_point:
            raise ValueErrorException(f"The column '{data_point_name}' does not exist in the custom datapoints.")

        start_with_ts = data_point["columnType"].startswith("TS")

        if time_series and not start_with_ts:
            raise ValueErrorException(f"The column '{data_point_name}' is not for a time series.")

        if not time_series and start_with_ts:
            raise ValueErrorException(f"The column '{data_point_name}' is not for a single value.")

        return data_point

    def get_all_data_points(self) -> ListDatapointType:
        if not self.all_data_points:
            url = f"{_config.custom_data_points_service_url()}v1/cdp/definitions?type={self.database_type.value}"
            self.all_data_points = self._custom_database_api_request.do_get_request(url)
        return self.all_data_points

    def is_a_valid_value(self, value: Optional[TypeValue], data_point_definition: dict, investment: str) -> Optional[TypeValue]:
        if value is None:
            return value
        validator = ValidCustomDataPointTypes(data_point_definition)
        try:
            result: TypeValue = validator.is_valid(value)
            return result
        except Exception as e:
            raise ValueErrorException(f"Error in '{data_point_definition['name']}' column with '{investment}': {str(e)}")

    def convert_to_objects(self, values: DataFrame) -> Any:
        return values.astype(object).where(values.notnull(), None).where(values != "", None)

    def filter_null_values(
        self,
        values: DataFrame,
        blank: Blank,
        key_column: str,
        date_column: Optional[str] = None,
    ) -> Any:
        ops = Blank.options()
        if blank is Blank.warning and values.isnull().values.any():
            raise ValueErrorException(
                f"The dataframe contains null or blank values. \nPlease choose one of the following as a `blank` arg: {ops} \n"
            ) from None
        ignore_columns = [
            key_column,
            date_column if date_column is not None else key_column,
        ]
        if blank is Blank.ignore and values.drop(list(set(ignore_columns)), axis=1).isnull().values.all():
            raise ValueErrorException(
                f"All the dataframe values are null or blank. \nPlease choose one of the following as a `blank` arg: {ops} \n"
            ) from None
        return values

    def get_to_delete(self, value: TypeValue, blank: Blank) -> bool:
        if blank is Blank.ignore:
            return False
        else:
            if value is None:
                return True
        return False

    @_decorator.typechecked
    def save_historical_values(
        self,
        values: DataFrame,
        key_column: str = "SecId",
        date_column: str = "Date",
        blank: Union[Blank, str] = Blank.warning,
    ) -> ResultType:
        """Saves historical time series values for data points in the user's or firm's database in Morningstar Direct.

        Args:
            values (:obj:`DataFrame`): A DataFrame object with the time series values to be saved in the user’s or firm’s database. The DataFrame columns should include:

                * `SecId` - Column containing investment ID values. If a column named `SecId` is not present, the column name specified in the key_column argument will be used.
                * `DataPointName` - Data point name which will be updated. Multiple data points can be updated in one request.
                * `Date` - Column containing date values. If a column named `Date` is not present, the column name specified in the date_column argument will be used.

                For example::

                    ====================  ========  ===============  ===============  ===============
                    SecId                   Date    DataPointName1   DataPointName2   DataPointName3
                    ====================  ========  ===============  ===============  ===============
                    <InvestmentId>         <date>       <value>          <value>          <value>
                    <InvestmentId>         <date>       <value>          <value>          <value>
                    ====================  ========  ===============  ===============  ===============

            key_column (:obj:`str`): Name of the column containing investment ID values. If not specified, the column `SecId` will be used.

            date_column (:obj:`str`): Name of the column containing date values. If not specified, the column `Date` will be used.

            blank (:obj:`md.direct.data_type.Blank`): Argument specifying how blank values in the DataFrame should be handled. Valid enum values for this argument are:

                    * warning - Raises an error if the DataFrame contains at least one null or empty (str) value.
                    * ignore - Saves all DataFrame values except for null or empty (str) values.
                    * update - Replaces/updates all existing values with the provided values. Deletes any value that is sent as null, None or empty (str) values.

        :Returns:
            DataFrame:

        :Examples:

        ::

            import morningstar_data as md
            from pandas import DataFrame

            df = DataFrame({"SecId": ["FOUSA00C3M", "F000010NJ5"],
                            "Date": ["2022-02-10","2022-04-2"],
                            "DataPointNameText": ["2","9"],
                            "DataPointNameNumber": [12.2178, 78514]})

            md.direct.my_database.save_historical_values(values=df, key_column="SecId", blank=md.direct.data_type.Blank.warning)



        ::

            import morningstar_data as md
            from pandas import DataFrame

            df = DataFrame({"SecId": ["FOUSA00C3M", "F000010NJ5"],
                            "Date": ["2022-02-10","2022-04-2"],
                            "DataPointNameText": ["2","9"],
                            "DataPointNameNumber": [12.2178, 78514]})

            md.direct.firm_database.save_historical_values(values=df, key_column="SecId", blank=md.direct.data_type.Blank.warning)

        :Errors:
            AccessDeniedError: Raised when the user is not authenticated.

            ForbiddenError: Raised when the user does not have permission to access the requested resource.

            InternalServerError: Raised when the server encounters an unhandled error.

            NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

            ResourceNotFoundError: Raised when the requested resource does not exist in Direct.

        """
        if not isinstance(blank, Blank):
            warnings.warn(
                "The use of string values for the 'blank' parameter is deprecated and will be removed in the next major version. Use Blank enum values instead",
                FutureWarning,
                stacklevel=2,
            )

        blank = Blank[blank]
        assert isinstance(blank, Blank)

        if key_column not in values:
            raise ValueErrorException(f"The key column '{key_column}' does not exist.")

        if date_column not in values:
            raise ValueErrorException(f"The date column '{date_column}' does not exist")

        values = self.filter_null_values(self.convert_to_objects(values), blank, key_column, date_column)

        def get_values(last_dict: Dict, column: str, value: Any, row: Dict) -> Dict:
            if column in [key_column, date_column]:
                return last_dict

            date = self.is_a_valid_value(
                row[date_column],
                {
                    "name": date_column,
                    "columnType": "Date",
                },
                row[key_column],
            )
            data_point: Dict[str, str] = self.get_data_point_from_name(column, True)

            assert isinstance(blank, Blank)
            if value is None and blank is Blank.ignore:
                return last_dict
            ts_value = self.is_a_valid_value(value, data_point, row[key_column])
            relation = f"{data_point['id']}-{row[key_column]}"

            if relation not in last_dict:
                last_dict[relation] = []

            to_delete = self.get_to_delete(ts_value, blank)
            last_dict[relation].append({"date": date, "value": ts_value, "toDelete": to_delete})
            return last_dict

        relations: DictValues = _mapper_data_frame_return_dict(values, get_values)
        data: List[RelValuesWithDatapoint] = [
            {
                "dataPointId": relation.split("-")[0],
                "investmentId": relation.split("-")[1],
                "timeSeriesValue": points,
            }
            for relation, points in relations.items()
        ]
        return self._request_save_values(data)

    @_decorator.typechecked
    def save_values(
        self,
        values: DataFrame,
        key_column: str = "SecId",
        blank: Union[Blank, str] = Blank.warning,
    ) -> ResultType:
        """Saves values for data points in the user's or firm's database.

        Args:
            values (:obj:`DataFrame`): A DataFrame object with the values to be saved in the user's or firm's database. The DataFrame columns should include:

                * `SecId` - Column containing investment ID values. If a column named `SecId` is not present, the column name specified in the key_column argument will be used.
                * `DataPointName` - Data point name which will be updated. Multiple data points can be updated in one request.

                For example::

                    ====================  ===============  ===============  ===============
                    SecId                 DataPointName1   DataPointName2   DataPointName3
                    ====================  ===============  ===============  ===============
                    <InvestmentId>            <value>          <value>          <value>
                    <InvestmentId>            <value>          <value>          <value>
                    ====================  ===============  ===============  ===============

            key_column (:obj:`str`): Name of the column containing investment ID values. If not specified, the column `SecId` will be used.

            blank (:obj:`md.direct.data_type.Blank`): Argument specifying how blank values in the DataFrame should be handled. Valid enum values for this argument are:

                * warning - Raises an error if the DataFrame contains at least one null or empty (str) value.
                * ignore - Saves all DataFrame values except for null or empty (str) values.
                * update - Replaces/updates all existing values with the provided values. Deletes any value that is sent as null, None or empty (str) values.
        :Returns:
            DataFrame:

        :Examples:

        ::

            import morningstar_data as md
            from pandas import DataFrame
            df = DataFrame({"SecId": ["FOUSA00C3M", "F000010NJ5"],
                               "DataPointNameText": ["2","9"],
                               "DataPointNameDate": ["2022-02-10","2022-04-2"],
                               "DataPointNameNumber": [12.2178, 78514]})
            md.direct.my_database.save_values(values=df, key_column="SecId", blank=md.direct.data_type.Blank.warning)

        ::

            import morningstar_data as md
            from pandas import DataFrame
            df = DataFrame({"SecId": ["FOUSA00C3M", "F000010NJ5"],
                               "DataPointNameText": ["2","9"],
                               "DataPointNameDate": ["2021-05-31","2021-02-01"],
                               "DataPointNameNumber": [12.2178, 78514]})
            md.direct.firm_database.save_values(values=df, key_column="SecId", blank=md.direct.data_type.Blank.warning)

        :Errors:
            AccessDeniedError: Raised when the user is not authenticated.

            ForbiddenError: Raised when the user does not have permission to access the requested resource.

            InternalServerError: Raised when the server encounters an unhandled error.

            NetworkExceptionError: Raised when the request fails to reach the server due to a network error.

            ResourceNotFoundError: Raised when the requested resource does not exist in Direct.
        """
        if not isinstance(blank, Blank):
            warnings.warn(
                "The use of string values for the 'blank' parameter is deprecated and will be removed in the next major version. Use Blank enum values instead",
                FutureWarning,
                stacklevel=2,
            )

        blank = Blank[blank]
        assert isinstance(blank, Blank)

        if key_column not in values:
            raise ValueErrorException(f"The key column '{key_column}' does not exist.")

        values = self.filter_null_values(self.convert_to_objects(values), blank, key_column)

        def get_values(column: str, value: Any, row: dict) -> Optional[RelValuesWithDatapoint]:
            if column is key_column:
                return None

            assert isinstance(blank, Blank)
            if value is None and blank is Blank.ignore:
                return None
            data_point = self.get_data_point_from_name(column)
            single_value = self.is_a_valid_value(value, data_point, row[key_column])
            return {
                "dataPointId": data_point["id"],
                "investmentId": row[key_column],
                "singleValue": single_value,
                "toDelete": self.get_to_delete(value, blank),
            }

        data: List[RelValuesWithDatapoint] = _mapper_data_frame_return_list(values, get_values)

        return self._request_save_values(data)

    def _request_save_values(self, data: List[RelValuesWithDatapoint]) -> ResultType:
        url = f"{_config.custom_data_points_service_url()}v1/cdp/investments/values?type={self.database_type.value}"
        result: ResultType = self._custom_database_api_request.do_put_request(url, json.dumps(data, ignore_nan=True))
        return result


my_database = _CustomDatabase(DatabaseCD.user, CustomDatabaseAPIBackend())
firm_database = _CustomDatabase(DatabaseCD.firm, CustomDatabaseAPIBackend())
