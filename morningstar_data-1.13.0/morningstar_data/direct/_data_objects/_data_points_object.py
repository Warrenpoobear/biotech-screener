import copy
import functools
import inspect
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pandas import Series

from ...direct.user_items.data_set import (
    _get_binding_data_set,
    _get_defaultview_data_points,
    _get_user_data_set,
)
from .._config_key import ALL_ASSET_FLOW_DATA_POINTS
from .._data_point import (
    _data_point_request_builder,
    _get_asset_flow_data_points_by_ids,
    _get_data_point_details,
)
from .._error_messages import (
    BAD_REQUEST_ERROR_ALIAS_DUPLICATED,
    BAD_REQUEST_ERROR_INCLUDE_DIFF_CALCULATION_DATA_POINTS,
    BAD_REQUEST_ERROR_INCLUDE_NON_CALCULATION_DATA_POINTS,
    BAD_REQUEST_ERROR_NO_DATA_POINT_ID_OR_ALIAS,
    BAD_REQUEST_ERROR_NO_DATA_POINTS,
)
from .._exceptions import BadRequestException
from .._utils import _is_uuid

DatapointIdType = Optional[str]
DatapointIdsType = Optional[List[str]]
DataPointsSettingsType = Optional[pd.DataFrame]
DataPointsType = Union[DatapointIdType, DatapointIdsType, DataPointsSettingsType]

DefaultDataPoints = [{"datapointId": "HP010"}]


class DataPoints:
    def __init__(self, data_points_object: DataPointsType) -> None:
        self.data_set_id: DatapointIdType = None
        self.data_point_ids: DatapointIdsType = None
        self.data_point_settings: DataPointsSettingsType = None
        self._parse_data_points_object(data_points_object)

    def _parse_data_points_object(self, data_points_object: DataPointsType) -> None:
        if isinstance(data_points_object, str):
            self.data_set_id = data_points_object
        elif isinstance(data_points_object, list):
            if self._is_list_of_dicts(data_points_object):
                self.data_point_ids = data_points_object
            else:
                try:
                    self.data_point_settings = self._convert_morningstar_data_points_object(data_points_object)
                except Exception:
                    ValueError("Invalid data points object")
        elif isinstance(data_points_object, pd.DataFrame):
            self.data_point_settings = data_points_object
        elif data_points_object is not None:
            raise ValueError("Invalid data points object")

    def validate_data_point_ids(self) -> None:
        if self.data_point_ids is not None:
            settings_data_frame = pd.DataFrame(self.data_point_ids)
            if settings_data_frame.empty:
                raise BadRequestException("data_point_ids is empty.")
            elif "datapointId" not in settings_data_frame.columns:
                raise BadRequestException("datapointId is required in data_point_ids.")
        elif self.data_point_settings is not None:
            if self.data_point_settings.empty:
                raise BadRequestException("data_point_settings is empty.")
            elif "datapointId" not in self.data_point_settings.columns:
                raise BadRequestException("datapointId is required in data_point_settings.")

    @functools.lru_cache()
    def get_data_points(
        self,
        list_id: Optional[str] = None,
        search_criteria_id: Optional[str] = None,
        display_name: bool = False,
    ) -> pd.DataFrame:
        data_point_list = []
        if self.data_point_ids:
            settings_data_frame = pd.DataFrame(self.data_point_ids)
            data_point_list = self._convert_data_point_data_frame(settings_data_frame)

            data_point_id_set = set(settings_data_frame["datapointId"])
            valid_data_point_id_set = set(map(lambda x: str(x["datapointId"]), data_point_list))
            invalid_data_point_id_set = data_point_id_set - valid_data_point_id_set

            if data_point_id_set == invalid_data_point_id_set:  # All invalid data points
                if len(invalid_data_point_id_set) == 1:
                    invalid_data_point_id = list(invalid_data_point_id_set)[0]
                    raise BadRequestException(
                        f"The specified data point ID does not exist: {invalid_data_point_id}. "
                        f"If you were trying to specify a data set ID, use: data_points='{invalid_data_point_id}'"
                    )
                else:
                    raise BadRequestException("All of the specified data point IDs do not exist.")
            elif len(invalid_data_point_id_set) >= 1:
                invalid_data_points = ", ".join(list(invalid_data_point_id_set))
                raise BadRequestException(
                    f"The specified data point IDs do not exist: {invalid_data_points} or the datapoint settings specified are not valid."
                )

        elif self.data_point_settings is not None and not self.data_point_settings.empty:
            data_point_list = self._convert_data_point_data_frame(self.data_point_settings)

        elif self.data_set_id is not None and len(self.data_set_id.strip()) > 0:
            data_set_content = self._get_data_set_data_points(None)
            self._insert_name_data_point(data_set_content=data_set_content)
            data_points = _data_point_request_builder(data_set_content)
            # displayName field reflects the column names that are shown in direct for user created data sets.
            # It will have the custom datapoint name if the column name was changed by the user otherwise
            # it will have the default datapoint name. We use desktopDisplayName for datapoints where Direct
            # appends a specific string at the end of the datapoints. For eg. For currency data points (OS060),
            # Base Currency is appended at the end of the datapoint name. In the below logic, the order or preference
            # is desktopDisplayName -> displayName -> defaultDatapointName

            data_point_list = []
            for data_set, data_point in zip(data_set_content, data_points):
                alias_exists = "alias" in data_set and data_set["alias"] == data_point["alias"]
                desktop_display_name_exists = "desktopDisplayName" in data_set and data_set["desktopDisplayName"] is not None
                datapoint_name_exists = "datapointName" in data_set
                if alias_exists and desktop_display_name_exists:
                    display_value = data_set["general"]["desktopDisplayName"]
                elif display_name is True:
                    display_value = data_set["general"]["displayName"]
                else:
                    if datapoint_name_exists:
                        display_value = data_set["datapointName"]  # For user-created data sets
                    else:
                        display_value = data_set["name"]  # For morningstar data sets

                data_point_entry = {**data_point, **{"displayName": display_value}}
                data_point_list.append(data_point_entry)
            if not data_point_list:
                raise BadRequestException("No available data points in the data set.")

        elif list_id is not None and len(list_id.strip()) > 0:
            data_set_content = self._get_binding_data_set_for_user_object(user_object_id=list_id, id_type="LIST")
            self._insert_name_data_point(data_set_content=data_set_content)
            data_point_list = _data_point_request_builder(data_set_content)

        elif search_criteria_id is not None and len(search_criteria_id.strip()) > 0:
            data_set_content = self._get_binding_data_set_for_user_object(user_object_id=search_criteria_id, id_type="SCREEN")
            self._insert_name_data_point(data_set_content=data_set_content)
            data_point_list = _data_point_request_builder(data_set_content)

        if not data_point_list:
            data_point_list = self._get_default_data_point_list()

        result = pd.DataFrame(data_point_list)
        if "windowType" in result.columns:
            result["windowType"] = result["windowType"].astype("Int64")
            result["windowType"] = result["windowType"].astype("object").where(result["windowType"].notna(), None)
        result = result.where(result.notnull(), None)
        return result

    def _insert_name_data_point(self, data_set_content: list) -> None:
        if data_set_content and all(x.get("datapointId") != "OS01W" for x in data_set_content):
            data_set_content.insert(
                0,
                _get_data_point_details([{"datapointId": "OS01W", "isTsdp": False}])[0],
            )

    def _get_binding_data_set_for_user_object(self, user_object_id: str, id_type: str) -> list:
        """
        Get binding data set data points for a investment list or search criteria.
        :param user_object_id: The ID of a saved investment list or search criteria from Direct.
        :param id_type: "LIST" or "SCREEN"
        """
        if user_object_id is not None and len(user_object_id.strip()) > 0 and id_type is not None and len(id_type.strip()) > 0:
            dataset_resp = _get_binding_data_set(user_object_id=user_object_id, id_type=id_type)
            dataset_content = dataset_resp.get("content", []) if dataset_resp else list()
            return self._get_visible_data_points(dataset_content=dataset_content)
        return list()

    def _get_visible_data_points(self, dataset_content: list) -> list:
        available_data_points = []
        if dataset_content:
            for dp in dataset_content:
                desktop_attributions = dp.get("desktopAttributions", None)
                hide = desktop_attributions.get("hide", "0") if desktop_attributions else 0
                if hide != "1":
                    available_data_points.append(dp)
        return available_data_points

    def _validate_data_point_ids_and_alias(self) -> Optional[None]:
        data_point_df = None
        if self.data_point_ids:
            data_point_df = pd.DataFrame(self.data_point_ids)

        elif self.data_point_settings is not None and not self.data_point_settings.empty:
            data_point_df = self.data_point_settings

        if data_point_df is not None and not data_point_df.empty:
            if (
                "datapointId" not in data_point_df.columns
                or "alias" not in data_point_df.columns
                or len(data_point_df[(data_point_df["datapointId"] == "") | (data_point_df["datapointId"].isnull())]) > 0
                or len(data_point_df[(data_point_df["alias"] == "") | (data_point_df["alias"].isnull())]) > 0
            ):
                raise BadRequestException(BAD_REQUEST_ERROR_NO_DATA_POINT_ID_OR_ALIAS)
            if data_point_df[["alias"]].duplicated().sum() >= 1:
                raise BadRequestException(BAD_REQUEST_ERROR_ALIAS_DUPLICATED)

    def get_peer_group_data_points(self) -> pd.DataFrame:
        data_point_df = None
        if self.data_point_ids:
            data_point_df = pd.DataFrame(self.data_point_ids)

        elif self.data_point_settings is not None and not self.data_point_settings.empty:
            data_point_df = self.data_point_settings

        if data_point_df is not None and not data_point_df.empty:
            data_point_df = data_point_df.where(data_point_df.notnull(), None)
            if len(data_point_df[~data_point_df["datapointId"].str.isdigit()]) > 0:
                raise BadRequestException(BAD_REQUEST_ERROR_INCLUDE_NON_CALCULATION_DATA_POINTS)
            if len(data_point_df[["datapointId"]].drop_duplicates()) > 1:
                raise BadRequestException(BAD_REQUEST_ERROR_INCLUDE_DIFF_CALCULATION_DATA_POINTS)

            self._handle_special_data_point_columns(data_point_df)
            data_point_list = self._convert_settings_to_builder(data_point_df)

            if data_point_list:
                result = pd.DataFrame(data_point_list)
                result = result.where(result.notnull(), None)
                return result

        raise BadRequestException(BAD_REQUEST_ERROR_NO_DATA_POINTS)

    def _convert_settings_to_builder(self, settings_data_frame: pd.DataFrame) -> list:
        request_builder = []
        if "isTsdp" not in settings_data_frame.columns:
            settings_data_frame["isTsdp"] = None
        settings_data_frame = settings_data_frame.where(settings_data_frame.notnull(), None)
        default_settings = _get_data_point_details(settings_data_frame[["datapointId", "isTsdp"]].to_dict(orient="records"))
        if default_settings and isinstance(default_settings, list):
            settings = self._replace_value(default_settings, settings_data_frame)
            request_builder = _data_point_request_builder(settings)
            for data_point in request_builder:
                if data_point.get("isTsdp") is True and "frequency" not in data_point:
                    data_point["frequency"] = "m"
        return request_builder

    def _handle_special_data_point_columns(self, settings_data_frame: pd.DataFrame) -> None:
        """
        For values in the columns "windowType", "windowSize", "stepSize" in the settings dataframe, replace empty values with None
        """
        convert_columns = ["windowType", "windowSize", "stepSize"]
        column_list = settings_data_frame.columns.tolist()
        convert_exist_columns = [x for x in convert_columns if x in column_list]
        settings_data_frame[convert_exist_columns] = settings_data_frame[convert_exist_columns].replace(
            r"^\s*$", None, regex=True
        )
        settings_data_frame[convert_exist_columns] = (
            settings_data_frame[convert_exist_columns].astype(float).astype(pd.Int64Dtype()).fillna(-1)
        )

    def _convert_data_point_data_frame(self, settings_data_frame: pd.DataFrame) -> list:
        if "OS01W" not in settings_data_frame["datapointId"].tolist():
            name_data_point = pd.DataFrame([{"datapointId": "OS01W", "isTsdp": False}])
            settings_data_frame = pd.concat([name_data_point, settings_data_frame], axis=0)

        settings_data_frame = settings_data_frame.apply(self._set_alias, axis=1)
        settings_data_frame = settings_data_frame.where(settings_data_frame.notnull(), None)

        # separate data points
        asset_flow_data_points = (
            settings_data_frame[settings_data_frame["datapointId"].isin(ALL_ASSET_FLOW_DATA_POINTS)]
            .reset_index()
            .drop(["index"], axis=1)
        )
        normal_data_points = (
            settings_data_frame[~settings_data_frame["datapointId"].isin(ALL_ASSET_FLOW_DATA_POINTS)]
            .reset_index()
            .drop(["index"], axis=1)
        )
        alias_data_point_dict = dict()
        if not asset_flow_data_points.empty:
            default_settings = _get_asset_flow_data_points_by_ids(asset_flow_data_points["datapointId"].tolist())
            if default_settings:
                asset_flow_list = self._replace_asset_flow_settings(default_settings, asset_flow_data_points)
                alias_data_point_dict.update({x.get("alias"): x for x in asset_flow_list})

        if not normal_data_points.empty:
            self._handle_special_data_point_columns(normal_data_points)
            normal_list = self._convert_settings_to_builder(normal_data_points)
            normal_list = normal_list if normal_list else list()
            self._get_no_map_data_points_request_builder(normal_list, normal_data_points)
            alias_data_point_dict.update({x.get("alias"): x for x in normal_list})

        data_point_list = []
        for _, item in settings_data_frame.iterrows():
            target = alias_data_point_dict.get(item["alias"])
            if target:
                display_name = item.get("displayName")
                if display_name is not None:
                    target["displayName"] = str(display_name)
                data_point_list.append(target)
            target = alias_data_point_dict.get(item["alias"] + "_2")
            if target:
                display_name = item.get("displayName")
                if display_name is not None:
                    target["displayName"] = str(display_name)
                data_point_list.append(target)

        return data_point_list

    def _get_data_set_data_points(self, data_point_alias: Optional[list] = None) -> List[Any]:
        if (
            self.data_set_id is not None and re.match(r"^\d{4}-\d{4}$", self.data_set_id, re.M) is not None
        ):  # morningstar data view
            data_set_resp = _get_defaultview_data_points(self.data_set_id)
            if not data_set_resp:
                raise BadRequestException("Data set not Found.")
            data_set_content: List[Any] = data_set_resp.get("content", [])
            if data_set_content:
                for data_point in data_set_content:
                    data_point["datapointId"] = data_point["id"]
            data_set_content = [{**item, **{"displayName": item["general"]["displayName"]}} for item in data_set_content]
            return data_set_content
        else:  # user created data set
            data_set_resp = _get_user_data_set(self.data_set_id)
            if not data_set_resp:
                raise BadRequestException("Data set not Found.")
            data_set_content = data_set_resp.get("content", [])
            sub_data_set_content = (
                [x for x in data_set_content if x.get("alias", "") in data_point_alias] if data_point_alias else data_set_content
            )

            # Add the displayName, desktopDisplayName attribute for each data point in the dataset accordingly
            sub_data_set_content = [
                {
                    **item,
                    **{
                        "displayName": item["general"]["displayName"],
                        "desktopDisplayName": item["general"]["desktopDisplayName"]
                        if "calcCurType" in item["desktopAttributions"]
                        and item["desktopAttributions"]["calcCurType"] != ""
                        and item["desktopAttributions"]["calcCurType"] is not None
                        else None,
                    },
                }
                for item in sub_data_set_content
            ]
            return sub_data_set_content

    def _replace_asset_flow_settings(
        self,
        default_settings: List[Dict[str, Any]],
        asset_flow_data_points: pd.DataFrame,
    ) -> list:
        data_point_list: List[Any] = []
        if asset_flow_data_points is None or asset_flow_data_points.empty:
            return data_point_list
        data_point_id_default_settings_dict = {x.get("datapointId"): x for x in default_settings}
        for _, data_point_setting in asset_flow_data_points.iterrows():
            data_point_id = data_point_setting["datapointId"]
            default_setting = data_point_id_default_settings_dict.get(data_point_id, None)
            if not default_setting:
                continue
            default_setting_copy = copy.deepcopy(default_setting)
            for key, value in data_point_setting.items():
                if value is None:
                    continue
                if key in default_setting_copy.keys() or key == "marketId":
                    default_setting_copy[key] = value
            data_point_list.append(default_setting_copy)

        return data_point_list

    def _set_alias(self, settings: dict) -> dict:
        # Normally, aliases are not specified by users when specifying data point settings
        # for methods like get_investment_data() as they are auto-generated by the code below.
        # Other parts of the code include the alias in the request for certain DO API calls so it can
        # map data points in the API request to data point values in the API response.
        # The only time we need to specify aliases in data point settings is when we do unit tests to
        # ensure we use the same set of UUIDs on every test run.
        if "alias" not in settings or not _is_uuid(str(settings["alias"])):
            settings["alias"] = str(uuid.uuid4())
        return settings

    def _replace_value(self, default_settings: list, data_point_settings: pd.DataFrame) -> list:
        data_point_list = []
        for _, data_point_setting in data_point_settings.iterrows():
            data_point_id = data_point_setting["datapointId"]
            is_ts_dp = data_point_setting.get("isTsdp", None)
            if isinstance(is_ts_dp, bool):
                target = self._filter_data_point(data_point_id, is_ts_dp, default_settings)
                if target is not None:
                    data_point_list.append(self._copy_and_replace_value(target, data_point_setting))
            else:
                data_points = list()
                target_ts = self._filter_data_point(data_point_id, True, default_settings)
                target_current = self._filter_data_point(data_point_id, False, default_settings)
                if target_ts is not None and target_current is not None:
                    """
                    If target_ts is not None and target_current is not None, the datapoint is a time series datapoint.
                    In this scenario, if startDate and endDate are passed, only get data for target_ts.
                    """
                    if self._is_start_end_date_available(data_point_setting):
                        target_current = None

                if target_ts is not None:
                    data_points.append(self._copy_and_replace_value(target_ts, data_point_setting))

                if target_current is not None:
                    data_points.append(self._copy_and_replace_value(target_current, data_point_setting))
                if len(data_points) == 2:
                    data_points[1]["alias"] = data_points[1]["alias"] + "_2"
                if data_points:
                    data_point_list.extend(data_points)
        return data_point_list

    def _is_start_end_date_available(self, data_point_setting: Series) -> bool:
        start_date = data_point_setting.get("startDate", None)
        end_date = data_point_setting.get("endDate", None)
        if start_date is not None and len(start_date.strip()) > 0 and end_date is not None and len(end_date.strip()) > 0:
            return True
        return False

    def _filter_data_point(self, data_point_id: str, is_ts: bool, default_settings: list) -> Optional[dict]:
        target_list = list(
            filter(
                lambda x: x.get("datapointId", "") == data_point_id and x.get("isTsdp", "") == is_ts,
                default_settings,
            )
        )
        return target_list[0] if len(target_list) > 0 else None

    def _copy_and_replace_value(self, target: dict, data_point_setting: pd.Series) -> dict:
        target_copy = copy.deepcopy(target)
        for key, value in data_point_setting.items():
            if value is None:
                continue
            if key in target_copy.keys() or key == "alias":
                target_copy[key] = value
            else:
                for name_key in [
                    "general",
                    "source",
                    "calculation",
                    "desktopAttributions",
                ]:
                    if name_key not in target_copy:
                        target_copy[name_key] = {}
                    target_copy[name_key][key] = value

        return target_copy

    def _get_no_map_data_points_request_builder(self, request_builder_list: list, df: pd.DataFrame) -> None:
        no_map_data_points = [
            "DC09A",
            "TE190",
            "TE158",
            "TE157",
            "TE156",
            "TE164",
            "TE163",
            "TE162",
            "TE161",
            "TE218",
        ]
        for row in df.itertuples():
            data_point_id = getattr(row, "datapointId")
            if data_point_id in no_map_data_points:
                no_map_data = {
                    "alias": getattr(row, "alias"),
                    "annualDays": "0",
                    "datapointId": data_point_id,
                    "isTsdp": getattr(row, "isTsdp"),
                    "requireContinueData": True,
                    "sourceId": getattr(row, "datapointId"),
                }

                if no_map_data["isTsdp"] is True:
                    for name_column in ["frequency", "startDate", "endDate"]:
                        if getattr(row, name_column):
                            no_map_data[name_column] = getattr(row, name_column)

                request_builder_list.append(no_map_data)

    def _convert_morningstar_data_points_object(self, data_points: list) -> pd.DataFrame:
        """
        Converts a list of morningstar_data_points DataPoints or TimeSeriesDataPoints objects into a
        data points settings DataFrame object.
        """
        formatted_points = []
        for point in data_points:
            if not isinstance(point, dict):
                formatted_points.append(self._convert_data_point(point))
            else:
                formatted_points.append(point)
        return pd.DataFrame(formatted_points)

    def _convert_data_point(self, dp: dict) -> dict:
        """Converts a single data point.

        Creates a single data point in the format for get_investment_data.

        Args:
            dp: A data point

        Returns:
            Vals: A dictionary of the needed settings for get_investment_data function
        """

        vals = {}
        for k, v in dp.__dict__.items():
            if k == "name":
                pass
            elif k == "datapointId":
                vals[k] = v
            elif k == "sourceData":
                vals["sourceId"] = str(v)
            elif isinstance(v, datetime):
                vals[k] = (str(v))[0:10]
            else:
                vals[k] = str(v)

        vals["datapointName"] = dp.name  # type: ignore
        dp_module_name = inspect.getmodule(type(dp)).__name__  # type: ignore
        if "_not_tsdp" in dp_module_name:
            vals["isTsdp"] = False
        else:
            vals["isTsdp"] = True

        return vals

    def _is_list_of_dicts(self, data_points_list: list) -> bool:
        return all(isinstance(elem, dict) for elem in data_points_list)

    def _get_default_data_point_list(self) -> list:
        default_settings_data_frame = pd.DataFrame(DefaultDataPoints)
        default_data_point_list = self._convert_data_point_data_frame(default_settings_data_frame)
        return default_data_point_list
