from pandas import DataFrame

from .. import _error_messages
from .._data_objects import DataPoints, Investments
from .._exceptions import ResourceNotFoundError


def _get_investment_ids(investment_object: Investments) -> list:
    investment_id_list = investment_object.get_investment_ids()
    if not investment_id_list:
        raise ResourceNotFoundError(_error_messages.RESOURCE_NOT_FOUND_INVESTMENT_DATA)
    return investment_id_list


def _get_data_points(investment_object: Investments, data_point_object: DataPoints, display_name: bool = False) -> DataFrame:
    data_point_object.validate_data_point_ids()
    return data_point_object.get_data_points(
        list_id=investment_object.list_id, search_criteria_id=investment_object.search_criteria, display_name=display_name
    )


def _get_data_point_col_names(data_points: DataFrame, data_point_alias_to_cols: dict) -> list:
    col_names = []
    for _, data_point in data_points.iterrows():
        alias = data_point.get("alias", "")
        columns = data_point_alias_to_cols.get(alias, [])
        col_names.extend(columns)
    return col_names


# def _build_result_data_frame(investment_data_list: list) -> DataFrame:
#     result = DataFrame(investment_data_list)
#     # TODO : .where() with Pandas 1.1.5 has a bug which converts all the columns including float to object
#     # this was solved in Pandas update 1.3.5. Until we update Pandas, we will have to use this _replace_null_values()
#     # as a workaround
#     # results = df.where(result.notnull(), None)
#     return result
