from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, List, NamedTuple
from typing import OrderedDict as OrderedDictType

import pandas as pd
from pandas import DataFrame

from ..data_type import TimeSeriesFormat


class Column(NamedTuple):
    name: ColumnNameType
    value: ColumnValueType


InvestmentIdType = str
DataPointAliasType = str
ColumnNameType = Any
ColumnValueType = Any
ColumnListType = List[Column]
ColumnDictType = OrderedDictType[ColumnNameType, ColumnValueType]
MetaDataKeyType = Any
MetaDataValueType = Any


def pivot_time_series_data(
    df: DataFrame, id_vars: list = ["Id"], variable_column_name: str = "Date", value_column_name: str = "value"
) -> DataFrame:
    """
    Pivot Time series dataframe that we receive from the DataFrame.
    """
    pivoted_df = pd.melt(df, id_vars=id_vars, var_name=variable_column_name, value_name=value_column_name)
    pivoted_df[variable_column_name] = pivoted_df[variable_column_name].str[-10:]
    pivoted_df[variable_column_name] = pd.to_datetime(pivoted_df[variable_column_name])
    pivoted_df = pivoted_df.sort_values(by=["Id", variable_column_name])

    return pivoted_df


@dataclass(frozen=True)
class InvestmentDataRequest:
    """Stores parameters for getting investment data from DO APIs"""

    investment_ids: list = field(default_factory=list)
    data_points: DataFrame = field(default_factory=DataFrame)


@dataclass
class InvestmentDataResults:
    """Stores multiple investment results returned by DO API, indexed by investment ID"""

    _data: OrderedDictType[InvestmentIdType, InvestmentDataResult] = field(default_factory=OrderedDict)

    # Commands
    def add_column_data(self, investment_id: InvestmentIdType, alias: DataPointAliasType, column_data: ColumnListType) -> None:
        self._ensure_investment_id_exists(investment_id)
        self._data[investment_id].add_column_data(alias, column_data)

    def add_meta_data(self, investment_id: InvestmentIdType, key: MetaDataKeyType, value: MetaDataValueType) -> None:
        self._ensure_investment_id_exists(investment_id)
        self._data[investment_id].add_meta_data(key, value)

    def merge_with(self, other_results: InvestmentDataResults, in_place: bool = False) -> InvestmentDataResults:
        new_results = self
        if not in_place:
            new_results = copy.deepcopy(self)

        for investment_id, other_result in other_results._data.items():
            new_results._ensure_investment_id_exists(investment_id)
            new_results._data[investment_id].merge_with(other_result)

        return new_results

    # Queries
    def get_column_data_value(
        self,
        investment_id: InvestmentIdType,
        alias: DataPointAliasType,
        col_name: ColumnNameType,
        default_value: ColumnValueType = None,
    ) -> ColumnValueType:
        return self._data.get(investment_id, InvestmentDataResult()).get_column_data_value(alias, col_name, default_value)

    def get_meta_data(
        self, investment_id: InvestmentIdType, key: MetaDataKeyType, default_value: MetaDataValueType = None
    ) -> MetaDataValueType:
        return self._data.get(investment_id, InvestmentDataResult()).get_meta_data(key, default_value)

    def get_data_point_alias_to_col_dict(self) -> dict:
        if len(self._data) == 0:
            return dict()
        one_investment: InvestmentDataResult = list(self._data.values())[0]
        return one_investment.get_data_point_alias_to_col_dict()

    def as_list(self) -> list:
        new_list: list = []
        for investment_id, result in self._data.items():
            # TODO: find a better place for this check
            if result.get_meta_data("entitled", None) is False:
                continue
            new_item: dict = dict()
            # Always include the Investment ID column
            new_item["Id"] = investment_id
            for alias in result.get_aliases():
                new_item.update(result.get_column_data(alias))

            new_list.append(new_item)
        return new_list

    def as_long_frame(self) -> DataFrame:
        all_investments_data_frame = DataFrame()
        for investment_id, result in self._data.items():
            all_df = DataFrame()
            ts_df = DataFrame()
            non_ts_data: dict = dict()
            df_list = []
            data_point_cols = []
            # Always include the Investment ID column
            non_ts_data["Id"] = investment_id
            for alias in result.get_aliases():
                # get investment_id data point values dict
                data_point_value_dict = result.get_column_data(alias)

                # get investment_id data point meta
                data_point_meta_data = result.get_meta_data(alias, None)
                isTsdp = data_point_meta_data.get("isTsdp", None)
                data_point_name = data_point_meta_data.get("name", data_point_meta_data.get("datapointName", None))
                # columns order
                data_point_cols.append(data_point_name)
                if isTsdp:
                    # pivot ts data
                    data_point_value_dict["Id"] = investment_id
                    df_list.append(pivot_time_series_data(DataFrame([data_point_value_dict]), value_column_name=data_point_name))
                    # merge ts data
                    ts_df = reduce(lambda left, right: pd.merge(left, right, on=["Id", "Date"], how="outer"), df_list)
                else:
                    non_ts_data.update(data_point_value_dict)
            # merge ts and non ts
            all_df = (
                pd.merge(DataFrame([non_ts_data]), ts_df, on=["Id"], how="outer") if not ts_df.empty else DataFrame([non_ts_data])
            )
            # reorder columns
            all_df = all_df[["Id", "Date"] + data_point_cols] if not ts_df.empty else all_df[data_point_cols]
            all_investments_data_frame = pd.concat([all_investments_data_frame, all_df])

        return all_investments_data_frame.reset_index(drop=True)

    def as_data_frame(self, order_cols_by: list = [], time_series_format: TimeSeriesFormat = TimeSeriesFormat.WIDE) -> DataFrame:
        if time_series_format == time_series_format.LONG:
            return self.as_long_frame()
        if time_series_format == time_series_format.LONG_WITHOUT_NAME:
            return self.as_long_frame().drop(columns=["Name"])

        new_df = DataFrame(self.as_list())
        if order_cols_by:
            # Always include the Investment ID column
            if "Id" not in order_cols_by:
                order_cols_by = ["Id"] + order_cols_by
            return new_df[order_cols_by]
        else:
            return new_df

    # Helpers
    def _ensure_investment_id_exists(self, investment_id: InvestmentIdType) -> None:
        if investment_id not in self._data:
            self._data[investment_id] = InvestmentDataResult()


@dataclass
class InvestmentDataResult:
    """Stores one investment result containing multiple data points, indexed by data point alias.
    Each data point can contain one or more values (e.g., if data point contains time-series data).
    Each value represents a column."""

    _data: OrderedDictType[DataPointAliasType, ColumnDictType] = field(default_factory=OrderedDict)
    _meta_data: OrderedDictType[MetaDataKeyType, MetaDataValueType] = field(default_factory=OrderedDict)

    # Commands
    def add_column_data(self, alias: DataPointAliasType, column_data: ColumnListType) -> None:
        self._ensure_alias_exists(alias)
        self._data[alias].update(column_data)

    def add_meta_data(self, key: MetaDataKeyType, value: MetaDataValueType) -> None:
        self._meta_data[key] = value

    def merge_with(self, other_result: InvestmentDataResult) -> None:
        for alias, other_values in other_result._data.items():
            self._ensure_alias_exists(alias)
            self._data[alias].update(other_values)
        self._meta_data.update(other_result._meta_data)

    # Queries
    def get_column_data(self, alias: DataPointAliasType, default_value: ColumnDictType = OrderedDict()) -> ColumnDictType:
        return self._data.get(alias, default_value)

    def get_column_data_value(
        self, alias: DataPointAliasType, col_name: ColumnNameType, default_value: ColumnValueType = None
    ) -> ColumnValueType:
        return self.get_column_data(alias).get(col_name, default_value)

    def get_meta_data(self, key: MetaDataKeyType, default_value: MetaDataValueType = None) -> MetaDataValueType:
        return self._meta_data.get(key, default_value)

    def get_data_point_alias_to_col_dict(self) -> dict:
        result = dict()
        for alias, columns in self._data.items():
            result[alias] = list(columns.keys())
        return result

    def get_aliases(self) -> list:
        return list(self._data.keys())

    # Helpers
    def _ensure_alias_exists(self, alias: DataPointAliasType) -> None:
        if alias not in self._data:
            self._data[alias] = OrderedDict()
