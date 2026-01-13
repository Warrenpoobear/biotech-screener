import re
from typing import Any, Optional

from pandas import DataFrame

from ._table import Table


class TempTable(Table):
    def __init__(self, name: str, df: DataFrame, table_stage_id: Optional[str] = None) -> None:
        super().__init__(name, df)
        self.table_stage_id = table_stage_id

    def _validate_column_names(self, df: DataFrame) -> DataFrame:
        columns = df.columns

        for col in columns:
            if len(col) > 127:
                raise ValueError(f"Invalid column name {col}: Column names must be less than 128 characters in length.")
            if not (col[0] == "_" or col[0].isalpha()):
                raise ValueError(f"Invalid column name {col}: Column names can only start with a letter or underscore.")
            if not re.search("^[\w|\$]*$", col):
                raise ValueError(
                    f"Invalid column name {col}: Column names can only contain alphanumeric characters, underscores (_) or dollar signs ($)."
                )

            return df

    def get_create_query(self) -> Any:
        query_string = f'CREATE TEMP TABLE "{self.name}" ('
        for col_type, col_name in zip(self.df.dtypes, self.df):  # type: ignore
            converted_type = self._pd_dtype_to_redshift_dtype(col_type.name)
            query_string += f'\n"{col_name}" {converted_type},'

        query_string = query_string[:-1] + "\n)"

        return query_string

    def get_insert_query(self, bucket_path: str, bucket_name: Optional[str], iam_role: Optional[str]) -> str:  # type: ignore
        query = f"""
                COPY {self.name}
                FROM 's3://{bucket_name}/{bucket_path}'
                IAM_ROLE '{iam_role}'
                CSV;
                """
        return query

    def _pd_dtype_to_redshift_dtype(self, dtype: Any) -> str:
        if dtype.startswith("int64"):
            return "BIGINT"
        elif dtype.startswith("int"):
            return "INTEGER"
        elif dtype.startswith("float"):
            return "REAL"
        elif dtype.startswith("datetime"):
            return "TIMESTAMP"
        elif dtype == "bool":
            return "BOOLEAN"
        else:
            return "VARCHAR(256)"
