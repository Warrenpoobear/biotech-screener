import io
import json
import logging
import os
from enum import Enum
from typing import List, Optional

import pandas as pd
from pandas import DataFrame

from ..direct._backend_apis._signed_url_backend import SignedUrlBackend
from ..direct._config import _Config
from ._data_objects import CSVFile, TempTable
from ._exceptions import TempTableNameNotFoundException

_logger = logging.getLogger(__name__)

DL_ACCOUNT = os.getenv("DL_ACCOUNT")

region = os.getenv("DL_AWS_REGION", "us-east-1")
is_external_user = os.getenv("IS_EXTERNAL_USER")
uim_user_id = os.getenv("UIM_USER_ID")
bucket_name = os.getenv("DL_TEMP_TABLE_BUCKET_NAME")
redshift_database_name = os.getenv("REDSHIFT_DATABASE_NAME")


class Source(Enum):
    DATALAKE = 1
    LAKEHOUSE = 2


def query(query_str: str, temp_tables: Optional[List[TempTable]] = None) -> DataFrame:
    """
    :bdg-ref-danger:`Upcoming Feature <../upcoming_feature>`

    Retrieve the results of a SQL query from the Morningstar Data Lake.

    Args:
        query_str: SQL query to be executed in Morningstar Data Lake
        temp_tables: A list of temporary tables that will exist for the duration of the query.

    :Returns:
        A DataFrame object with results of the SQL query:

    :Examples:

        Submit a query using a temp table

    ::

        import morningstar_data as md
        import pandas as pd


        df_my_table = pd.DataFrame({'sec_id': ['F0GBR0606A', 'F00000SYAH', 'F00000WP51'], 'closing_price': [128.372, 23.02, 528.33]})
        df_query = md.datalake.query(query_str = 'select * from my_table;', temp_tables = [md.datalake.TempTable('my_table', df_my_table)])

    :Output:

        ===========  ===============
         sec_id       closing_price
        ===========  ===============
        F0GBR0606A      128.372
        F00000SYAH       23.02
        F00000WP51      528.33
        ===========  ===============

    :Errors:
        InvalidQueryException: When ``query_str`` contains invalid SQL syntax.

        UnauthorizedDataLakeAccessError: When the calling user is not authorized to query the Morningstar Data Lake.

        TempTableNameNotFoundException: When one or more temp tables being used are not found in the query string.

    """
    signed_url_backend = SignedUrlBackend()
    config = _Config()

    if temp_tables:
        _logger.debug(f"Query function invoked with temp tables {temp_tables}")

        missing_tables = []
        for table in temp_tables:
            if table.name is None or table.name.strip() == "":
                _logger.debug("Temp table name is empty")
                raise TempTableNameNotFoundException from None

            if table.name not in query_str:
                missing_tables.append(table.name)

            body = {"is_temp": True, "table_name": table.name}

            # get a signed url to upload this table to
            _logger.debug(f"Uploading to stage table_name: {table.name}")
            resp = json.loads(signed_url_backend.do_post_request(url=config.temp_table_url(), data=body))

            table.table_stage_id = resp["table_id"]
            upload_url = resp["upload_url"]

            resp = signed_url_backend.put_csv_file(csv=CSVFile(table.df), signed_upload_url=upload_url)

            _logger.info(f"Upload complete under the following s3 id {table.table_stage_id} with url {upload_url}")

        if len(missing_tables) > 0:
            _logger.warn(
                f" Not all temp_tables are in query_str, make sure your query is correct. Missing table names: {missing_tables}"
            )

    df = _get_query_as_df(signed_url_backend, config, query_str, temp_tables)

    return df


def _construct_query_endpoint_payload(query_str: str, temp_tables: Optional[List[TempTable]] = None) -> str:
    # Sample body from md package to POST /query endpoint

    # {
    #     "query": "select * from xyz join temp_a join temp_b;",
    #     "tables": [
    #         {"table_id": "sd98f-dsf89sdf-sdf-df", "table_name": "temp_a", "columns": [{"name": "col1", "type": "int64"}, {"name": "col2", "type": "float"}]},
    #         {"table_id": "ad98f-dsf89sdf-sdf-df", "table_name": "temp_b",
    # "columns": [{"name": "col1", "type": "int64"}, {"name": "col2", "type": "float"}]},
    #     ],
    # }

    # Within a Temp table the DF argument is not optional
    if temp_tables:
        payload = {
            "query": query_str,
            "tables": [
                {
                    "table_id": table.table_stage_id,
                    "table_name": table.name,
                    "columns": [{"name": col, "type": str(table.df[col].dtype)} for col in table.df.columns],  # type: ignore
                }
                for table in temp_tables
            ],
        }
    else:
        payload = {"query": query_str}

    return json.dumps(payload)


def _get_query_as_df(
    signed_url_backend: SignedUrlBackend, config: _Config, query_str: str, temp_tables: Optional[List[TempTable]]
) -> pd.DataFrame:
    payload = _construct_query_endpoint_payload(query_str, temp_tables)

    _logger.debug(f"Hitting the data lake query endpoint with this payload\n{payload}\n")

    download_url_response = signed_url_backend.do_post_request(url=config.query_api_url(), data=payload)
    download_url = json.loads(download_url_response)["url"]

    _logger.info(f"Downloading from s3 at the url {download_url}")

    resp = signed_url_backend.get_bytes(url=download_url)
    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except pd.errors.EmptyDataError:
        _logger.warning(f"Empty Data Error for query {query}, returning an empty dataframe")
        return pd.DataFrame()

    return df
