import uuid
from datetime import datetime
from io import StringIO
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError
from pandas import DataFrame

from ._file import File


class CSVFile(File):
    def __init__(self, df: DataFrame) -> None:
        self.csv_buffer = StringIO()
        super().__init__(df)
        self.df.to_csv(self.csv_buffer, index=False, header=False)

    def upload_to_s3(self, bucket: Optional[str]) -> Any:
        s3_resource = boto3.resource("s3")
        path = self.create_s3_file_path()
        try:
            s3_resource.Object(bucket, path).put(Body=self.csv_buffer.getvalue())
        except ClientError as e:
            raise e
        return path

    def create_s3_file_path(self) -> Any:
        basename = "temp_csv"
        obj = uuid.uuid4()
        suffix = datetime.now().strftime("%y%m%d_%H%M%S") + ".csv"
        filename = "_".join([basename, suffix])
        path = f"{obj}/{filename}"
        return path
