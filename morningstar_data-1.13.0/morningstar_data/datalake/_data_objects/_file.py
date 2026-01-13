from abc import ABC, abstractmethod

from pandas import DataFrame


class File(ABC):
    def __init__(self, df: DataFrame) -> None:
        self.df = df

    @abstractmethod
    def upload_to_s3(self, bucket: str) -> None:
        """Upload a file to an S3 bucket
        :param bucket: Bucket to upload to
        :return: True if file was uploaded, else False
        """
        pass

    @abstractmethod
    def create_s3_file_path(self) -> None:
        """
        Constructs an S3 file path comprised of a UUID and timestamp
        """
        pass
