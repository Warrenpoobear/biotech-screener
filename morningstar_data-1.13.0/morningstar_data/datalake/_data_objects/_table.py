from abc import ABC, abstractmethod

from pandas import DataFrame


class Table(ABC):
    def __init__(self, name: str, df: DataFrame) -> None:
        self.name = name
        self.df = self._validate_column_names(df)

    @abstractmethod
    def _validate_column_names(self, df: DataFrame) -> None:
        """
        Validate that column names in Data Frame do not contain reserved words and conform to Redshift's requirements:
        https://docs.aws.amazon.com/redshift/latest/dg/r_names.html

        Args:
            DataFrame
        Returns:
            The validated DataFrame
        """
        pass

    @abstractmethod
    def get_create_query(self) -> None:
        pass

    @abstractmethod
    def get_insert_query(self) -> None:
        pass
