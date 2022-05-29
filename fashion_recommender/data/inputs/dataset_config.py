from dataclasses import dataclass

from typing import List


@dataclass
class CSVDatasetConfig:
    """
    A class which configures a CSV dataset
    to be read.
    File path is the path in disk.
    Columns are the columns which are supposed to be
    read into memory.
    """
    file_path: str
    columns: List[str]
    index: str

