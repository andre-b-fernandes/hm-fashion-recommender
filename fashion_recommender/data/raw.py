import pandas as pd

from fashion_recommender.data.inputs.dataset_config import CSVDatasetConfig


def csv_to_pandas(input_csv: CSVDatasetConfig) -> pd.DataFrame:
    """
    A function which creates a pandas dataframe 
    out of a CSVDataSetConfig
    """
    return pd.read_csv(
        input_csv.file_path,
        usecols=input_csv.columns + [input_csv.index],
        index_col=input_csv.index
    )
