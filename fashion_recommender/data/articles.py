from fashion_recommender.data.inputs.dataset_config import CSVDatasetConfig
from fashion_recommender.data.raw import csv_to_pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

from typing import Generator

import numpy as np
import pandas as pd
import tensorflow as tf


def _articles_raw_dataset() -> pd.DataFrame:
    """"""
    return csv_to_pandas(
        CSVDatasetConfig(
            file_path="data/articles.csv",
            columns=[
                "prod_name",
                "product_type_name",
                "product_group_name",
                "graphical_appearance_name",
                "colour_group_name"
            ],
            index="article_id"
        )
    )

def _articles_dataset(n_rows: int = None) -> pd.DataFrame:
    """
    This function returns the articles dataset
    filtered by a number of rows.
    Filtering will also apply a shuffle.
    #TODO: Eliminate memory duplication from this function.
    (Currently we have it on line 49)
    """
    raw_df = _articles_raw_dataset()
    if n_rows:
        raw_df = raw_df.sample(frac=1)
        raw_df = raw_df.head(n_rows)

    txt_series = ( raw_df["prod_name"] + " " 
        # + raw_df["product_type_name"] + " "
        + raw_df["product_group_name"] + " " 
        + raw_df["graphical_appearance_name"] + " " 
        + raw_df["colour_group_name"]
    )
    return pd.DataFrame(txt_series, columns=["txt"])
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html


def sparse_matrix_to_sparse_tensor(X):
    # https://stackoverflow.com/questions/40896157/scipy-sparse-csr-matrix-to-tensorflow-sparsetensor-mini-batch-gradient-descent
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

# TODO: shuffle the articles dataset
# TODO: Eliminate words not in vocabulary
# TODO: Filter number of lines
# TODO: Check for a way to create ngrams with CountVectorizer directly.

class ArticlesDataset:
    """
    A class representing the articles dataset.
    One-Hot Encodes the articles vocabulary.
    Also count-vectorizes the articles.
    """
    BATCH_SIZE = 128
    # number of batches of size BATCH_SIZE to prefetch
    PREFETCH_SIZE = 5
    MAX_ROWS = 2250
    MAX_FEATURES = 1024

    def __init__(self, window_size: int):
        self.window = window_size
        self._dataset = _articles_dataset(n_rows=self.MAX_ROWS)
        self.word_encoder = OneHotEncoder(handle_unknown="ignore")
        self.paragraph_vectorizer = CountVectorizer(max_features=self.MAX_FEATURES, stop_words={"english"})
        self.paragraph_vectorizer.fit(self._dataset["txt"])
        self.word_encoder.fit(
            np.asarray(
                list(self.paragraph_vectorizer.vocabulary_.keys())
            ).reshape(-1, 1)
        )

    @property
    def vocabulary_size(self) -> int:
        """The vocabulary size"""
        return len(self.paragraph_vectorizer.vocabulary_)

    def __len__(self) -> int:
        return self._dataset.shape[0]

    def __iter__(self) -> Generator:
        """
        A function which iterates over this dataset
        Yields the paragraph vector, a context (2-dim) and a target
        """
        for paragraph in self._dataset["txt"][:self.MAX_ROWS]:
            n_grams = self._n_grams_from_sentence(sentence=paragraph)
            paragraph_repr = self.paragraph_vectorizer.transform([paragraph]).toarray()
            paragraph_repr = np.squeeze(paragraph_repr)
            
            if np.sum(paragraph_repr):
                for n_gram in n_grams:
                    context = n_gram[:-1].toarray()
                    context = np.sum(context, axis=0).astype(np.int)
                    target = n_gram[-1].toarray()
                    if np.sum(context) and np.sum(target):
                        yield (paragraph_repr, context),  target


    def _n_grams_from_sentence(self, sentence: str):
        # garantee lowercasing for vocabulary matching
        parts = [part.lower() for part in sentence.split()]
        return [
            self.word_encoder.transform(
                np.asarray(
                    parts[i:i+self.window]
                ).reshape(self.window, 1)
            )
            for i
            in range(len(parts)- self.window + 1)
        ]

    def to_tf_dataset(self) -> tf.data.Dataset:
        """
        Convert this dataset into a tf.data.Dataset
        using a generator
        """
        features, targets = [], []
        for (paragraph, context), target in iter(self):
            features.append(np.vstack([paragraph, context]))
            targets.append(target)

        return tf.data.Dataset.from_tensor_slices((features, targets)).batch(self.BATCH_SIZE, drop_remainder=True)

