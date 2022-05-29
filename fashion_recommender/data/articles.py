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

def _articles_dataset() -> pd.DataFrame:
    """"""
    raw_df = _articles_raw_dataset()
    txt_series = ( raw_df["prod_name"] + " " 
        + raw_df["product_type_name"] + " "
        + raw_df["product_group_name"] + " " 
        + raw_df["graphical_appearance_name"] + " " 
        + raw_df["colour_group_name"]
    )
    return pd.DataFrame(txt_series, columns=["txt"])
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html



class ArticlesDataset:
    """
    A class representing the articles dataset.
    One-Hot Encodes the articles vocabulary.
    Also count-vectorizes the articles.
    """
    BATCH_SIZE = 128
    # number of batches of size BATCH_SIZE to prefetch
    PREFETCH_SIZE = 5

    def __init__(self, window_size: int):
        self.window = window_size
        self._dataset = _articles_dataset()
        self.word_encoder = OneHotEncoder(handle_unknown="ignore")
        self.paragraph_vectorizer = CountVectorizer()
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

    def __iter__(self) -> Generator:
        """
        A function which iterates over this dataset
        Yields the paragraph vector, a context (2-dim) and a target
        """
        for paragraph in self._dataset["txt"]:
            n_grams = self._n_grams_from_sentence(sentence=paragraph)
            paragraph_repr = self.paragraph_vectorizer.transform([paragraph]).toarray()
            for n_gram in n_grams:
                context = n_gram[:-1]
                target = np.expand_dims(n_gram[-1], axis=0)
                yield (paragraph_repr, context),  target
                # TODO: build generator https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator

    def _n_grams_from_sentence(self, sentence: str):
        parts = sentence.split()
        return [
            self.word_encoder.transform(
                np.asarray(
                    parts[i:i+self.window]
                ).reshape(self.window, 1)
            ).toarray()
            for i
            in range(len(parts)- self.window + 1)
        ]

    def to_tf_dataset(self) -> tf.data.Dataset:
        """
        Convert this dataset into a tf.data.Dataset
        using a generator
        """
        paragraph_spec = tf.TensorSpec(
            shape=(1, self.vocabulary_size),
            dtype=tf.int64
        )
        context_spec = tf.TensorSpec(
            shape=(self.window - 1, self.vocabulary_size),
            dtype=tf.float64
        )
        target_spec = tf.TensorSpec(
            shape=(1, self.vocabulary_size),
            dtype=tf.float64
        )
        tf_ds = tf.data.Dataset.from_generator(
            self.__iter__,
            output_signature=(
                (paragraph_spec, context_spec),
                target_spec
            )
        )
        return tf_ds.batch(self.BATCH_SIZE).prefetch(self.PREFETCH_SIZE)
    #TODO: pass this into a model. Find what is happening slowing the batch processing
