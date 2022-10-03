from fashion_recommender.data.inputs.dataset_config import CSVDatasetConfig
from fashion_recommender.data.raw import csv_to_pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from tqdm import tqdm

from typing import Generator, List
from typing_extensions import Self


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
    """
    This function returns the articles dataset
    #TODO: Eliminate memory duplication from this function.
    (Currently we have it on line 49)
    """
    raw_df = _articles_raw_dataset()
    raw_df = raw_df.sample(frac=1)

    txt_series = ( raw_df["prod_name"] + " " 
        # + raw_df["product_type_name"] + " "
        + raw_df["product_group_name"] + " " 
        + raw_df["graphical_appearance_name"] + " " 
        + raw_df["colour_group_name"]
    )
    #TODO: apply lowercase
    #TODO: Eliminate words not in vocab
    return pd.DataFrame(txt_series, columns=["txt"])
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html


def sparse_matrix_to_sparse_tensor(X):
    # https://stackoverflow.com/questions/40896157/scipy-sparse-csr-matrix-to-tensorflow-sparsetensor-mini-batch-gradient-descent
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

class Articles:
    """
    A class representing the articles dataset.
    One-Hot Encodes the articles vocabulary.
    Also count-vectorizes the articles.
    """
    BATCH_SIZE = 128
    # number of batches of size BATCH_SIZE to prefetch
    PREFETCH_SIZE = 5
    MAX_ROWS = 20000
    MAX_FEATURES = 1024
    MIN_NUMBER_OF_WORDS = 5
    
    def __init__(self, window: int) -> Self:
        word_encoder = OneHotEncoder(handle_unknown="ignore")
        pid_encoder = LabelEncoder()
        paragraph_vectorizer = CountVectorizer(max_features=self.MAX_FEATURES, stop_words={"english"})
        _dataset = _articles_dataset()

        def _lower_and_filter(dataset: pd.DataFrame, vocabulary: set) -> pd.DataFrame:
            def _auxiliar(sentence: str) -> List[str]:
                parts = [part.lower() for part in sentence.split()]
                return [part for part in parts if part in vocabulary]

            dataset["txt"] = dataset["txt"].apply(_auxiliar)
            dataset = dataset[dataset["txt"].map(len) > self.MIN_NUMBER_OF_WORDS]
            return dataset

        def _n_grams_from_sentence(sentence: List[str]):
            # garantee lowercasing for vocabulary matching
            return [
                word_encoder.transform(
                    np.asarray(
                        sentence[i:i+window]
                    ).reshape(window, 1)
                )
                for i
                in range(len(sentence)- window + 1)
            ]

        def _iterator(_dataset):
            _iterations = _dataset.T["txt"].map(lambda words: len(words) - window + 1).sum()
            batch_to_eliminate = []
            max_batch_size = 500

            with tqdm(total=_iterations, desc="Dataset iteration...", unit=f" {window}-grams") as pbar:
                # for pid, row in dataset.iterrows():
                for pid in _dataset.columns:
                    batch_to_eliminate.append(pid)

                    series = _dataset[pid]
                    txt = series.values[0]
                    
                    if len(batch_to_eliminate) % max_batch_size == 0:
                        _dataset.drop(columns=batch_to_eliminate, inplace=True)
                        batch_to_eliminate.clear()
                    # txt = row.txt

                    encoded_pid = pid_encoder.transform([pid])[0]
                    n_grams = _n_grams_from_sentence(sentence=txt)
                    paragraph_repr = paragraph_vectorizer.transform(
                        [" ".join(txt)]
                    ).toarray()
                    paragraph_repr = np.squeeze(paragraph_repr)

                    for n_gram in n_grams:
                        context = n_gram[:-1].toarray()
                        context = np.sum(context, axis=0).astype(np.int)
                        target = n_gram[-1].toarray()
                        pbar.update(1)
                        yield (encoded_pid, paragraph_repr, context),  target


        paragraph_vectorizer.fit(_dataset["txt"])
        word_encoder.fit(
            np.asarray(
                list(paragraph_vectorizer.vocabulary_.keys())
            ).reshape(-1, 1)
        )
        vocabulary = set(paragraph_vectorizer.vocabulary_.keys())
        _dataset = _lower_and_filter(dataset=_dataset, vocabulary=vocabulary)
        
        if self.MAX_ROWS:
            _dataset = _dataset.head(self.MAX_ROWS)
        
        pid_encoder.fit(_dataset.index)
        self.n_paragraphs = len(_dataset)

        _dataset = _dataset.T
        features, targets, paragraphs = [], [], []
        for (paragraph, paragraph_repr, context), target in _iterator(_dataset):
            paragraphs.append([paragraph])
            features.append(np.vstack([paragraph_repr, context]))
            targets.append(target)

        self.dataset = tf.data.Dataset.from_tensor_slices(
            ((paragraphs, features), targets)
        ).batch(
            self.BATCH_SIZE, drop_remainder=True
        ).shuffle(self.BATCH_SIZE)
        self.window = window
        self.vocabulary = vocabulary
        self.size = len(paragraphs)
    
    @property
    def vocabulay_size(self):
        return len(self.vocabulary)

    def __len__(self):
        return self.size
