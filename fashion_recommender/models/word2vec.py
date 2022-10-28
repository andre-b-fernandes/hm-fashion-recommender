import tensorflow as tf
import numpy as np
from typing import List
from fashion_recommender.data.articles import Articles
# TODO: Solve articles and subclass

from fashion_recommender.models.base import Model


# TODO: Solve code duplication
class Word2Vec(Model):
    EMBEDDING_SIZE = 128

    def __init__(self, vocabulary_size: int, ngram_size: int):
        super(Word2Vec, self).__init__()
        self.ngram_size = ngram_size
        self.vocabulary_size = vocabulary_size

        self.word_embeddings = tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=self.EMBEDDING_SIZE,
            input_length=vocabulary_size
        )

        self.softmax = tf.keras.layers.Softmax()
        self.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        )
            

    def fit(self, data: Articles, epochs: int):
        self.add_extra(data.pid_encoder, name="pid_encoder")
        return super().fit(data, epochs)

    def call(self, inputs: tuple):
        # EXPLORE USE TF SPARSE TENSORS
        _paragraphs, features = inputs
        paragraph_repr, _context_repr = features[:,0,:], features[:,1,:]

        word_embedding = self.word_embeddings(paragraph_repr)

        reduced = tf.reduce_mean(
            word_embedding,
            axis=2
        )
        reduced = tf.expand_dims(reduced, axis=1)

        return self.softmax(reduced)

    def get_config(self) -> dict:
        return {
            "vocabulary_size": self.vocabulary_size,
            "ngram_size": self.vocabulary_size,
        }
    
    @classmethod
    def from_config(cls, config: dict) -> "Word2Vec":
        return cls(**config)

    def predict(self, ids: List[int]) -> np.array:
        raise NotImplementedError
    