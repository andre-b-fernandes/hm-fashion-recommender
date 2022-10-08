from typing import List
import numpy as np
import tensorflow as tf
from fashion_recommender.data.articles import Articles

from fashion_recommender.models.base import Model


class Doc2Vec(Model):
    EMBEDDING_SIZE = 128

    def __init__(self, n_paragraphs: int, vocabulary_size: int, ngram_size: int):
        super(Doc2Vec, self).__init__()
        self.ngram_size = ngram_size
        self.vocabulary_size = vocabulary_size
        self.n_paragraphs = n_paragraphs
        self.paragraph_embeddings = tf.keras.layers.Embedding(
            input_dim=n_paragraphs,
            output_dim=self.EMBEDDING_SIZE,
            input_length=1
        )

        self.word_embeddings = tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=self.EMBEDDING_SIZE,
            input_length=vocabulary_size
        )

        self.context_embeddings = tf.keras.layers.Embedding(
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
        paragraphs, features = inputs
        paragraph_repr, context_repr = features[:,0,:], features[:,1,:]

        paragraph_embedding = self.paragraph_embeddings(paragraphs)
        context_embedding = self.context_embeddings(context_repr)
        word_embedding = self.word_embeddings(paragraph_repr)

        reduced = tf.reduce_mean(
            paragraph_embedding + context_embedding + word_embedding,
            axis=2
        )
        reduced = tf.expand_dims(reduced, axis=1)

        return self.softmax(reduced)

    def get_config(self) -> dict:
        return {
            "vocabulary_size": self.vocabulary_size,
            "ngram_size": self.vocabulary_size,
            "n_paragraphs": self.n_paragraphs
        }
    
    @classmethod
    def from_config(cls, config: dict) -> "Doc2Vec":
        return cls(**config)

    def predict(self, ids: List[int]) -> np.array:
        import pdb; pdb.set_trace()
        transfomed_ids = self.extras["pid_encoder"].transform(ids)
        return self.paragraph_embeddings.get_weights()[0][transfomed_ids]
