import tensorflow as tf

from fashion_recommender.models.base import Model


class Doc2Vec(Model):
    EMBEDDING_SIZE = 256

    def __init__(self, vocabulary_size: int, ngram_size: int):
        super(Doc2Vec, self).__init__()
        self.ngram_size = ngram_size
        self.vocabulary_size = vocabulary_size
        self.paragraph_embeddings = tf.keras.layers.Embedding(
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
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        )

    def call(self, inputs):
        # EXPLORE USE TF SPARSE TENSORS
        paragraph, context = inputs[:,0], inputs[:,1]

        paragraph_embedding = self.paragraph_embeddings(paragraph)
        context_embedding = self.context_embeddings(context)
            
        reduced = tf.reduce_mean(
            paragraph_embedding + context_embedding,
            axis=2
        )
        reduced = tf.expand_dims(reduced, axis=1)

        return self.softmax(reduced)

