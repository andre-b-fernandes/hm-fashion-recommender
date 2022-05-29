import tensorflow as tf


class Doc2Vec(tf.keras.Model):
    EMBEDDING_SIZE = 256

    def __init__(self, vocabulary_size: int, ngram_size: int):
        super(Doc2Vec, self).__init__()
        self.paragraph_embeddings = tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=self.EMBEDDING_SIZE,
            input_length=1
        )

        self.context_embeddings = tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=self.EMBEDDING_SIZE,
            input_length=ngram_size - 1
        )

        self.softmax = tf.keras.layers.Softmax()

       # TODO: add this
       #  self.target_embedding = tf.keras.layers.Embedding(
       #      input_dim=vocabulary_size,
       #      output_dim=self.EMBEDDING_SIZE,
       #      input_length=1
       #  )

    def call(self, inputs):
        # EXPLORE USE TF SPARSE TENSORS
        paragraph, context = inputs
        paragraph_embedding = self.paragraph_embeddings(paragraph)
        context_embedding = self.context_embeddings(context)
        mean_context = tf.math.reduce_mean(context_embedding, axis=1)

        return tf.reduce_sum(
            tf.math.multiply(
                tf.squeeze(paragraph_embedding),
                mean_context
            ), 
            axis=2
        )
