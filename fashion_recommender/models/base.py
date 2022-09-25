from abc import ABCMeta
import tensorflow as tf
import pandas as pd

class Model(tf.keras.Model, metaclass=ABCMeta):
    def call(self, *args, **kwargs):
        raise NotImplementedError

    def model_training_history(self) -> pd.DataFrame:
        """Model training results as a pd.DataFrame"""
        return pd.DataFrame(self.history.history)