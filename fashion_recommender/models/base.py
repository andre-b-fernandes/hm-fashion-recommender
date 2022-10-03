from abc import ABCMeta
import tensorflow as tf
import pandas as pd

class Model(tf.keras.Model, metaclass=ABCMeta):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def model_training_history(self) -> pd.DataFrame:
        """Model training results as a pd.DataFrame"""
        return pd.DataFrame(self.history.history)
    
    def get_config(self):
        raise NotImplementedError