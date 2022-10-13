from abc import ABCMeta, abstractmethod
import tensorflow as tf
import pandas as pd

from typing import Any, Dict

from fashion_recommender.data.base import FashionDataset

class Model(tf.keras.Model, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extras: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def fit(self, data: FashionDataset, epochs: int):
        super(Model, self).fit(data.dataset, epochs=epochs)

    def model_training_history(self) -> pd.DataFrame:
        """Model training results as a pd.DataFrame"""
        return pd.DataFrame(self.history.history)
    
    def get_config(self):
        raise NotImplementedError

    def add_extra(self, extra, name: str) -> None:
        self.extras[name] =  extra
    
    @abstractmethod
    def predict(self, x, batch_size=None, verbose="auto", steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        raise NotImplementedError
