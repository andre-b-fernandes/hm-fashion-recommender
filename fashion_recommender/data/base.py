import tensorflow as tf
from typing import List

from abc import ABC, abstractmethod

class DatasetExtra:
    def __init__(self, extra) -> None:
        self._extra = extra

class FashionDataset(ABC):
    """"""
    def __init__(self) -> None:
        self.dataset = self._create_dataset()
    
    @abstractmethod
    def _create_dataset(self) -> tf.data.Dataset:
        raise NotImplementedError
