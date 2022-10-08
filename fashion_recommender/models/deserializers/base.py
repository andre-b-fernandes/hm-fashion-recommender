from abc import ABC, abstractmethod

from typing import Type
from fashion_recommender.models.base import Model

class ModelDeserializer(ABC):
    @abstractmethod
    def load(self, custom_type: Type[Model]):
        raise NotImplementedError
