from abc import ABC, abstractmethod
from fashion_recommender.models.base import Model


class ModelSerializer(ABC):
    @abstractmethod
    def write(self, model: Model):
        raise NotImplementedError
