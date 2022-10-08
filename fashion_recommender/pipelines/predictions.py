from fashion_recommender.models.deserializers.base import ModelDeserializer

from typing import Any, Generator, Iterable,  Type, List, Union
from types import GeneratorType
from fashion_recommender.models.base import Model

class ToPredict:
    def __init__(self, iterable: Union[List, Iterable], batch_size: int = 256) -> None:
        self._data = iterable
        self._batch_size = batch_size
    
    def __iter__(self):
        if type(self._data) is GeneratorType:
            yield from self._data

        else:
            while len(self._data) > 0:
                curr_batch = self._data[:self._batch_size]
                yield curr_batch
                self._data = self._data[self._batch_size:]


class PredictionPipeline:
    def __init__(self, model_type: Type[Model], deserializer: ModelDeserializer) -> None:
        self.model: Model = deserializer.load(
            custom_type=model_type
        )

    def run(self, to_predict: List[Any]) -> Generator:
        to_predict = ToPredict(iterable=to_predict)
        for batch in to_predict:
            prediction = self.model.predict(batch)
            yield prediction
