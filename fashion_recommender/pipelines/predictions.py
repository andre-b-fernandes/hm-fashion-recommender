from pyparsing import Optional
from fashion_recommender.models.deserializers.base import ModelDeserializer

from typing import Any, Generator, Iterable,  Type, List, Union
from types import GeneratorType
from fashion_recommender.models.base import Model
from fashion_recommender.sink.vector import VectorSink

from tqdm import tqdm

class ToPredict:
    def __init__(
        self,
        iterable: Union[List, Iterable],
        batch_size: int = 256
    ) -> None:
        self._data = iterable
        self._batch_size = batch_size
    
    def __iter__(self):
        if type(self._data) is GeneratorType:
            yield from enumerate(self._data)
        else:
            with tqdm(total=len(self._data), unit=f"samples") as pbar:
                idx = 0
                while len(self._data) > 0:
                    curr_batch = self._data[:self._batch_size]
                    yield [idx + i for i in range(len(curr_batch))], curr_batch
                    self._data = self._data[self._batch_size:]
                    idx += self._batch_size
                    pbar.update(self._batch_size)


class PredictionPipeline:
    def __init__(self, model_type: Type[Model], deserializer: ModelDeserializer) -> None:
        self.model: Model = deserializer.load(
            custom_type=model_type
        )

    def run(self, to_predict: List[Any], batch_size: int = 256) -> Generator:
        to_predict = ToPredict(iterable=to_predict, batch_size=batch_size)
        for idx, batch in to_predict:
            prediction = self.model.predict(batch)
            yield idx, prediction

class PredictionSinkPipeline(PredictionPipeline):
    def __init__(self, model_type: Type[Model], deserializer: ModelDeserializer, sink: VectorSink) -> None:
        super().__init__(model_type, deserializer)
        self._sink = sink
    
    def run(self, to_predict: List[Any],  structure_name: str, batch_size: int = 256) -> None:
        for idx, predictions in super().run(to_predict, batch_size):
            self._sink.sink(identifiers = idx, data=predictions, structure=structure_name)
