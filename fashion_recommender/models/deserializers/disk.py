from typing import Type
import tensorflow as tf
import pickle

from pathlib import Path

from fashion_recommender.models.base import Model
from fashion_recommender.models.deserializers.base import ModelDeserializer

class DiskDeserializer(ModelDeserializer):
    MODELS_OUTPUT_PATH = "models/"
    EXTRAS_OUTPUT_PATH = "extras/"

    def load(self, custom_type: Type[Model]) -> Model:
        model: Model = tf.keras.models.load_model(
            filepath=f"{self.MODELS_OUTPUT_PATH}{custom_type.__name__}",
            custom_objects={
                custom_type.__name__: custom_type
            }
        )
        extras_path = Path(f"{self.EXTRAS_OUTPUT_PATH}{model.name}")
        for extra_path in extras_path.iterdir():
            with extra_path.open(mode="rb") as file:
                extra = pickle.load(file)
                model.add_extra(extra=extra, name=extra_path.stem)
                
        return model
