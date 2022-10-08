from fashion_recommender.models.base import Model
from fashion_recommender.models.serializers.base import ModelSerializer

from pathlib import Path

import pickle

class DiskSerializer(ModelSerializer):
    MODELS_OUTPUT_PATH = "models/"
    EXTRAS_OUTPUT_PATH = "extras/"

    def write(self, model: Model) -> None:
        model.save(
            filepath=f"{self.MODELS_OUTPUT_PATH}{model.name}",
            overwrite=True,
            save_format="tf"
        )

        for name, value in model.extras.items():
            filepath = f"{self.EXTRAS_OUTPUT_PATH}{model.name}/{name}.p"
            path = Path(filepath)
            path.parent.mkdir(exist_ok=True, parents=True)
            path.unlink(missing_ok=True)
            path.touch(exist_ok=False)
            with path.open(mode="wb") as file:
                pickle.dump(value, file=file)
