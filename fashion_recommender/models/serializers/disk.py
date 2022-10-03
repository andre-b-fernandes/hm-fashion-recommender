from fashion_recommender.models.base import Model

class DiskSerializer:
    MODELS_OUTPUT_PATH = "models/"

    def write(self, model: Model) -> None:
        model.save(
            filepath=f"{self.MODELS_OUTPUT_PATH}{model.name}",
            overwrite=True,
            save_format="tf"
        )
