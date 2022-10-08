from fashion_recommender.models.base import Model
from fashion_recommender.data.base import FashionDataset
from fashion_recommender.models.plotters import ModelPlotter
from fashion_recommender.models.serializers.base import ModelSerializer

class TrainingPipeline:
    def __init__(self, model: Model, data: FashionDataset, serializer: ModelSerializer) -> None:
        self.model = model
        self.data = data 
        self.serializer = serializer 
    
    def run(self, epochs: int):
        self.model.fit(self.data, epochs=epochs)
        plotter = ModelPlotter()
        self.serializer.write(self.model)
        plotter.plot(model=self.model)
        plotter.show()
