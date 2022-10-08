from fashion_recommender.data.articles import Articles
from fashion_recommender.models.doc2vec import Doc2Vec
from fashion_recommender.models.serializers.disk import DiskSerializer
from fashion_recommender.models.deserializers.disk import DiskDeserializer
from fashion_recommender.pipelines.predictions import PredictionPipeline
from fashion_recommender.pipelines.training import TrainingPipeline


if __name__ == "__main__":
    articles = Articles(window=5)
    model = Doc2Vec(
        n_paragraphs=articles.n_paragraphs,
        vocabulary_size=articles.vocabulay_size,
        ngram_size=articles.window
    )
    serializer = DiskSerializer()
    pipeline = TrainingPipeline(model=model, data=articles, serializer=serializer)
    pipeline.run(epochs=10)
    # deserializer = DiskDeserializer()
    # pipeline = PredictionPipeline(model_type=Doc2Vec, deserializer=deserializer)
    # results = pipeline.run(to_predict=)
    # lel = list(results)

    # import pdb; pdb.set_trace()
