from fashion_recommender.data.articles import Articles
from fashion_recommender.models.doc2vec import Doc2Vec
from fashion_recommender.models.plotters import ModelPlotter
from fashion_recommender.models.serializers.disk import DiskSerializer
from fashion_recommender.models.deserializers.disk import DiskDeserializer


if __name__ == "__main__":
    articles = Articles(window=5)
    model = Doc2Vec(
        n_paragraphs=articles.n_paragraphs,
        vocabulary_size=articles.vocabulay_size,
        ngram_size=articles.window
    )

    model.fit(articles.dataset, epochs=50)
    plotter = ModelPlotter()
    serializer = DiskSerializer()
    serializer.write(model)
    plotter.plot(model=model)
    plotter.show()
    # deserializer = DiskDeserializer()
    # model = deserializer.load(custom_type=Doc2Vec)
    # import pdb; pdb.set_trace()
