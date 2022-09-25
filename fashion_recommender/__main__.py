from fashion_recommender.data.articles import ArticlesDataset
from fashion_recommender.models.doc2vec import Doc2Vec
from fashion_recommender.models.plotters import ModelPlotter


if __name__ == "__main__":
    articles_dataset = ArticlesDataset(window_size=3)
    ds = articles_dataset.to_tf_dataset()
    model = Doc2Vec(
        vocabulary_size=articles_dataset.vocabulary_size,
        ngram_size=articles_dataset.window
    )

    model.fit(ds, epochs=40)
    plotter = ModelPlotter()
    plotter.plot(model=model)
    plotter.show()
