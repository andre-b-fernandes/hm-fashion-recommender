import matplotlib.pyplot as plt
from fashion_recommender.models.base import Model

class ModelPlotter:
    """Model plotter class to plot models"""
    def plot(self, model: Model):
        history = model.model_training_history()
        for column in history.columns:
            series = history[column]
            ax_plt = series.plot()
            ax_plt.set_xlabel("Epochs")
            ax_plt.set_ylabel(column)

    def show(self):
        plt.show()
