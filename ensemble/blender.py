import numpy as np


class Blender:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        preds = np.array(preds)

        # Majority vote (classification)
        return np.round(preds.mean(axis=0))