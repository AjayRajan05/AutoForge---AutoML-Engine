class Pipeline:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def fit(self, X, y):
        if self.preprocessor:
            X = self.preprocessor.fit_transform(X)
        self.model.fit(X, y)

    def predict(self, X):
        if self.preprocessor:
            X = self.preprocessor.transform(X)
        return self.model.predict(X)