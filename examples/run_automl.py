from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from api.automl import AutoML

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

automl = AutoML(n_trials=20)
automl.fit(X_train, y_train)

preds = automl.predict(X_test)

print(preds)