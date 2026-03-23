from automl.api.automl import AutoML
from sklearn.datasets import load_iris


def test_automl_runs():
    X, y = load_iris(return_X_y=True)

    automl = AutoML(n_trials=5)
    automl.fit(X, y)

    preds = automl.predict(X)

    assert len(preds) == len(y)