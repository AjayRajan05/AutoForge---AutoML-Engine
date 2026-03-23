import numpy as np


def is_classification(y):
    return len(set(y)) < 20


def to_numpy(X):
    return X.values if hasattr(X, "values") else X