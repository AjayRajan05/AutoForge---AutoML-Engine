import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def detect_column_types(X):
    if isinstance(X, pd.DataFrame):
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    else:
        # fallback (numpy array)
        numeric_cols = list(range(X.shape[1]))
        categorical_cols = []

    return numeric_cols, categorical_cols


def build_preprocessor(X):
    numeric_cols, categorical_cols = detect_column_types(X)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor