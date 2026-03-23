"""
Utility functions for AutoML data validation and task detection
"""

import numpy as np
import pandas as pd


def validate_input(X, y):
    """
    Comprehensive input validation and data quality checks
    """
    # Convert to numpy arrays for consistency
    if hasattr(X, 'values'):  # pandas DataFrame
        X_array = X.values
    else:
        X_array = np.array(X)

    if hasattr(y, 'values'):  # pandas Series
        y_array = y.values
    else:
        y_array = np.array(y)

    # Check dimensions
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError(f"X and y have mismatched dimensions: {X_array.shape[0]} vs {y_array.shape[0]}")

    if X_array.shape[0] < 10:
        raise ValueError("Dataset too small: need at least 10 samples")

    if X_array.shape[1] < 1:
        raise ValueError("Dataset has no features")

    # Check for missing values
    missing_ratio = np.isnan(X_array).sum() / X_array.size
    if missing_ratio > 0.8:
        raise ValueError(f"Too many missing values: {missing_ratio:.2%}")

    # Check target variable
    unique_targets = len(np.unique(y_array[~np.isnan(y_array)]))
    if unique_targets < 2:
        raise ValueError("Target variable has less than 2 unique classes")

    return X_array, y_array


def detect_task_type(y):
    """
    Automatically detect if this is classification or regression
    """
    # Remove NaN values for analysis
    y_clean = y[~np.isnan(y)]

    # Check if target is integer-like with few unique values
    unique_values = len(np.unique(y_clean))
    is_integer = np.issubdtype(y_clean.dtype, np.integer) or \
                 np.allclose(y_clean, np.round(y_clean))

    # Heuristic: if integer-like and <= 20 unique values, treat as classification
    if is_integer and unique_values <= 20:
        return "classification"
    else:
        return "regression"
