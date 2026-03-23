from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np


def build_pipeline(params, model):
    """
    Enhanced pipeline builder with more preprocessing options
    """
    steps = []

    # Enhanced Imputation
    imputer_strategy = params.get("imputer", "mean")
    if imputer_strategy == "knn":
        imputer = KNNImputer(n_neighbors=params.get("knn_neighbors", 5))
    elif imputer_strategy == "constant":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    else:
        imputer = SimpleImputer(strategy=imputer_strategy)
    steps.append(("imputer", imputer))

    # Advanced Scaling
    scaler_type = params.get("scaler", "standard")
    if scaler_type == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler_type == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif scaler_type == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scaler_type == "power":
        steps.append(("scaler", PowerTransformer(method="yeo-johnson")))
    elif scaler_type == "quantile":
        steps.append(("scaler", QuantileTransformer(output_distribution="normal")))

    # Dimensionality Reduction (optional)
    if params.get("dimensionality_reduction", False):
        n_components = params.get("n_components", 0.95)  # Keep 95% variance
        if isinstance(n_components, float):
            n_components = int(n_components * params.get("n_features", 10))
        steps.append(("pca", PCA(n_components=n_components)))

    # Smart Feature Selection
    if params.get("feature_selection", False):
        selection_method = params.get("selection_method", "univariate")
        
        if selection_method == "univariate":
            score_func = f_classif if "classification" in params.get("model", "") else f_regression
            k = min(params.get("k_features", 10), params.get("n_features", 10))
            steps.append(("feature_selection", SelectKBest(score_func=score_func, k=k)))
        
        elif selection_method == "model_based":
            # Use model-based feature selection
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42),
                threshold=params.get("feature_threshold", "median")
            )
            steps.append(("feature_selection", selector))
        
        elif selection_method == "rfe":
            # Recursive Feature Elimination
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=params.get("k_features", 10))
            steps.append(("feature_selection", selector))

    # Model
    steps.append(("model", model))

    return Pipeline(steps)


def build_adaptive_pipeline(params, model, X_sample=None):
    """
    Adaptive pipeline that adjusts preprocessing based on data characteristics
    """
    steps = []

    # Adaptive imputation based on missing data pattern
    imputer_strategy = params.get("imputer", "mean")
    if X_sample is not None:
        missing_ratio = np.isnan(X_sample).sum() / X_sample.size
        if missing_ratio > 0.3:  # High missing data
            imputer = KNNImputer(n_neighbors=3)
        else:
            imputer = SimpleImputer(strategy=imputer_strategy)
    else:
        imputer = SimpleImputer(strategy=imputer_strategy)
    
    steps.append(("imputer", imputer))

    # Adaptive scaling based on feature distribution
    scaler_type = params.get("scaler", "standard")
    if scaler_type == "adaptive" and X_sample is not None:
        # Check for outliers and skewness
        from scipy import stats
        outlier_ratio = np.sum(np.abs(stats.zscore(X_sample)) > 3) / X_sample.size
        skewness = stats.skew(X_sample.flatten())
        
        if outlier_ratio > 0.1:  # Many outliers
            steps.append(("scaler", RobustScaler()))
        elif abs(skewness) > 2:  # Highly skewed
            steps.append(("scaler", PowerTransformer(method="yeo-johnson")))
        else:
            steps.append(("scaler", StandardScaler()))
    else:
        # Regular scaling
        if scaler_type == "standard":
            steps.append(("scaler", StandardScaler()))
        elif scaler_type == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif scaler_type == "robust":
            steps.append(("scaler", RobustScaler()))

    # Model
    steps.append(("model", model))

    return Pipeline(steps)