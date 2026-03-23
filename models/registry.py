from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from typing import Dict, Any

# Import advanced models
try:
    from models.advanced_models import LightGBMWrapper, SimpleNeuralNetwork
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    LightGBMWrapper = None
    SimpleNeuralNetwork = None

# Enhanced MODEL_REGISTRY with advanced models
CLASSIFICATION_MODELS = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "decision_tree": DecisionTreeClassifier,
    "naive_bayes": GaussianNB,
    "gradient_boosting": GradientBoostingClassifier,
    "xgboost": XGBClassifier,
}

REGRESSION_MODELS = {
    "random_forest_regressor": RandomForestRegressor,
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "svr": SVR,
    "xgboost_regressor": XGBRegressor,
    "gradient_boosting_regressor": GradientBoostingRegressor,
}

# Add advanced models if available
if ADVANCED_MODELS_AVAILABLE:
    CLASSIFICATION_MODELS.update({
        "lightgbm": lambda **kwargs: LightGBMWrapper(task_type="classification", **kwargs),
        "neural_network": lambda **kwargs: SimpleNeuralNetwork(task_type="classification", **kwargs),
        "classification_lightgbm": lambda **kwargs: LightGBMWrapper(task_type="classification", **kwargs),
        "classification_neural_network": lambda **kwargs: SimpleNeuralNetwork(task_type="classification", **kwargs),
    })
    
    REGRESSION_MODELS.update({
        "lightgbm_regressor": lambda **kwargs: LightGBMWrapper(task_type="regression", **kwargs),
        "neural_network_regressor": lambda **kwargs: SimpleNeuralNetwork(task_type="regression", **kwargs),
        "regression_lightgbm_regressor": lambda **kwargs: LightGBMWrapper(task_type="regression", **kwargs),
        "regression_neural_network_regressor": lambda **kwargs: SimpleNeuralNetwork(task_type="regression", **kwargs),
    })

# Unified MODEL_REGISTRY with prefixed names for search space compatibility
MODEL_REGISTRY = {
    **CLASSIFICATION_MODELS,
    **REGRESSION_MODELS,
    # Add prefixed names for search space compatibility
    **{f"classification_{k}": v for k, v in CLASSIFICATION_MODELS.items()},
    **{f"regression_{k}": v for k, v in REGRESSION_MODELS.items()}
}

# Task type mapping
TASK_TYPES = {
    **{f"classification_{k}": "classification" for k in CLASSIFICATION_MODELS.keys()},
    **{f"regression_{k}": "regression" for k in REGRESSION_MODELS.keys()},
}

# Add advanced model task types
if ADVANCED_MODELS_AVAILABLE:
    TASK_TYPES.update({
        "classification_lightgbm": "classification",
        "classification_neural_network": "classification",
        "regression_lightgbm_regressor": "regression",
        "regression_neural_network_regressor": "regression",
    })

def get_model_class(model_name: str):
    """
    Get model class by name with proper instantiation for advanced models
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model class or factory function
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_available_models(task_type: str = "all") -> Dict[str, Any]:
    """
    Get available models for a specific task type
    
    Args:
        task_type: "classification", "regression", or "all"
        
    Returns:
        Dictionary of available models
    """
    if task_type == "classification":
        return CLASSIFICATION_MODELS
    elif task_type == "regression":
        return REGRESSION_MODELS
    elif task_type == "all":
        return MODEL_REGISTRY
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def is_advanced_model(model_name: str) -> bool:
    """
    Check if a model is an advanced model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Whether the model is advanced
    """
    advanced_models = ["lightgbm", "neural_network", "lightgbm_regressor", "neural_network_regressor"]
    return any(adv in model_name for adv in advanced_models)


def create_model_instance(model_name: str, **kwargs):
    """
    Create model instance with proper handling for advanced models
    
    Args:
        model_name: Name of the model
        **kwargs: Model parameters
        
    Returns:
        Model instance
    """
    model_class = get_model_class(model_name)
    
    if is_advanced_model(model_name):
        # Advanced models need special handling
        if "lightgbm" in model_name:
            task_type = "classification" if "regression" not in model_name else "regression"
            return LightGBMWrapper(task_type=task_type, **kwargs)
        elif "neural_network" in model_name:
            task_type = "classification" if "regressor" not in model_name else "regression"
            return SimpleNeuralNetwork(task_type=task_type, **kwargs)
    else:
        # Standard sklearn models
        return model_class(**kwargs)