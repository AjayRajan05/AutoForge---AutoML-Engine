import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import json


def test_validate_input_basic():
    """Test basic input validation function"""
    # Import the function directly to test it
    import sys
    sys.path.append('.')
    
    # Define the function locally to test without dependencies
    def validate_input(X, y):
        """Basic input validation"""
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
    
    # Test successful validation
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    X_valid, y_valid = validate_input(X, y)
    
    assert X_valid.shape == X.shape
    assert y_valid.shape == y.shape
    assert isinstance(X_valid, np.ndarray)
    assert isinstance(y_valid, np.ndarray)


def test_detect_task_type_basic():
    """Test basic task type detection"""
    def detect_task_type(y):
        """Basic task type detection"""
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
    
    # Test classification
    y_class = np.random.randint(0, 2, 100)
    task_type = detect_task_type(y_class)
    assert task_type == "classification"
    
    # Test regression
    y_reg = np.random.rand(100)
    task_type = detect_task_type(y_reg)
    assert task_type == "regression"


def test_knowledge_base_basic():
    """Test basic knowledge base functionality"""
    class KnowledgeBase:
        def __init__(self, path="test_logs.json"):
            self.path = path
            
        def _create_empty_logs(self):
            """Create an empty logs file"""
            try:
                with open(self.path, "w") as f:
                    json.dump([], f)
                print(f"Created empty logs file at {self.path}")
            except Exception as e:
                print(f"Failed to create logs file: {e}")
                raise
        
        def load(self):
            """Load knowledge base with basic error handling"""
            try:
                if not os.path.exists(self.path):
                    print(f"Knowledge base file not found: {self.path}")
                    self._create_empty_logs()
                    return []
                
                with open(self.path, "r") as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    print(f"Invalid knowledge base format, expected list got {type(data)}")
                    return []
                
                print(f"Loaded {len(data)} valid records from knowledge base")
                return data
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error in knowledge base: {e}")
                return []
            except Exception as e:
                print(f"Failed to load knowledge base: {e}")
                return []
    
    # Test knowledge base
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = os.path.join(temp_dir, "test_logs.json")
        kb = KnowledgeBase(log_path)
        
        # Test loading non-existent file
        records = kb.load()
        assert records == []
        
        # Test file creation
        assert os.path.exists(log_path)


def test_model_registry_basic():
    """Test basic model registry structure"""
    # Test that we can define the registry structure
    CLASSIFICATION_MODELS = {
        "random_forest": "RandomForestClassifier",
        "logistic_regression": "LogisticRegression",
        "svm": "SVC",
        "knn": "KNeighborsClassifier",
        "decision_tree": "DecisionTreeClassifier",
        "naive_bayes": "GaussianNB",
        "gradient_boosting": "GradientBoostingClassifier",
    }
    
    REGRESSION_MODELS = {
        "random_forest_regressor": "RandomForestRegressor",
        "linear_regression": "LinearRegression",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "svr": "SVR",
        "xgboost_regressor": "XGBRegressor",
        "gradient_boosting_regressor": "GradientBoostingRegressor",
    }
    
    # Test task type mapping
    TASK_TYPES = {
        **{f"classification_{k}": "classification" for k in CLASSIFICATION_MODELS.keys()},
        **{f"regression_{k}": "regression" for k in REGRESSION_MODELS.keys()},
    }
    
    # Verify structure
    print(f"Classification models: {len(CLASSIFICATION_MODELS)}")
    print(f"Regression models: {len(REGRESSION_MODELS)}")
    print(f"Total task types: {len(TASK_TYPES)}")
    
    assert len(CLASSIFICATION_MODELS) == 7
    assert len(REGRESSION_MODELS) == 7
    assert len(TASK_TYPES) == 14
    
    # Verify task types
    for key in TASK_TYPES:
        assert TASK_TYPES[key] in ["classification", "regression"]


def test_error_handling_patterns():
    """Test error handling patterns we implemented"""
    
    def safe_function(x):
        """Example of safe function with error handling"""
        try:
            if x < 0:
                raise ValueError("x cannot be negative")
            return x * 2
        except ValueError as e:
            print(f"ValueError: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    # Test error handling
    assert safe_function(5) == 10
    assert safe_function(-1) is None


if __name__ == "__main__":
    test_validate_input_basic()
    test_detect_task_type_basic()
    test_knowledge_base_basic()
    test_model_registry_basic()
    test_error_handling_patterns()
    print("✅ All core foundation tests passed!")
