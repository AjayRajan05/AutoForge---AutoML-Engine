"""
🔥 UNIVERSAL PARAMETER HANDLER
Handle ANY parameter combination gracefully
"""

import logging
from typing import Dict, Any, List
from models.registry import MODEL_REGISTRY, MODEL_SEARCH_SPACE, DEFAULT_PARAMS

logger = logging.getLogger(__name__)


class UniversalParameterHandler:
    """
    🔥 Universal Parameter Handler - Handle ANY parameter combination
    """
    
    def __init__(self):
        """Initialize universal parameter mappings"""
        self.parameter_mappings = {
            # Handle prefixed/unprefixed names
            'logistic_regression': ['logistic_regression', 'classification_logistic_regression'],
            'neural_network': ['neural_network', 'classification_neural_network', 'regression_neural_network'],
            'svm': ['svm', 'classification_svm', 'svr', 'regression_svr'],
            'random_forest': ['random_forest', 'classification_random_forest', 'regression_random_forest_regressor'],
            'xgboost': ['xgboost', 'classification_xgboost', 'regression_xgboost_regressor'],
            'gradient_boosting': ['gradient_boosting', 'classification_gradient_boosting', 'regression_gradient_boosting_regressor'],
            'knn': ['knn', 'classification_knn', 'regression_knn'],
            'decision_tree': ['decision_tree', 'classification_decision_tree', 'regression_decision_tree'],
            'naive_bayes': ['naive_bayes', 'classification_naive_bayes'],
            'linear_regression': ['linear_regression', 'regression_linear_regression'],
            'ridge': ['ridge', 'regression_ridge'],
            'lasso': ['lasso', 'regression_lasso'],
        }
        
        # Create reverse mapping for quick lookup
        self.reverse_mappings = {}
        for standard, variants in self.parameter_mappings.items():
            for variant in variants:
                self.reverse_mappings[variant] = standard
        
        # Default fallback parameters
        self.fallback_params = {
            'classification': {
                'solver': 'lbfgs',
                'C': 1.0,
                'kernel': 'rbf',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
            },
            'regression': {
                'solver': 'lbfgs',
                'C': 1.0,
                'kernel': 'rbf',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
            }
        }
    
    def normalize_model_name(self, model_name: str) -> str:
        """
        Convert any model name to standard form
        
        Args:
            model_name: Input model name (any format)
            
        Returns:
            Standard model name
        """
        # Direct lookup in reverse mappings
        if model_name in self.reverse_mappings:
            return self.reverse_mappings[model_name]
        
        # Try prefix removal
        if model_name.startswith('classification_'):
            base_name = model_name.replace('classification_', '')
            if base_name in self.parameter_mappings:
                return base_name
        
        elif model_name.startswith('regression_'):
            base_name = model_name.replace('regression_', '')
            if base_name in self.parameter_mappings:
                return base_name
        
        # Return as-is if unknown (will be handled later)
        return model_name
    
    def get_valid_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get valid parameters for a model
        
        Args:
            model_name: Standard model name
            
        Returns:
            Dictionary of valid parameters with defaults
        """
        # Try to get from search space
        if model_name in MODEL_SEARCH_SPACE:
            valid_params = MODEL_SEARCH_SPACE[model_name].copy()
            
            # Add default values
            for param_name, param_space in valid_params.items():
                if isinstance(param_space, tuple) and len(param_space) == 2:
                    # Numeric range - use middle value as default
                    valid_params[param_name] = {
                        'type': 'range',
                        'min': param_space[0],
                        'max': param_space[1],
                        'default': (param_space[0] + param_space[1]) / 2
                    }
                elif isinstance(param_space, list):
                    # Categorical - use first value as default
                    valid_params[param_name] = {
                        'type': 'categorical',
                        'values': param_space,
                        'default': param_space[0]
                    }
            
            return valid_params
        
        # Try to get from default params
        if model_name in DEFAULT_PARAMS:
            return {k: {'type': 'fixed', 'default': v} for k, v in DEFAULT_PARAMS[model_name].items()}
        
        # Return empty if unknown
        return {}
    
    def is_valid_value(self, param_name: str, value: Any, param_info: Dict[str, Any]) -> bool:
        """
        Check if a parameter value is valid
        
        Args:
            param_name: Parameter name
            value: Parameter value
            param_info: Parameter information
            
        Returns:
            True if valid
        """
        param_type = param_info.get('type', 'fixed')
        
        if param_type == 'range':
            try:
                min_val, max_val = param_info['min'], param_info['max']
                return min_val <= value <= max_val
            except (TypeError, ValueError):
                return False
        
        elif param_type == 'categorical':
            return value in param_info['values']
        
        elif param_type == 'fixed':
            return value == param_info['default']
        
        return False
    
    def validate_and_correct(self, model_name: str, params: Dict[str, Any], task_type: str = 'classification') -> Dict[str, Any]:
        """
        Validate and auto-correct ANY parameter set
        
        Args:
            model_name: Model name (any format)
            params: Input parameters
            task_type: Task type (classification/regression)
            
        Returns:
            Corrected parameters
        """
        # Normalize model name
        standard_name = self.normalize_model_name(model_name)
        
        # Get valid parameters for this model
        valid_params = self.get_valid_params(standard_name)
        
        # Auto-correct invalid parameters
        corrected = {}
        
        # Process pipeline parameters first
        pipeline_params = ['scaler', 'imputer', 'feature_selection', 'model']
        for param_name in pipeline_params:
            if param_name in params:
                corrected[param_name] = params[param_name]
        
        # Process model parameters
        for key, value in params.items():
            if key in pipeline_params:
                continue  # Already handled
            
            if key in valid_params:
                if self.is_valid_value(key, value, valid_params[key]):
                    corrected[key] = value
                else:
                    # Use default value
                    corrected[key] = valid_params[key]['default']
                    logger.warning(f"Corrected invalid parameter {key}={value} to {corrected[key]}")
            else:
                # Unknown parameter - skip or use fallback
                if task_type in self.fallback_params and key in self.fallback_params[task_type]:
                    corrected[key] = self.fallback_params[task_type][key]
        
        # Ensure we have some basic parameters
        if not corrected or len(corrected) <= len(pipeline_params):
            # Use fallback parameters
            fallback = self.get_fallback_params(standard_name, task_type)
            corrected.update(fallback)
            logger.info(f"Using fallback parameters for {model_name}")
        
        return corrected
    
    def get_fallback_params(self, model_name: str, task_type: str) -> Dict[str, Any]:
        """
        Get fallback parameters for a model
        
        Args:
            model_name: Model name
            task_type: Task type
            
        Returns:
            Fallback parameters
        """
        # Try to get from DEFAULT_PARAMS
        standard_name = self.normalize_model_name(model_name)
        if standard_name in DEFAULT_PARAMS:
            return DEFAULT_PARAMS[standard_name]
        
        # Use generic fallback based on model type
        if 'logistic' in standard_name.lower():
            return {'solver': 'lbfgs', 'C': 1.0}
        elif 'svm' in standard_name.lower() or 'svr' in standard_name.lower():
            return {'kernel': 'rbf', 'C': 1.0}
        elif 'forest' in standard_name.lower():
            return {'n_estimators': 100, 'max_depth': 6}
        elif 'xgboost' in standard_name.lower():
            return {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
        elif 'neural' in standard_name.lower():
            return {'optimizer': 'adam', 'learning_rate': 0.001}
        else:
            # Generic fallback
            return self.fallback_params.get(task_type, {})
    
    def handle_unknown_model(self, model_name: str, params: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """
        Handle completely unknown model names
        
        Args:
            model_name: Unknown model name
            params: Input parameters
            task_type: Task type
            
        Returns:
            Safe parameters for a similar known model
        """
        logger.warning(f"Unknown model: {model_name}, using fallback")
        
        # Try to guess model type from name
        if 'logistic' in model_name.lower():
            fallback_model = 'logistic_regression'
        elif 'svm' in model_name.lower() or 'svr' in model_name.lower():
            fallback_model = 'svm'
        elif 'forest' in model_name.lower():
            fallback_model = 'random_forest'
        elif 'xgboost' in model_name.lower():
            fallback_model = 'xgboost'
        elif 'neural' in model_name.lower():
            fallback_model = 'neural_network'
        else:
            # Use simple model as fallback
            fallback_model = 'logistic_regression' if task_type == 'classification' else 'linear_regression'
        
        return self.validate_and_correct(fallback_model, params, task_type)


# Global instance
universal_parameter_handler = UniversalParameterHandler()
