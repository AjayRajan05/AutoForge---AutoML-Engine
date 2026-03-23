"""
Prediction methods for AutoML
"""

import logging


class PredictionHandler:
    """Handles prediction methods for AutoML"""
    
    def __init__(self, task_type="classification"):
        self.task_type = task_type
        self.logger = logging.getLogger(__name__)
    
    def set_task_type(self, task_type):
        """Update the task type"""
        self.task_type = task_type
    
    def predict(self, X, best_pipeline, feature_engineering_transformer=None):
        """Make predictions with error handling"""
        if best_pipeline is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        try:
            # Use the stored feature engineering transformer if available
            if feature_engineering_transformer and feature_engineering_transformer.is_fitted:
                X_transformed = feature_engineering_transformer.transform(X)
            else:
                X_transformed = X
            
            return best_pipeline.predict(X_transformed)
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def predict_proba(self, X, best_pipeline, ensemble=None, feature_engineering_transformer=None):
        """Make probability predictions for classification"""
        # Check if this is a regression task - if so, raise appropriate error
        if hasattr(best_pipeline, 'predict'):
            # Try to determine if this is regression by checking if predict returns continuous values
            try:
                import numpy as np
                test_pred = best_pipeline.predict(X[:1] if len(X.shape) > 1 else X.reshape(1, -1))
                if isinstance(test_pred[0], (int, float, np.integer, np.floating)):
                    # This looks like regression - check if values are continuous
                    if len(np.unique(test_pred)) > 20:  # Likely regression if many unique values
                        raise ValueError("predict_proba only available for classification tasks")
            except:
                pass
        
        if best_pipeline is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        try:
            # Use the stored feature engineering transformer if available
            if feature_engineering_transformer and feature_engineering_transformer.is_fitted:
                X_transformed = feature_engineering_transformer.transform(X)
            else:
                X_transformed = X
            
            if hasattr(best_pipeline, 'predict_proba'):
                return best_pipeline.predict_proba(X_transformed)
            else:
                if ensemble and hasattr(ensemble, 'predict_proba'):
                    return ensemble.predict_proba(X_transformed)
                else:
                    raise ValueError(
                        "Model does not support predict_proba and no ensemble fallback available"
                    )
        except Exception as e:
            raise ValueError(f"Probability prediction failed: {e}")
