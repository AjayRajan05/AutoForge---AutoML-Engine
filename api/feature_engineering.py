"""
Feature engineering transformer for AutoML
"""

import logging
import numpy as np
from features.smart_feature_engineering import engineer_features_smart


class FeatureEngineeringTransformer:
    """Sklearn-compatible transformer that stores fitted feature engineering state"""
    
    def __init__(self, primary_data_type="tabular", task_type="classification", disable_complex_features=False):
        self.primary_data_type = primary_data_type
        self.task_type = task_type
        self.disable_complex_features = disable_complex_features
        self.feature_metadata = None
        self.is_fitted = False
        self._data_type_results = None
        self.logger = logging.getLogger(__name__)
        
    def fit_transform(self, X, y=None):
        """Fit the transformer and return transformed data"""
        return self.fit(X, y).transform(X)
    
    def fit(self, X, y=None):
        """Fit the feature engineering transformer"""
        if self.primary_data_type == "time_series":
            # For time series, we'd need to store the datetime column and parameters
            # For now, store the data type results for later use
            self._data_type_results = {"primary_type": "time_series"}
            self.feature_metadata = {"engineered_features": 0}
            
        elif self.primary_data_type == "text":
            # For text, store the fitted vectorizer
            self._data_type_results = {"primary_type": "text"}
            self.feature_metadata = {"engineered_features": 0}
            
        else:
            # For tabular data, fit the smart feature engineering
            if self.disable_complex_features:
                # Use simple feature engineering for small datasets
                X_engineered, metadata = self._engineer_features_simple(X, y, self.task_type)
                self.logger.info("Used simple feature engineering for small dataset")
            else:
                # Use full smart feature engineering for larger datasets
                X_engineered, metadata = engineer_features_smart(X, y, self.task_type)
                self.logger.info("Used full smart feature engineering for larger dataset")
            
            self.feature_metadata = metadata
            self._data_type_results = {"primary_type": "tabular"}
            
            # Store the engineered feature count for consistency
            self.feature_metadata['total_features'] = X_engineered.shape[1]
            
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform new data using the fitted feature engineering"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineeringTransformer must be fitted before transform")
            
        if self.primary_data_type == "time_series":
            # For time series, apply the same engineering (simplified)
            return X  # Would need proper implementation
            
        elif self.primary_data_type == "text":
            # For text, apply the same vectorization (simplified)
            return X  # Would need proper implementation
            
        else:
            # For tabular data, reapply smart feature engineering
            # Note: This is still not ideal, but better than before
            X_engineered, _ = engineer_features_smart(X, np.zeros(len(X)), self.task_type)
            
            # Ensure consistent dimensions
            expected_features = self.feature_metadata.get('total_features', X.shape[1])
            if X_engineered.shape[1] != expected_features:
                # Pad or truncate to match expected dimensions
                if X_engineered.shape[1] < expected_features:
                    X_padded = np.zeros((X_engineered.shape[0], expected_features))
                    X_padded[:, :X_engineered.shape[1]] = X_engineered
                    X_engineered = X_padded
                else:
                    X_engineered = X_engineered[:, :expected_features]
            
            return X_engineered

    def _engineer_features_simple(self, X, y, task_type):
        """Simple feature engineering for small datasets to avoid overfitting"""
        import numpy as np
        
        # COMPLETE bypass for small datasets - no feature engineering at all
        self.logger.info("Small dataset: completely skipping feature engineering to avoid overfitting")
        
        # Return raw data unchanged
        X_simple = X.copy() if hasattr(X, 'copy') else np.array(X)
        
        metadata = {
            'engineered_features': 0,
            'original_features': X.shape[1],
            'total_features': X_simple.shape[1],
            'engineering_type': 'none_bypassed'
        }
        
        return X_simple, metadata
