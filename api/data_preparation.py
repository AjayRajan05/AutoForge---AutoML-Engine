"""
Data preparation logic for AutoML
"""

import logging
from core.data_type_detector import DataTypeDetector
from core.dataset_optimizer import DatasetOptimizer
from meta_learning.dataset_profiler import profile_dataset
from .utils import validate_input, detect_task_type
from .feature_engineering import FeatureEngineeringTransformer


class DataPreparation:
    """Handles data validation, detection, and feature engineering"""
    
    def __init__(self, use_dataset_optimization=True):
        self.use_dataset_optimization = use_dataset_optimization
        self.dataset_optimizer = DatasetOptimizer() if use_dataset_optimization else None
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self, X, y):
        """
        Handle input validation, data type detection, and feature engineering
        Returns: (X_processed, y_processed, data_type_results, feature_metadata, primary_data_type)
        """
        # Input validation
        X_validated, y_validated = validate_input(X, y)
        
        # Data-type intelligence detection
        data_type_detector = DataTypeDetector()
        data_type_results = data_type_detector.detect_data_type(X_validated, y_validated)
        primary_data_type = data_type_results["primary_type"]
        
        self.logger.info(f"Detected primary data type: {primary_data_type}")
        self.logger.info(f"Data type recommendations: {data_type_results['recommendations']}")
        
        # Task type detection
        task_type = detect_task_type(y_validated)
        self.logger.info(f"Detected task type: {task_type}")
        
        # Smart feature engineering constraints for small datasets
        disable_complex_features = X_validated.shape[0] < 100
        disable_neural_networks = X_validated.shape[0] < 100
        if disable_complex_features:
            self.logger.info("Small dataset detected: disabling complex feature engineering to avoid overfitting")
        if disable_neural_networks:
            self.logger.info("Small dataset detected: disabling neural networks to avoid instability")
        
        # Apply feature engineering using the transformer approach
        feature_engineering_transformer = FeatureEngineeringTransformer(
            primary_data_type=primary_data_type, 
            task_type=task_type,
            disable_complex_features=disable_complex_features
        )
        
        if primary_data_type == "time_series":
            X_engineered, feature_metadata = self._apply_time_series_engineering(
                X_validated, y_validated, data_type_results
            )
            self.logger.info(f"Applied time series feature engineering: {feature_metadata['engineered_features']} features")
        elif primary_data_type == "text":
            X_engineered, feature_metadata = self._apply_text_engineering(
                X_validated, data_type_results
            )
            self.logger.info(f"Applied text feature engineering: {feature_metadata['engineered_features']} features")
        else:
            # Use the new transformer for tabular data
            X_engineered = feature_engineering_transformer.fit_transform(X_validated, y_validated)
            feature_metadata = feature_engineering_transformer.feature_metadata
            self.logger.info(f"Applied smart feature engineering: {feature_metadata['engineered_features']} features")
        
        # Store feature metadata
        if 'total_features' not in feature_metadata:
            feature_metadata['total_features'] = X_engineered.shape[1]
        
        # Dataset optimization (adaptive sampling)
        dataset_metadata = {}
        if self.dataset_optimizer:
            (X_optimized, y_optimized), strategy, dataset_metadata = (
                self.dataset_optimizer.optimize_dataset(X_engineered, y_validated, task_type)
            )
            self.logger.info(f"Dataset optimization: {strategy}")
            if strategy == "sampled":
                self.logger.info(
                    f"Using sampled dataset: {dataset_metadata['sampled_size']:,} / "
                    f"{dataset_metadata['original_size']:,} samples"
                )
            X_final, y_final = X_optimized, y_optimized
        else:
            X_final, y_final = X_engineered, y_validated
        
        # Dataset profiling
        dataset_profile = profile_dataset(X_final, y_final)
        self.logger.info(f"Dataset profile: {dataset_profile}")
        
        return (
            X_final, y_final, data_type_results, feature_metadata, 
            primary_data_type, task_type, dataset_profile, dataset_metadata,
            feature_engineering_transformer, disable_neural_networks
        )
    
    def _apply_time_series_engineering(self, X, y, data_type_results):
        """Apply time series feature engineering"""
        try:
            from features.time_series_features import engineer_time_series_features
            
            datetime_cols = data_type_results["detection_results"]["time_series"].get(
                "datetime_columns", []
            )
            datetime_col = datetime_cols[0] if datetime_cols else None

            X_engineered, metadata = engineer_time_series_features(
                X, y,
                datetime_col=datetime_col,
                max_lags=min(12, len(X) // 10),
                rolling_windows=[3, 7, min(30, len(X) // 5)]
            )
            return X_engineered, metadata

        except Exception as e:
            self.logger.warning(f"Time series engineering failed: {e}")
            return X, {"engineered_features": 0}

    def _apply_text_engineering(self, X, data_type_results):
        """Apply text feature engineering"""
        try:
            from features.text_features import engineer_text_features
            
            text_cols = data_type_results["detection_results"]["text"].get("text_columns", [])

            X_engineered, metadata = engineer_text_features(
                X,
                text_columns=text_cols,
                max_features=min(5000, X.shape[0] // 2),
                ngram_range=(1, 2)
            )
            return X_engineered, metadata

        except Exception as e:
            self.logger.warning(f"Text engineering failed: {e}")
            return X, {"engineered_features": 0}
