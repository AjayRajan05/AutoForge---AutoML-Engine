"""
Core coordinator for AutoML - orchestrates the main workflow
"""

import logging
from typing import Dict, Any, Optional

from ..data_preparation import DataPreparation
from ..optimization import OptimizationManager
from ..pipeline_builder import PipelineBuilder
from ..prediction import PredictionHandler
from ..feature_engineering import FeatureEngineeringTransformer
from core.progress_tracker import create_progress_tracker


class AutoMLCoordinator:
    """
    Core coordinator that orchestrates the AutoML workflow
    """
    
    def __init__(self, n_trials=50, timeout=None, cv=3, 
                 use_adaptive_optimization=True, use_dataset_optimization=True,
                 use_caching=True, show_progress=True):
        """Initialize coordinator with configuration"""
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.use_adaptive_optimization = use_adaptive_optimization
        self.use_dataset_optimization = use_dataset_optimization
        self.use_caching = use_caching
        self.show_progress = show_progress

        # Initialize specialized components
        self.progress_tracker = create_progress_tracker(show_progress=show_progress) if show_progress else None
        self.data_preparation = DataPreparation(use_dataset_optimization=use_dataset_optimization)
        self.optimization_manager = OptimizationManager(
            n_trials=n_trials, timeout=timeout, cv=cv, 
            use_adaptive_optimization=use_adaptive_optimization
        )
        self.pipeline_builder = PipelineBuilder(use_caching=use_caching)
        self.prediction_handler = PredictionHandler()

        # State tracking
        self.best_pipeline = None
        self.best_model_name = None
        self.task_type = None
        self.dataset_profile = None
        self.feature_metadata = None
        self.feature_engineering_transformer = None
        self.optimization_metadata = {}
        self.dataset_metadata = {}

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def run_automl_workflow(self, X, y):
        """
        Execute the complete AutoML workflow
        Returns: fitted AutoML instance
        """
        try:
            # Start progress tracking
            if self.progress_tracker:
                self.progress_tracker.start_optimization(self.n_trials, "unknown")
            
            # Step 1: Data preparation
            (X_processed, y_processed, data_type_results, feature_metadata, 
             primary_data_type, task_type, dataset_profile, dataset_metadata,
             feature_engineering_transformer, disable_neural_networks) = self.data_preparation.prepare_data(X, y)
            
            # Store state
            self.task_type = task_type
            self.feature_metadata = feature_metadata
            self.feature_engineering_transformer = feature_engineering_transformer
            self.dataset_profile = dataset_profile
            self.dataset_metadata = dataset_metadata
            
            # Update prediction handler with task type
            self.prediction_handler.set_task_type(task_type)
            
            # Step 2: Run optimization search (meta-learning + hyperparameter search)
            best_params, best_model_name, optimization_metadata = self.optimization_manager.run_optimization(
                X_processed, y_processed, task_type, dataset_profile, 
                dataset_metadata, feature_metadata, data_type_results, 
                self.progress_tracker, disable_neural_networks
            )
            
            # Store optimization results
            self.best_model_name = best_model_name
            self.optimization_metadata = optimization_metadata
            
            # Fix metric direction for user clarity
            if task_type == "regression":
                # Convert negative MSE to positive MSE for user display
                self.best_score = -optimization_metadata.get("best_score", 0.0)
            else:
                # For classification, keep accuracy as positive
                self.best_score = optimization_metadata.get("best_score", 0.0)
            
            # Step 3: Build final pipeline with best parameters
            self.best_pipeline = self.pipeline_builder.build_final_pipeline(
                X_processed, y_processed, best_params, best_model_name,
                task_type, self.n_trials, optimization_metadata,
                dataset_metadata, data_type_results["primary_type"], feature_metadata
            )
            
            if self.progress_tracker:
                self.progress_tracker.finish_optimization()
            
            return self
            
        except Exception as e:
            self.logger.error(f"AutoML workflow failed: {e}")
            if self.progress_tracker:
                self.progress_tracker.finish_optimization()
            raise

    def predict(self, X):
        """Make predictions using the trained pipeline"""
        return self.prediction_handler.predict(
            X, self.best_pipeline, self.feature_engineering_transformer
        )

    def predict_proba(self, X):
        """Make probability predictions for classification"""
        # Check if this is a regression task
        if self.task_type == "regression":
            raise ValueError("predict_proba only available for classification tasks")
            
        return self.prediction_handler.predict_proba(
            X, self.best_pipeline, None, self.feature_engineering_transformer
        )
