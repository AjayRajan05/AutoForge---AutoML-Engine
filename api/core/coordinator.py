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

# Import advanced components with error handling
try:
    from nas.revolutionary_nas import AdvancedNAS
    NAS_AVAILABLE = True
except ImportError:
    NAS_AVAILABLE = False
    AdvancedNAS = None

try:
    from multimodal.intelligent_multimodal import AdvancedMultimodalAutoML
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    AdvancedMultimodalAutoML = None

try:
    from distributed.intelligent_distributed import AdvancedDistributedAutoML
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    AdvancedDistributedAutoML = None


class AutoMLCoordinator:
    """
    Core coordinator that orchestrates the AutoML workflow
    """
    
    def __init__(self, n_trials=50, timeout=None, cv=3, 
                 use_adaptive_optimization=True, use_dataset_optimization=True,
                 use_caching=True, show_progress=True, enable_advanced_features=True):
        """Initialize coordinator with configuration"""
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.use_adaptive_optimization = use_adaptive_optimization
        self.use_dataset_optimization = use_dataset_optimization
        self.use_caching = use_caching
        self.show_progress = show_progress
        self.enable_advanced_features = enable_advanced_features

        # Initialize specialized components
        self.progress_tracker = create_progress_tracker(show_progress=show_progress) if show_progress else None
        self.data_preparation = DataPreparation(use_dataset_optimization=use_dataset_optimization)
        self.optimization_manager = OptimizationManager(
            n_trials=n_trials, timeout=timeout, cv=cv, 
            use_adaptive_optimization=use_adaptive_optimization
        )
        self.pipeline_builder = PipelineBuilder(use_caching=use_caching)
        self.prediction_handler = PredictionHandler()
        
        # Initialize advanced components
        if enable_advanced_features:
            # 🔥 PRODUCTION-GRADE: Safe initialization with fallbacks
            self.nas_engine = AdvancedNAS() if NAS_AVAILABLE and AdvancedNAS else None
            self.multimodal_engine = AdvancedMultimodalAutoML() if MULTIMODAL_AVAILABLE and AdvancedMultimodalAutoML else None
            self.distributed_engine = AdvancedDistributedAutoML() if DISTRIBUTED_AVAILABLE and AdvancedDistributedAutoML else None
            
            # Log what's actually available
            available_features = []
            if self.nas_engine:
                available_features.append("NAS")
            if self.multimodal_engine:
                available_features.append("Multimodal")
            if self.distributed_engine:
                available_features.append("Distributed")
            
            if available_features:
                self.logger.info(f"🚀 Advanced AutoML features enabled: {', '.join(available_features)}")
            else:
                self.logger.warning("⚠️ Advanced features requested but none available")
        else:
            self.nas_engine = None
            self.multimodal_engine = None
            self.distributed_engine = None

        # State tracking
        self.best_pipeline = None
        self.best_model_name = None
        self.task_type = None
        self.dataset_profile = None
        self.feature_metadata = None
        self.feature_engineering_transformer = None
        self.optimization_metadata = {}
        self.dataset_metadata = {}

    def run_automl_workflow(self, X, y):
        """
        Execute the complete AutoML workflow with advanced features
        Returns: fitted AutoML instance
        """
        try:
            # Start progress tracking
            if self.progress_tracker:
                self.progress_tracker.start_optimization(self.n_trials, "unknown")
            
            # Step 0: Advanced multimodal analysis (if enabled)
            if self.enable_advanced_features and self.multimodal_engine:
                self.logger.info("🌐 Running advanced multimodal analysis...")
                multimodal_analysis = self.multimodal_engine.analyze_multimodal_data(X, y)
                self.logger.info(f"🎯 Detected modalities: {list(multimodal_analysis['modalities'].keys())}")
            
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
            
            # Step 2: Advanced NAS for neural networks (if enabled and applicable)
            if (self.enable_advanced_features and self.nas_engine and 
                not disable_neural_networks and 'neural_network' in str(dataset_metadata)):
                self.logger.info("🧠 Running advanced Neural Architecture Search...")
                best_architecture = self.nas_engine.search_architecture(
                    X_processed, y_processed, task_type, max_trials=min(20, self.n_trials//2)
                )
                self.logger.info(f"🏆 Best NAS architecture found with {best_architecture['layers']} layers")
            
            # Step 3: Run optimization search (meta-learning + hyperparameter search)
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
            
            # Step 4: Build final pipeline with best parameters
            self.best_pipeline = self.pipeline_builder.build_final_pipeline(
                X_processed, y_processed, best_params, best_model_name,
                task_type, self.n_trials, optimization_metadata,
                dataset_metadata, data_type_results["primary_type"], feature_metadata
            )
            
            if self.progress_tracker:
                self.progress_tracker.finish_optimization()
            
            # Step 5: Advanced distributed optimization report (if enabled)
            if self.enable_advanced_features and self.distributed_engine:
                self.logger.info("☁️ Distributed optimization intelligence available")
                # Store distributed patterns for future use
                self.distributed_engine.learn_resource_performance(
                    task_type, {'trials': self.n_trials}, self.best_score
                )
            
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
