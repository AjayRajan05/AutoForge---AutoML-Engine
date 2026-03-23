"""
AutoML - Ultra-minimal coordinator that delegates to specialized modules
"""

import logging
from typing import Dict, Any, Optional

# Import core components
from .core import AutoMLCoordinator, ExplainabilityManager, MetaLearningManager


class AutoML:
    """
    Ultra-minimal AutoML coordinator - only 50 lines!
    Delegates all complex logic to specialized modules.
    """
    
    def __init__(self,
                 n_trials=50,
                 timeout=None,
                 cv=3,
                 use_adaptive_optimization=True,
                 use_dataset_optimization=True,
                 use_caching=True,
                 show_progress=True,
                 use_explainability=True):
        """Initialize AutoML with configuration"""
        # Core coordinator handles the main workflow
        self.coordinator = AutoMLCoordinator(
            n_trials=n_trials, timeout=timeout, cv=cv,
            use_adaptive_optimization=use_adaptive_optimization,
            use_dataset_optimization=use_dataset_optimization,
            use_caching=use_caching, show_progress=show_progress
        )
        
        # Explainability manager handles all explanation functionality
        self.explainability_manager = ExplainabilityManager(use_explainability=use_explainability)
        
        # Meta-learning manager handles pattern learning
        self.meta_learning_manager = MetaLearningManager()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_dataset_info(self, X):
        """Get basic dataset information"""
        if hasattr(X, 'shape'):
            return {
                "num_rows": X.shape[0],
                "num_cols": X.shape[1],
                "has_missing": hasattr(X, 'isnull') and X.isnull().sum().sum() > 0 if hasattr(X, 'isnull') else False,
            }
        return {}

    def fit(self, X, y):
        """
        Ultra-minimal fit - delegates everything to coordinator
        """
        self.coordinator.run_automl_workflow(X, y)
        return self

    def predict(self, X):
        """Make predictions - delegates to coordinator"""
        return self.coordinator.predict(X)

    def predict_proba(self, X):
        """Make probability predictions - delegates to coordinator"""
        return self.coordinator.predict_proba(X)

    # Explainability methods - delegate to explainability manager
    def explain(self, X, y=None, use_shap=True, top_n=10):
        """Generate model explanations."""
        return self.explainability_manager.explain_model(
            X, y, self.coordinator.best_pipeline, 
            self.coordinator.task_type, use_shap, top_n
        )

    def get_feature_importance(self, method="aggregated"):
        """Get feature importance."""
        return self.explainability_manager.get_feature_importance(method)

    def get_explanation_summary(self):
        """Get human-readable explanation summary."""
        return self.explainability_manager.get_explanation_summary()

    def plot_feature_importance(self, top_n=10, save_path=None, method="aggregated"):
        """Plot feature importance."""
        return self.explainability_manager.plot_feature_importance(top_n, save_path, method)

    def plot_shap_summary(self, save_path=None, max_display=20):
        """Plot SHAP summary plot."""
        return self.explainability_manager.plot_shap_summary(save_path, max_display)

    def get_top_features(self, n=10, method="aggregated"):
        """Get top N features by importance."""
        return self.explainability_manager.get_top_features(n, method)

    def explain_predictions(self, X, n_instances=5):
        """Explain individual predictions."""
        return self.explainability_manager.explain_predictions(
            X, self.coordinator.best_pipeline, n_instances
        )

    # Meta-learning methods - delegate to meta-learning manager
    def learn_from_experiment(self, experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Learn patterns from completed experiment."""
        return self.meta_learning_manager.learn_from_experiment(experiment_result)

    # Properties to access coordinator state
    @property
    def best_pipeline(self):
        """Access the trained pipeline"""
        return self.coordinator.best_pipeline

    @property
    def best_score(self):
        """Get the best score achieved"""
        return self.coordinator.best_score

    @property
    def best_model_name(self):
        """Get the name of the best model"""
        return self.coordinator.best_model_name

    @property
    def task_type(self):
        """Get the detected task type"""
        return self.coordinator.task_type

    @property
    def feature_metadata(self):
        """Get feature analysis results"""
        return self.coordinator.feature_metadata

    @property
    def dataset_profile(self):
        """Get dataset characteristics"""
        return self.coordinator.dataset_profile

    @property
    def optimization_metadata(self):
        """Get optimization metadata"""
        return self.coordinator.optimization_metadata

    @property
    def n_trials(self):
        """Get number of trials used"""
        return self.coordinator.n_trials