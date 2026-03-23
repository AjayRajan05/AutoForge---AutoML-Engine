"""
Explainability and visualization functionality for AutoML
"""

import logging
from typing import Dict, Any, Optional

from explainability.model_explainability import ModelExplainability


# SHAP availability check
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ExplainabilityManager:
    """
    Handles model explanations and visualizations
    """
    
    def __init__(self, use_explainability=True):
        self.use_explainability = use_explainability
        self.explainer = ModelExplainability() if use_explainability else None
        self.explanations = {}
        self.logger = logging.getLogger(__name__)

    def explain_model(self, X, y, best_pipeline, task_type, use_shap=True, top_n=10):
        """
        Generate model explanations
        """
        if best_pipeline is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if not self.use_explainability:
            raise ValueError(
                "Explainability not enabled. Set use_explainability=True when initialising AutoML."
            )

        try:
            self.logger.info("Generating model explanations...")

            # Extract the underlying estimator from the pipeline
            model = self._extract_model_from_pipeline(best_pipeline)

            explanations = self.explainer.explain_model(model, X, y, task_type)
            self.explanations = explanations

            # Log top features
            top_features = (
                explanations.get("interpretability_report", {})
                             .get("top_features", {})
                             .get("top_5", [])
            )
            if top_features:
                self.logger.info("Top 5 Most Important Features:")
                for i, (feature, importance) in enumerate(top_features, 1):
                    self.logger.info(f"  {i}. {feature} → {importance:.4f}")

            return explanations

        except Exception as e:
            self.logger.error(f"Explainability analysis failed: {e}")
            raise RuntimeError(f"Failed to generate explanations: {e}")

    def _extract_model_from_pipeline(self, pipeline):
        """Extract the underlying model from sklearn pipeline"""
        if hasattr(pipeline, 'named_steps'):
            model = pipeline.named_steps.get('model')
            if model is None:
                model = (
                    pipeline.named_steps.get('classifier')
                    or pipeline.named_steps.get('regressor')
                )
        else:
            model = pipeline

        return model if model is not None else pipeline

    def get_feature_importance(self, method="aggregated"):
        """Get feature importance."""
        if not self.explanations:
            raise ValueError("No explanations available. Call explain() first.")

        feature_importance = self.explanations.get("feature_importance", {})

        if method not in feature_importance:
            available_methods = list(feature_importance.keys())
            if available_methods:
                fallback_method = available_methods[0]
                self.logger.warning(f"Method '{method}' not available, using '{fallback_method}'")
                method = fallback_method
            else:
                raise ValueError("No feature importance data available")

        return feature_importance[method]

    def get_explanation_summary(self):
        """Get human-readable explanation summary"""
        if not self.explanations:
            return "No explanations available. Call explain() first."
        return self.explainer.get_explanation_summary()

    def plot_feature_importance(self, top_n=10, save_path=None, method="aggregated"):
        """Plot feature importance."""
        if not self.explanations:
            raise ValueError("No explanations available. Call explain() first.")
        self.explainer.plot_feature_importance(top_n, save_path, method)

    def plot_shap_summary(self, save_path=None, max_display=20):
        """Plot SHAP summary plot."""
        if not self.explanations:
            raise ValueError("No explanations available. Call explain() first.")
        self.explainer.plot_shap_summary(save_path, max_display)

    def get_top_features(self, n=10, method="aggregated"):
        """Get top N features by importance."""
        importance = self.get_feature_importance(method)
        return list(importance.items())[:n]

    def explain_predictions(self, X, best_pipeline, n_instances=5):
        """Explain individual predictions."""
        if not self.explanations or not SHAP_AVAILABLE:
            return {"error": "SHAP explanations not available"}

        try:
            shap_explanations = self.explanations.get("shap_explanations", {})
            if "values" not in shap_explanations:
                return {"error": "SHAP values not available"}

            # Sample instances
            if hasattr(X, 'head'):
                X_sample = X.head(n_instances)
            else:
                X_sample = X[:n_instances]

            shap_values = shap_explanations["values"]
            feature_names = self.explanations.get("feature_names", [])

            if isinstance(shap_values, list):
                # Multi-class: average across classes
                shap_values = [sv[:n_instances] for sv in shap_values]
                shap_values = sum(shap_values) / len(shap_values)
            else:
                shap_values = shap_values[:n_instances]

            instance_explanations = {}

            for i in range(min(n_instances, len(X_sample))):
                instance_shap = shap_values[i]

                if feature_names:
                    feature_contributions = list(zip(feature_names, instance_shap))
                    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

                    # Get prediction for this instance
                    if hasattr(X_sample, 'iloc'):
                        instance_data = X_sample.iloc[i].values.reshape(1, -1)
                    else:
                        instance_data = X_sample[i].reshape(1, -1)
                    
                    prediction = best_pipeline.predict(instance_data)[0]

                    instance_explanations[f"instance_{i + 1}"] = {
                        "features": feature_contributions[:10],
                        "prediction": prediction,
                        "shap_values": instance_shap.tolist()
                    }

            return instance_explanations

        except Exception as e:
            self.logger.error(f"Individual prediction explanations failed: {e}")
            return {"error": str(e)}
