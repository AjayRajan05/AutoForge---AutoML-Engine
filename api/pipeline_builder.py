"""
Pipeline construction logic for AutoML
"""

import logging
from models.registry import MODEL_REGISTRY
from core.pipeline_builder import build_pipeline
from core.pipeline_cache import PipelineCache, CachedPipelineBuilder
from tracking.logger import ExperimentLogger


# Define CachedTransformer at module level for picklability
class CachedTransformer:
    """Custom transformer that wraps the cached builder for sklearn compatibility"""
    def __init__(self, cached_builder, pipeline_config):
        self.cached_builder = cached_builder
        self.pipeline_config = pipeline_config
        self.components = None
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X, y=None):
        _, self.components = self.cached_builder.build_cached_pipeline(
            X, self.pipeline_config
        )
        self.is_fitted = True
        return self
        

    def transform(self, X):
        if not self.is_fitted or self.components is None:
            # Fallback to regular pipeline building - create a simple preprocessing pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline as SKPipeline
            steps = []
            
            # Safely extract pipeline configuration
            steps_config = self.pipeline_config.get("steps", [])
            
            # Add imputer step
            if steps_config and "imputer" in steps_config[0].get("type", ""):
                imputer_type = steps_config[0]["type"].split("_")[-1]
                if imputer_type != "none":
                    steps.append(("imputer", SimpleImputer(strategy=imputer_type)))
            
            # Add scaler step
            if len(steps_config) > 1 and "scaler" in steps_config[1].get("type", ""):
                scaler_type = steps_config[1]["type"].split("_")[-1]
                if scaler_type != "none":
                    steps.append(("scaler", StandardScaler()))
            
            # Create and fit the pipeline
            if steps:
                pipeline = SKPipeline(steps)
                try:
                    pipeline.fit(X)
                    return pipeline.transform(X)
                except Exception as e:
                    self.logger.warning(f"Preprocessing pipeline failed: {e}")
                    return X  # Return raw data if preprocessing fails
            else:
                return X  # No preprocessing steps, return raw data
                
        try:
            return self.cached_builder.transform_with_cache(
                X, self.components, self.pipeline_config
            )
        except Exception as e:
            self.logger.warning(f"Cached transform failed: {e}")
            return X  # Return raw data if cached transform fails


class PipelineBuilder:
    """Handles construction and fitting of the final pipeline"""
    
    def __init__(self, use_caching=True):
        self.use_caching = use_caching
        self.pipeline_cache = PipelineCache() if use_caching else None
        self.logger = logging.getLogger(__name__)
    
    def build_final_pipeline(self, X, y, best_params, best_model_name, 
                            task_type, n_trials, optimization_metadata, 
                            dataset_metadata, data_type, feature_metadata):
        """
        Construct and fit the final pipeline with best parameters
        Returns: best_pipeline
        """
        try:
            model_cls = MODEL_REGISTRY[best_model_name]
            model_params = {
                k: v for k, v in best_params.items()
                if k not in ["scaler", "imputer", "feature_selection", "model"]
            }
            model = model_cls(**model_params)
            
            # Add dataset info to params
            best_params["n_features"] = X.shape[1]
            
            if self.pipeline_cache:
                cached_builder = CachedPipelineBuilder(self.pipeline_cache)
                pipeline_config = {
                    "steps": [
                        {"type": f"scaler_{best_params.get('scaler', 'standard')}", "params": {}},
                        {"type": f"imputer_{best_params.get('imputer', 'mean')}", "params": {}}
                    ]
                }
                
                if best_params.get("feature_selection", False):
                    pipeline_config["steps"].append({
                        "type": "selector_kbest",
                        "params": {"k": min(10, X.shape[1])}
                    })
                
                X_transformed, components = cached_builder.build_cached_pipeline(X, y, pipeline_config)
                model.fit(X_transformed, y)
                
                from sklearn.pipeline import Pipeline as SKPipeline
                best_pipeline = SKPipeline([
                    ("preprocessing", CachedTransformer(cached_builder, pipeline_config)),
                    ("model", model)
                ])
            
            else:
                pipeline = build_pipeline(best_params, model)
                pipeline.fit(X, y)
                best_pipeline = pipeline
            
            self.logger.info("Best pipeline trained successfully")
            
            # Final logging
            exp_logger = ExperimentLogger()
            exp_logger.log_params({
                "task_type": task_type,
                "best_model": best_model_name,
                "best_score": best_params.get("score", "unknown"),
                "n_trials": n_trials,
                "optimization_metadata": optimization_metadata,
                "dataset_metadata": dataset_metadata,
                "data_type": data_type,
                "feature_metadata": feature_metadata
            })
            exp_logger.save()
            
            return best_pipeline
            
        except Exception as e:
            self.logger.error(f"Pipeline construction failed: {e}")
            raise RuntimeError(f"Failed to build best pipeline: {e}")
