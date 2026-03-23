import optuna
import logging
from sklearn.model_selection import cross_val_score
from models.registry import MODEL_REGISTRY
from core.search_space import get_search_space
from tracking.logger import ExperimentLogger
from core.pipeline_builder import build_pipeline
from meta_learning.self_improver import SelfImprover


class OptunaOptimizer:
    def __init__(self, n_trials=50, cv=3, task_type="classification"):
        self.n_trials = n_trials
        self.cv = cv
        self.task_type = task_type
        self.study = None
        self.logger = logging.getLogger(__name__)

    def optimize(self, X, y):
        """
        Optimize hyperparameters with proper error handling and task type support
        """
        try:
            # Get priors from self-improver (with error handling)
            priors = None
            try:
                improver = SelfImprover()
                improver.analyze()
                priors = improver.get_best()
                self.logger.info(f"Using priors from self-improver: {priors}")
            except Exception as e:
                self.logger.warning(f"Failed to load priors: {e}")
                priors = None

            trials = []
            logger = ExperimentLogger()

            def objective(trial):
                try:
                    # Get search space with task type and priors
                    params = get_search_space(trial, task_type=self.task_type, priors=priors)
                    
                    # Extract model name and parameters
                    model_name = params.pop("model")
                    
                    # Validate model exists in registry
                    if model_name not in MODEL_REGISTRY:
                        self.logger.warning(f"Unknown model: {model_name}")
                        return 0.0
                    
                    model_cls = MODEL_REGISTRY[model_name]
                    
                    # Filter model parameters (remove preprocessing params)
                    model_params = {
                        k: v for k, v in params.items()
                        if k not in ["scaler", "imputer", "feature_selection", "model"]
                    }
                    
                    # Create and train model
                    model = model_cls(**model_params)
                    
                    # Build pipeline
                    pipeline = build_pipeline(params, model)
                    
                    # Evaluate with cross-validation
                    cv_scores = cross_val_score(
                        pipeline, X, y, 
                        cv=self.cv, 
                        scoring='accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
                    )
                    score = cv_scores.mean()
                    
                    # Store trial information
                    trials.append((score, model_name, params.copy()))
                    
                    # Log experiment
                    logger.log_model(model_name)
                    logger.log_params(params)
                    logger.log_metrics({
                        "cv_score": score,
                        "cv_std": cv_scores.std(),
                        "task_type": self.task_type
                    })
                    logger.save()
                    
                    self.logger.debug(f"Trial {len(trials)}: {model_name} = {score:.4f}")
                    
                    return score
                    
                except Exception as e:
                    self.logger.warning(f"Trial failed: {e}")
                    return 0.0  # Return worst possible score for failed trials

            # Create and run study
            self.study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            
            self.study.optimize(objective, n_trials=self.n_trials)
            
            # Sort trials by score (descending)
            trials.sort(key=lambda x: -x[0])
            
            self.logger.info(f"Optimization completed. Best score: {self.study.best_value:.4f}")
            self.logger.info(f"Total trials: {len(self.study.trials)}")
            
            return trials[:3]  # Return top 3 trials
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise RuntimeError(f"Hyperparameter optimization failed: {e}")
    
    def get_best_params(self):
        """Get best parameters from the study"""
        if self.study is None:
            raise ValueError("Study not created yet. Call optimize() first.")
        
        return self.study.best_params
    
    def get_optimization_history(self):
        """Get optimization history for plotting"""
        if self.study is None:
            raise ValueError("Study not created yet. Call optimize() first.")
        
        return self.study.trials_dataframe()