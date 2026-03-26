"""
Optimization logic for AutoML
"""

import logging
from optimizer.optuna_search import OptunaOptimizer
from optimizer.adaptive_optimizer import AdaptiveOptimizer
from models.registry import MODEL_REGISTRY
from optimizer.search_space import get_search_space
from meta_learning.knowledge_base import KnowledgeBase
from meta_learning.recommender import MetaRecommender
from tracking.logger import ExperimentLogger
from core.pipeline_builder import build_pipeline


class OptimizationManager:
    """Handles meta-learning and hyperparameter optimization"""
    
    def __init__(self, n_trials=50, timeout=None, cv=3, use_adaptive_optimization=True):
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.use_adaptive_optimization = use_adaptive_optimization
        self.logger = logging.getLogger(__name__)
    
    def _filter_model_params(self, model_name, params):
        """Filter parameters to only include those relevant to the specific model"""
        # Define which parameters belong to which models
        model_param_map = {
            # Ridge regression specific parameters
            "ridge": ["alpha", "solver"],
            
            # Lasso regression specific parameters  
            "lasso": ["alpha", "selection"],
            
            # Neural network specific parameters - simplified to avoid Optuna issues
            "neural_network": ["max_layers", "max_neurons", "activation", "solver", 
                              "alpha", "learning_rate_init", "max_iter", "early_stopping"],
            
            # Random forest parameters
            "random_forest": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"],
            
            # XGBoost parameters
            "xgboost": ["learning_rate", "n_estimators", "max_depth", "subsample", "colsample_bytree"],
            
            # SVM/SVR parameters
            "svm": ["C", "kernel", "gamma", "probability"],
            "svr": ["C", "kernel", "gamma"],
            
            # Logistic regression parameters
            "logistic_regression": ["C", "penalty", "solver", "l1_ratio"],
            
            # Gradient boosting parameters
            "gradient_boosting": ["n_estimators", "learning_rate", "max_depth", "subsample"],
            
            # LightGBM parameters
            "lightgbm": ["n_estimators", "learning_rate", "max_depth", "num_leaves", 
                        "feature_fraction", "bagging_fraction", "bagging_freq", 
                        "min_child_samples", "reg_alpha", "reg_lambda"],
            
            # KNN parameters
            "knn": ["n_neighbors", "weights", "metric"],
            
            # Decision tree parameters
            "decision_tree": ["max_depth", "min_samples_split", "min_samples_leaf", "criterion"],
        }
        
        # Find which model category this belongs to
        allowed_params = []
        for model_type, param_list in model_param_map.items():
            if model_type in model_name:
                allowed_params.extend(param_list)
                break
        
        # Filter the parameters
        filtered_params = {}
        for key, value in params.items():
            if key in ["scaler", "imputer", "feature_selection", "model"] or key in allowed_params:
                filtered_params[key] = value
        
        return filtered_params
    
    def run_optimization(self, X, y, task_type, dataset_profile, dataset_metadata, 
                        feature_metadata, data_type_results, progress_tracker=None,
                        disable_neural_networks=False):
        """
        Run hyperparameter optimization with meta-learning integration
        
        Args:
            X: Features
            y: Target
            task_type: Type of ML task (classification/regression)
            dataset_profile: Dataset characteristics
            dataset_metadata: Additional dataset info
            feature_metadata: Feature analysis results
            data_type_results: Data type detection results
            progress_tracker: Progress tracking instance
            disable_neural_networks: Whether to disable neural network models
            
        Returns:
            Tuple of (best_params, best_model_name, optimization_metadata)
        """
        self.logger.info(f"DEBUG: Starting run_optimization for task_type: {task_type}")
        
        # Initialize experiment logger
        exp_logger = ExperimentLogger()
        exp_logger.log_params({
            "dataset_profile": dataset_profile,
            "dataset_metadata": dataset_metadata,
            "data_type_results": data_type_results,
            "feature_metadata": feature_metadata
        })
        
        # Meta-learning (with robust error handling)
        try:
            kb = KnowledgeBase()
            recommender = MetaRecommender(kb)
            recommendations = recommender.recommend(dataset_profile)
            
            # Defensive check for None recommendations
            if recommendations is None:
                self.logger.warning("Meta-learning recommender returned None, using empty recommendations")
                recommendations = []
            
            # Check if recommendations is empty
            if not recommendations:
                self.logger.info("No meta-learning recommendations available")
                filtered_recommendations = []
            else:
                # Filter recommendations by task type
                filtered_recommendations = []
                for rec in recommendations:
                    rec_task_type = rec.get('metrics', {}).get('task_type', 'unknown')
                    if rec_task_type == task_type or rec_task_type == 'unknown':
                        filtered_recommendations.append(rec)
            
            self.logger.info("Meta-Learning Recommendations:")
            for rec in filtered_recommendations:
                self.logger.info(f"  {rec.get('model', 'unknown')}: {rec.get('metrics', {})}")
            
            # Defensive check for None preprocessing hints
            preprocessing_hints = recommender.get_preprocessing_hints(dataset_profile)
            if preprocessing_hints is None:
                self.logger.warning("Meta-learning preprocessing hints returned None, using defaults")
                preprocessing_hints = {'imputer': ['mean'], 'scaler': ['standard']}
            elif not preprocessing_hints:
                self.logger.info("Preprocessing hints empty, using defaults")
                preprocessing_hints = {'imputer': ['mean'], 'scaler': ['standard']}
            else:
                self.logger.info(f"Preprocessing hints: {preprocessing_hints}")
            
        except Exception as e:
            self.logger.warning(f"Meta-learning failed: {e}")
            # Set default values if meta-learning fails
            filtered_recommendations = []
            preprocessing_hints = {'imputer': ['mean'], 'scaler': ['standard']}
        
        # Initialize top_trials and optimization_metadata to handle all cases
        top_trials = None
        optimization_metadata = {}
        
        self.logger.info("DEBUG: About to start hyperparameter optimization")
        
        # Hyperparameter optimization
        if self.use_adaptive_optimization:
            self.logger.info("DEBUG: Using adaptive optimization")
            
            # Adaptive configuration based on dataset size
            dataset_size = len(X) if hasattr(X, '__len__') else 0
            is_large_dataset = dataset_size > 10000
            
            if is_large_dataset:
                # Reduced trials for large datasets
                initial_trials = min(10, self.n_trials)
                max_trials = min(25, self.n_trials)
                cv_folds = min(3, self.cv)
                self.logger.info(f"Large dataset detected ({dataset_size} samples). Using reduced optimization: {max_trials} trials, {cv_folds} CV folds")
            else:
                initial_trials = min(20, self.n_trials)
                max_trials = self.n_trials
                cv_folds = self.cv
            
            optimizer = AdaptiveOptimizer(
                initial_trials=initial_trials,
                max_trials=max_trials,
                time_budget=self.timeout,
                cv_folds=cv_folds,
                large_dataset_threshold=10000,
                sample_ratio=0.3
            )
            
            _task_type = task_type
            
            def objective_func(trial, X, y, task_type):
                # Pass preprocessing_hints as priors to get_search_space
                params = get_search_space(trial, task_type=task_type, priors=preprocessing_hints, disable_neural_networks=disable_neural_networks)
                model_name = params.pop("model")
                
                if model_name not in MODEL_REGISTRY:
                    return -1e6 if task_type == 'regression' else 0.0, model_name, params
                
                # CRITICAL FIX: Filter parameters by model type to prevent leakage
                filtered_params = self._filter_model_params(model_name, params)
                
                model_cls = MODEL_REGISTRY[model_name]
                model_params = {
                    k: v for k, v in filtered_params.items()
                    if k not in ["scaler", "imputer", "feature_selection", "model"]
                }
                
                # DEBUG: Log parameter filtering
                self.logger.debug(f"DEBUG: Model: {model_name}")
                self.logger.debug(f"DEBUG: Original params: {list(params.keys())}")
                self.logger.debug(f"DEBUG: Filtered params: {list(filtered_params.keys())}")
                self.logger.debug(f"DEBUG: Model params for model creation: {model_params}")
                
                # Additional debug for logistic regression
                if "logistic_regression" in model_name:
                    self.logger.warning(f"DEBUG: Logistic regression model_params: {model_params}")
                
                model = model_cls(**model_params)
                # CRITICAL FIX: Use filtered_params for pipeline, not original params
                pipeline = build_pipeline(filtered_params, model)
                
                from sklearn.model_selection import cross_val_score
                import numpy as np
                
                try:
                    scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
                    
                    # Adaptive CV based on dataset size
                    dataset_size = len(X) if hasattr(X, '__len__') else 0
                    if dataset_size > 10000:
                        cv_folds = 3  # Reduced CV for large datasets
                    else:
                        cv_folds = self.cv
                    
                    cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring=scoring)
                    
                    # Handle infinite/NaN scores
                    mean_score = cv_scores.mean()
                    std_score = cv_scores.std()
                    
                    if not np.isfinite(mean_score):
                        if task_type == 'regression':
                            # For regression, return a large negative number for bad scores
                            return -1e6, model_name, params
                        else:
                            # For classification, return 0 for bad scores
                            return 0.0, model_name, params
                    
                    # Log CV results for visibility
                    self.logger.info(f"CV mean score: {mean_score:.4f} ± {std_score:.4f} | Model: {model_name}")
                    
                    return mean_score, model_name, params
                    
                except Exception as e:
                    self.logger.warning(f"Trial evaluation failed: {e}")
                    if task_type == 'regression':
                        return -1e6, model_name, params  # Bad score for failed regression trials
                    else:
                        return 0.0, model_name, params   # Bad score for failed classification trials
        
        try:
            self.logger.info("DEBUG: About to call optimizer.optimize_adaptive")
            top_trials, optimization_metadata = optimizer.optimize_adaptive(
                objective_func, X, y, _task_type
            )
            self.logger.info(f"DEBUG: optimize_adaptive returned - top_trials: {len(top_trials) if top_trials else 'None'}")
            
            self.logger.info(f"Adaptive optimization completed: {len(optimizer.study.trials)} trials")
            self.logger.info(f"Best trial score: {optimizer.study.best_value:.4f}")
            
            if progress_tracker:
                for i, (score, model_name, params) in enumerate(top_trials):
                    progress_tracker.add_trial_result(
                        trial_number=i, model_name=model_name, score=score, params=params
                    )
            
        except Exception as e:
            self.logger.error(f"Adaptive optimization failed: {e}")
            top_trials = None  # Explicitly set to None on failure
            optimization_metadata = {}  # Ensure metadata is also set
        
        if top_trials is None or not top_trials or len(top_trials) == 0:
            # Fallback to original optimizer
            self.logger.info("Adaptive optimization failed, falling back to standard optimizer")
            optimizer = OptunaOptimizer(n_trials=self.n_trials, cv=self.cv, task_type=task_type)
            
            try:
                top_trials = optimizer.optimize(X, y)
                optimization_metadata = {}
                self.logger.info(f"Optimization completed: {len(optimizer.study.trials)} trials")
                self.logger.info(f"Best trial score: {optimizer.study.best_trial.value:.4f}")
                
            except Exception as e:
                self.logger.error(f"Optimization failed: {e}")
                raise RuntimeError(f"Hyperparameter optimization failed: {e}")
        
        # Final guard against no successful trials
        if not top_trials or len(top_trials) == 0:
            raise RuntimeError("No successful trials found during optimization")
        
        # Return the best parameters and model name
        best_score, best_model_name, best_params = top_trials[0]
        
        self.logger.info(f"DEBUG: Returning from run_optimization - best_model: {best_model_name}, params: {best_params}")
        
        return best_params, best_model_name, optimization_metadata
