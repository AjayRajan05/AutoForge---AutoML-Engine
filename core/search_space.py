def get_search_space(trial, task_type="classification", priors=None, disable_neural_networks=False):
    """
    Enhanced search space with support for multiple models and task types
    """
    
    # Model options based on task type
    if task_type == "classification":
        model_options = [
            "classification_random_forest",
            "classification_logistic_regression", 
            "classification_xgboost",
            "classification_svm",
            "classification_knn",
            "classification_decision_tree",
            "classification_naive_bayes",
            "classification_gradient_boosting",
            "classification_lightgbm",
        ]
        if not disable_neural_networks:
            model_options.append("classification_neural_network")
    else:  # regression
        model_options = [
            "regression_random_forest_regressor",
            "regression_linear_regression",
            "regression_ridge",
            "regression_lasso", 
            "regression_svr",
            "regression_xgboost_regressor",
            "regression_gradient_boosting_regressor",
            "regression_lightgbm_regressor",
        ]
        if not disable_neural_networks:
            model_options.append("regression_neural_network_regressor")
    
    # Apply priors if available and not empty
    if priors and "model_priority" in priors and priors["model_priority"]:
        # Filter model options to only include those in priors that are valid for this task
        valid_priors = [model for model in priors["model_priority"] if model in model_options]
        if valid_priors:
            model_options = valid_priors
    
    # Ensure we always have valid options
    if not model_options:
        if task_type == "classification":
            model_options = ["classification_random_forest", "classification_logistic_regression"]
        else:
            model_options = ["regression_random_forest_regressor", "regression_linear_regression"]
    
    # Ensure scaler and imputer options are not empty and not None
    if not priors or "scaler_priority" not in priors or not priors["scaler_priority"]:
        scaler_options = ["standard", "minmax", "none"]
    else:
        scaler_options = priors["scaler_priority"]
        if scaler_options is None:
            scaler_options = ["standard", "minmax", "none"]
    
    if not priors or "imputer_priority" not in priors or not priors["imputer_priority"]:
        imputer_options = ["mean", "median", "most_frequent"]
    else:
        imputer_options = priors["imputer_priority"]
        if imputer_options is None:
            imputer_options = ["mean", "median", "most_frequent"]

    model_name = trial.suggest_categorical("model", model_options)
    scaler = trial.suggest_categorical("scaler", scaler_options)
    imputer = trial.suggest_categorical("imputer", imputer_options)

    use_feature_selection = trial.suggest_categorical("feature_selection", [True, False])

    params = {
        "model": model_name,
        "scaler": scaler,
        "imputer": imputer,
        "feature_selection": use_feature_selection
    }

    # Model-specific hyperparameters - ONLY suggest parameters relevant to the selected model
    if "random_forest" in model_name:
        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        })
    
    elif "logistic_regression" in model_name:
        params.update({
            "C": trial.suggest_float("C", 0.001, 100.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
            "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
        })
    
    elif "xgboost" in model_name:
        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        })
    
    elif "svm" in model_name or "svr" in model_name:
        params.update({
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        })
        if "svm" in model_name:  # Classification specific
            params["probability"] = True
    
    elif "knn" in model_name:
        params.update({
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
        })
    
    elif "decision_tree" in model_name:
        params.update({
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        })
    
    elif "naive_bayes" in model_name:
        # GaussianNB has no hyperparameters to tune
        pass
    
    elif "linear_regression" in model_name:
        # LinearRegression has no hyperparameters to tune
        pass
    
    elif "ridge" in model_name:
        params.update({
            "alpha": trial.suggest_float("alpha", 0.1, 10.0, log=True),
            "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky"]),
        })
    
    elif "lasso" in model_name:
        params.update({
            "alpha": trial.suggest_float("alpha", 0.001, 1.0, log=True),
            "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
        })
    
    elif "gradient_boosting" in model_name:
        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        })

    elif "lightgbm" in model_name:
        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        })
    
    elif "neural_network" in model_name:
        params.update({
            "max_layers": trial.suggest_int("max_layers", 1, 4),
            "max_neurons": trial.suggest_int("max_neurons", 32, 256),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
            "solver": trial.suggest_categorical("solver", ["adam", "sgd"]),
            "alpha": trial.suggest_float("alpha", 0.0001, 0.1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 0.001, 0.1, log=True),
            "max_iter": trial.suggest_int("max_iter", 500, 2000),
            "early_stopping": trial.suggest_categorical("early_stopping", [True, False]),
        })

    return params