import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import clone


class AdvancedStacker:
    def __init__(self, base_models, meta_model=None, n_folds=5, task_type="classification"):
        self.base_models = base_models
        self.task_type = task_type
        self.n_folds = n_folds
        
        # Choose appropriate meta-model
        if meta_model is None:
            if task_type == "classification":
                self.meta_model = LogisticRegression(max_iter=1000)
            else:
                self.meta_model = Ridge()
        else:
            self.meta_model = meta_model

    def fit(self, X, y):
        """Enhanced stacking with model performance tracking"""
        self.base_models_ = [list() for _ in self.base_models]
        self.model_scores = []
        
        # Generate meta-features using out-of-fold predictions
        self.meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for i, model in enumerate(self.base_models):
            fold_models = []
            fold_scores = []
            
            for train_idx, val_idx in kf.split(X):
                # Clone model to avoid contamination
                model_clone = clone(model)
                
                # Train on training fold
                X_train_fold, y_train_fold = X[train_idx], y[train_idx]
                model_clone.fit(X_train_fold, y_train_fold)
                
                # Predict on validation fold
                X_val_fold = X[val_idx]
                if self.task_type == "classification":
                    preds = model_clone.predict_proba(X_val_fold)[:, 1]  # Use probabilities
                else:
                    preds = model_clone.predict(X_val_fold)
                
                self.meta_features[val_idx, i] = preds
                fold_models.append(model_clone)
                
                # Calculate fold score
                y_val_fold = y[val_idx]
                if self.task_type == "classification":
                    fold_score = accuracy_score(y_val_fold, model_clone.predict(X_val_fold))
                else:
                    fold_score = -mean_squared_error(y_val_fold, model_clone.predict(X_val_fold))
                
                fold_scores.append(fold_score)
            
            self.base_models_[i] = fold_models
            self.model_scores.append(np.mean(fold_scores))

        # Train meta-model on meta-features
        self.meta_model.fit(self.meta_features, y)
        
        # Calculate model importance based on performance
        self.model_weights = np.array(self.model_scores)
        if self.task_type == "classification":
            # For classification, higher is better
            self.model_weights = self.model_weights / np.sum(self.model_weights)
        else:
            # For regression, less negative (closer to 0) is better
            self.model_weights = -self.model_weights
            self.model_weights = self.model_weights / np.sum(self.model_weights)

    def predict(self, X):
        """Predict using weighted ensemble of base models"""
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, models in enumerate(self.base_models_):
            # Average predictions from all folds
            model_preds = []
            for model in models:
                if self.task_type == "classification":
                    preds = model.predict_proba(X)[:, 1]
                else:
                    preds = model.predict(X)
                model_preds.append(preds)
            
            meta_features[:, i] = np.mean(model_preds, axis=0)

        return self.meta_model.predict(meta_features)

    def predict_proba(self, X):
        """Predict probabilities (classification only)"""
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")
        
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, models in enumerate(self.base_models_):
            model_preds = []
            for model in models:
                preds = model.predict_proba(X)[:, 1]
                model_preds.append(preds)
            
            meta_features[:, i] = np.mean(model_preds, axis=0)

        return self.meta_model.predict_proba(meta_features)

    def get_feature_importance(self):
        """Get model importance from meta-model"""
        if hasattr(self.meta_model, 'coef_'):
            return self.meta_model.coef_
        elif hasattr(self.meta_model, 'feature_importances_'):
            return self.meta_model.feature_importances_
        else:
            return self.model_weights


class AdaptiveStacker(AdvancedStacker):
    """Stacker that adapts model selection based on cross-validation performance"""
    
    def __init__(self, base_models, meta_model=None, n_folds=5, task_type="classification", 
                 max_models=5, performance_threshold=0.6):
        # Filter models based on initial performance
        self.max_models = max_models
        self.performance_threshold = performance_threshold
        
        # Evaluate base models quickly
        selected_models = self._select_best_models(base_models, X_sample=None, y_sample=None)
        
        super().__init__(selected_models, meta_model, n_folds, task_type)
    
    def _select_best_models(self, base_models, X_sample=None, y_sample=None):
        """Quick evaluation to select best performing models"""
        if X_sample is None or y_sample is None:
            # If no sample provided, return all models
            return base_models[:self.max_models]
        
        model_scores = []
        for model in base_models:
            try:
                # Quick cross-validation
                scores = cross_val_score(model, X_sample, y_sample, cv=3, scoring='accuracy')
                mean_score = np.mean(scores)
                model_scores.append((mean_score, model))
            except:
                model_scores.append((0.0, model))  # Failed models get 0 score
        
        # Sort by performance and select top models
        model_scores.sort(key=lambda x: x[0], reverse=True)
        
        selected = []
        for score, model in model_scores:
            if len(selected) >= self.max_models:
                break
            if score >= self.performance_threshold:
                selected.append(model)
        
        # Ensure we have at least one model
        if not selected:
            selected = [model_scores[0][1]]
        
        return selected


# Maintain backward compatibility
class Stacker(AdvancedStacker):
    """Backward compatible Stacker class"""
    pass