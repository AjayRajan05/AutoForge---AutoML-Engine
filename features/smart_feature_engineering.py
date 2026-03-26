"""
Smart Feature Engineering Engine
Intelligent feature generation and selection
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from itertools import combinations
import warnings

logger = logging.getLogger(__name__)


class SmartFeatureEngineering:
    """
    Intelligent feature engineering with automatic feature generation and selection
    """
    
    def __init__(self,
                 max_polynomial_degree: int = 2,
                 max_interaction_features: int = 50,
                 feature_selection_ratio: float = 0.5,
                 min_feature_importance: float = 0.01,
                 correlation_threshold: float = 0.95):
        """
        Initialize smart feature engineering
        
        Args:
            max_polynomial_degree: Maximum degree for polynomial features
            max_interaction_features: Maximum number of interaction features to create
            feature_selection_ratio: Ratio of features to keep after selection
            min_feature_importance: Minimum importance threshold for feature selection
            correlation_threshold: Correlation threshold for removing redundant features
        """
        self.max_polynomial_degree = max_polynomial_degree
        self.max_interaction_features = max_interaction_features
        self.feature_selection_ratio = feature_selection_ratio
        self.min_feature_importance = min_feature_importance
        self.correlation_threshold = correlation_threshold
        
        # Feature engineering metadata
        self.feature_metadata = {}
        self.original_features = []
        self.engineered_features = []
        self.selected_features = []
        
    def engineer_features(self, 
                         X: Union[np.ndarray, pd.DataFrame],
                         y: Union[np.ndarray, pd.Series],
                         task_type: str = "classification") -> Tuple[Union[np.ndarray, pd.DataFrame], Dict[str, Any]]:
        """
        Perform smart feature engineering
        
        Args:
            X: Input features
            y: Target variable
            task_type: Type of ML task
            
        Returns:
            Engineered features and metadata
        """
        logger.info("Starting smart feature engineering...")
        
        # Add constraints for large datasets to prevent excessive computation
        n_samples, n_features = X.shape
        
        # AGGRESSIVE CONSTRAINTS: Skip polynomial features for high-dimensional data
        if n_features > 15:  # Much more aggressive limit
            logger.warning(f"Too many features ({n_features}) for polynomial features. Skipping polynomial generation.")
            metadata = {
                'engineered_features': 0,
                'original_features': n_features,
                'total_features': n_features,
                'engineering_type': 'skipped_polynomial'
            }
            return X, metadata
        elif n_features > 8:  # Prevent polynomial explosion for medium-sized datasets
            logger.warning(f"Medium-sized dataset ({n_features} features). Limiting polynomial degree to prevent explosion.")
            self.max_polynomial_degree = 1  # Only linear features, no quadratic/cubic
        elif n_samples > 20000 or n_features > 200:
            logger.warning(f"Very large dataset detected ({n_samples} samples, {n_features} features). Skipping all feature engineering for performance.")
            metadata = {
                'engineered_features': 0,
                'original_features': n_features,
                'total_features': n_features,
                'engineering_type': 'skipped_large_dataset'
            }
            return X, metadata
        elif n_samples > 5000 or n_features > 100:
            logger.warning(f"Large dataset detected ({n_samples} samples, {n_features} features). Limiting feature engineering.")
            # Limit polynomial features for large datasets
            if n_features > 50:
                self.max_polynomial_degree = 1  # No polynomial features for very high dimensional data
            elif n_features > 20:
                self.max_polynomial_degree = 1  # No polynomial features for high dimensional data
            else:
                self.max_polynomial_degree = 1  # Conservative for large datasets
            
            # Limit interaction features
            self.max_interaction_features = min(10, n_features // 2)
            self.feature_selection_ratio = 0.5  # Less aggressive selection for medium datasets
        
        # Convert to DataFrame for easier handling
        if isinstance(X, np.ndarray):
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
            feature_names = list(X_df.columns)
        
        self.original_features = feature_names.copy()
        
        # Step 1: Analyze original features
        feature_analysis = self._analyze_features(X_df, y, task_type)
        
        # Step 2: Generate polynomial features (if beneficial)
        if self._should_create_polynomial_features(feature_analysis, X_df.shape[1]):
            X_df = self._create_polynomial_features(X_df, task_type)
        
        # Step 3: Create interaction features
        if X_df.shape[1] > 1:
            X_df = self._create_interaction_features(X_df, y, task_type)
        
        # Step 4: Remove highly correlated features
        X_df = self._remove_correlated_features(X_df)
        
        # Step 5: Intelligent feature selection
        X_df, selection_metadata = self._intelligent_feature_selection(X_df, y, task_type)
        
        # Step 6: Feature scoring and validation
        feature_scores = self._score_features(X_df, y, task_type)
        
        # Compile metadata
        engineering_metadata = {
            "original_features": len(self.original_features),
            "engineered_features": len(X_df.columns) - len(self.original_features),
            "total_features": len(X_df.columns),
            "feature_analysis": feature_analysis,
            "selection_metadata": selection_metadata,
            "feature_scores": feature_scores,
            "correlation_threshold": self.correlation_threshold,
            "feature_selection_ratio": self.feature_selection_ratio
        }
        
        logger.info(f"Feature engineering completed: {engineering_metadata['total_features']} features "
                   f"({engineering_metadata['engineered_features']} engineered)")
        
        # Return in original format
        if isinstance(X, np.ndarray):
            return X_df.values, engineering_metadata
        else:
            return X_df, engineering_metadata
    
    def _analyze_features(self, 
                         X: pd.DataFrame, 
                         y: Union[np.ndarray, pd.Series],
                         task_type: str) -> Dict[str, Any]:
        """
        Analyze original features to determine engineering strategy
        
        Args:
            X: Input features
            y: Target variable
            task_type: Type of ML task
            
        Returns:
            Feature analysis results
        """
        analysis = {
            "n_features": X.shape[1],
            "numeric_features_list": [],
            "categorical_features": [],
            "feature_importances": {},
            "feature_correlations": {},
            "polynomial_potential": False,
            "interaction_potential": False,
            "linear_model_scores": []
        }
        
        # Identify feature types
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                analysis["numeric_features_list"].append(col)
            else:
                analysis["categorical_features"].append(col)
        
        # Add numeric_features count for the _should_create_polynomial_features method
        analysis["numeric_features"] = len(analysis["numeric_features_list"])
        
        # Calculate feature importances using Random Forest
        if len(analysis["numeric_features_list"]) > 0:
            rf = RandomForestClassifier(n_estimators=50, random_state=42) if task_type == "classification" else RandomForestRegressor(n_estimators=50, random_state=42)
            
            try:
                rf.fit(X[analysis["numeric_features_list"]], y)
                importances = rf.feature_importances_
                analysis["feature_importances"] = dict(zip(analysis["numeric_features_list"], importances))
            except Exception as e:
                logger.warning(f"Failed to calculate feature importances: {e}")
        
        # Calculate correlations
        if len(analysis["numeric_features_list"]) > 1:
            corr_matrix = X[analysis["numeric_features_list"]].corr()
            analysis["feature_correlations"] = corr_matrix.to_dict()
            
            # Check for polynomial potential (non-linear relationships)
            analysis["polynomial_potential"] = self._check_polynomial_potential(X[analysis["numeric_features_list"]], y)
            
            # Check for interaction potential
            analysis["interaction_potential"] = self._check_interaction_potential(X[analysis["numeric_features_list"]], y)
            
            # Calculate linear model scores for polynomial feature decision
            try:
                linear_score = self._evaluate_linear_model(X[analysis["numeric_features_list"]], y, task_type)
                analysis["linear_model_scores"].append(linear_score)
            except Exception as e:
                logger.warning(f"Failed to calculate linear model score: {e}")
                analysis["linear_model_scores"].append(0.5)  # Default moderate score
        
        return analysis
    
    def _check_polynomial_potential(self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series]) -> bool:
        """
        Check if polynomial features might be beneficial
        
        Args:
            X: Numeric features
            y: Target variable
            
        Returns:
            Whether polynomial features should be created
        """
        if X.shape[1] < 2:
            return False
        
        # Simple heuristic: check if non-linear patterns exist
        try:
            # Fit linear model and check performance
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            lr = LinearRegression()
            lr.fit(X, y)
            linear_pred = lr.predict(X)
            linear_score = r2_score(y, linear_pred)
            
            # Fit polynomial degree 2 model and compare
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            
            lr_poly = LinearRegression()
            lr_poly.fit(X_poly, y)
            poly_pred = lr_poly.predict(X_poly)
            poly_score = r2_score(y, poly_pred)
            
            # If polynomial is significantly better, create polynomial features
            improvement = poly_score - linear_score
            return improvement > 0.05  # 5% improvement threshold
            
        except Exception:
            return False
    
    def _check_interaction_potential(self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series]) -> bool:
        """
        Check if interaction features might be beneficial
        
        Args:
            X: Numeric features
            y: Target variable
            
        Returns:
            Whether interaction features should be created
        """
        if X.shape[1] < 2:
            return False
        
        # Simple heuristic: check feature interactions
        try:
            # Test a few interactions
            n_features_to_test = min(5, X.shape[1])
            feature_cols = X.columns[:n_features_to_test]
            
            base_score = self._get_baseline_score(X[feature_cols], y)
            
            # Test interactions
            for i, j in combinations(feature_cols, 2):
                interaction = X[feature_cols[i]] * X[feature_cols[j]]
                X_test = X[feature_cols].copy()
                X_test[f"{i}_x_{j}"] = interaction
                
                interaction_score = self._get_baseline_score(X_test, y)
                
                # If interaction improves score, create interaction features
                if interaction_score > base_score + 0.02:  # 2% improvement
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _get_baseline_score(self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series]) -> float:
        """Get baseline score for feature evaluation"""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            
            if len(np.unique(y)) <= 20:  # Classification
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                scoring = 'accuracy'
            else:  # Regression
                model = RandomForestRegressor(n_estimators=10, random_state=42)
                scoring = 'r2'
            
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
            return np.mean(scores)
            
        except Exception:
            return 0.0
    
    def _evaluate_linear_model(self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series], task_type: str) -> float:
        """Evaluate linear model for feature scoring"""
        try:
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.model_selection import cross_val_score
            
            if task_type == "classification":
                model = LogisticRegression(max_iter=1000, random_state=42)
                scoring = 'accuracy'
            else:  # Regression
                model = LinearRegression()
                scoring = 'r2'
            
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
            return np.mean(scores)
            
        except Exception:
            return 0.0
    
    def _should_create_polynomial_features(self, feature_analysis: Dict[str, Any], n_features: int) -> bool:
        """
        Determine if polynomial features should be created
        
        Args:
            feature_analysis: Analysis of current features
            n_features: Number of current features
            
        Returns:
            Whether to create polynomial features
        """
        # RESPECT CONSTRAINTS: Check if polynomial degree was limited to 1 (no polynomial features)
        if self.max_polynomial_degree <= 1:
            return False
            
        # Don't create polynomial features if we already have many features
        if n_features > 50:
            return False
            
        # Check if we have enough numeric features
        numeric_features = feature_analysis.get('numeric_features', 0)
        if numeric_features < 2:
            return False
            
        # Check if features show non-linear relationships
        linear_scores = feature_analysis.get('linear_model_scores', [])
        if len(linear_scores) > 0:
            avg_linear_score = np.mean(linear_scores)
            # If linear models perform poorly, try polynomial features
            if avg_linear_score < 0.7:  # Threshold for poor linear performance
                return True
                
        # Default: create polynomial features for small to medium datasets
        return n_features <= 20
    
    def _create_polynomial_features(self, X: pd.DataFrame, task_type: str) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            X: Input features
            task_type: Type of ML task
            
        Returns:
            Features with polynomial features added
        """
        logger.info("Creating polynomial features...")
        
        # Only apply to numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return X
        
        # Aggressive constraints to prevent explosion of features
        n_samples, n_features = X.shape
        max_features_for_poly = 15  # Hard limit
        
        if len(numeric_cols) > max_features_for_poly:
            logger.warning(f"Too many numeric features ({len(numeric_cols)}) for polynomial features. Skipping polynomial generation.")
            return X
        
        # Calculate potential number of polynomial features
        if self.max_polynomial_degree >= 2:
            potential_features = len(numeric_cols) ** 2 + len(numeric_cols)
            if potential_features > 500:  # Hard limit on total features
                logger.warning(f"Polynomial features would create {potential_features} features. Limiting to degree 1.")
                self.max_polynomial_degree = 1
                return X
        
        try:
            poly = PolynomialFeatures(degree=self.max_polynomial_degree, include_bias=False)
            X_poly = poly.fit_transform(X[numeric_cols])
            
            # Create feature names
            poly_feature_names = poly.get_feature_names_out(numeric_cols)
            
            # Create DataFrame with polynomial features
            poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
            
            # Remove original features (they're included in polynomial features)
            poly_df = poly_df.drop(columns=numeric_cols, errors='ignore')
            
            # Combine with original non-numeric features
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                result_df = pd.concat([X[non_numeric_cols], poly_df], axis=1)
            else:
                result_df = poly_df
            
            self.engineered_features.extend(poly_feature_names)
            logger.info(f"Created {len(poly_feature_names)} polynomial features")
            
            return result_df
            
        except Exception as e:
            logger.warning(f"Failed to create polynomial features: {e}")
            return X
    
    def _create_interaction_features(self, 
                                   X: pd.DataFrame, 
                                   y: Union[np.ndarray, pd.Series],
                                   task_type: str) -> pd.DataFrame:
        """
        Create interaction features
        
        Args:
            X: Input features
            y: Target variable
            task_type: Type of ML task
            
        Returns:
            Features with interaction features added
        """
        logger.info("Creating interaction features...")
        
        # Only apply to numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return X
        
        # Skip interaction features for large datasets to prevent slowdown
        n_samples, n_features = X.shape
        if n_samples > 10000 or n_features > 50:
            logger.warning(f"Large dataset ({n_samples} samples, {n_features} features). Skipping interaction features.")
            return X
        
        X_result = X.copy()
        interactions_created = 0
        
        try:
            # Calculate feature importances to prioritize interactions
            feature_scores = {}
            if hasattr(self, '_last_feature_importances') and self._last_feature_importances:
                feature_scores = self._last_feature_importances
            else:
                # Calculate simple scores
                for col in numeric_cols:
                    scores = []
                    for _ in range(3):  # Multiple runs for stability
                        score = self._get_baseline_score(X[[col]], y)
                        scores.append(score)
                    feature_scores[col] = np.mean(scores)
            
            # Sort features by importance
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_features[:min(10, len(sorted_features))]]
            
            # Create interactions between top features
            for i, j in combinations(top_features, 2):
                if interactions_created >= self.max_interaction_features:
                    break
                
                # Create interaction feature
                interaction_name = f"{i}_x_{j}"
                X_result[interaction_name] = X[i] * X[j]
                
                # Test if interaction is beneficial
                interaction_score = self._get_baseline_score(X_result[list(X.columns) + [interaction_name]], y)
                baseline_score = self._get_baseline_score(X, y)
                
                if interaction_score > baseline_score + 0.01:  # 1% improvement
                    self.engineered_features.append(interaction_name)
                    interactions_created += 1
                else:
                    # Remove non-beneficial interaction
                    X_result.drop(columns=[interaction_name], inplace=True)
            
            logger.info(f"Created {interactions_created} beneficial interaction features")
            
        except Exception as e:
            logger.warning(f"Failed to create interaction features: {e}")
        
        return X_result
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features
        
        Args:
            X: Input features
            
        Returns:
            Features with redundant features removed
        """
        logger.info("Removing correlated features...")
        
        # Only apply to numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return X
        
        try:
            # Calculate correlation matrix
            corr_matrix = X[numeric_cols].corr().abs()
            
            # Find highly correlated pairs
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to remove
            to_remove = set()
            for i in range(len(upper_triangle.columns)):
                for j in range(len(upper_triangle.columns)):
                    if i < j and upper_triangle.iloc[i, j] > self.correlation_threshold:
                        # Remove the feature with lower importance (if available)
                        feature_i = upper_triangle.columns[i]
                        feature_j = upper_triangle.columns[j]
                        
                        if hasattr(self, '_last_feature_importances') and self._last_feature_importances:
                            imp_i = self._last_feature_importances.get(feature_i, 0)
                            imp_j = self._last_feature_importances.get(feature_j, 0)
                            
                            if imp_i < imp_j:
                                to_remove.add(feature_i)
                            else:
                                to_remove.add(feature_j)
                        else:
                            # Default: remove the second feature
                            to_remove.add(feature_j)
            
            # Remove correlated features
            X_result = X.drop(columns=list(to_remove))
            
            if len(to_remove) > 0:
                logger.info(f"Removed {len(to_remove)} highly correlated features")
            
            return X_result
            
        except Exception as e:
            logger.warning(f"Failed to remove correlated features: {e}")
            return X
    
    def _intelligent_feature_selection(self, 
                                      X: pd.DataFrame,
                                      y: Union[np.ndarray, pd.Series],
                                      task_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform intelligent feature selection
        
        Args:
            X: Input features
            y: Target variable
            task_type: Type of ML task
            
        Returns:
            Selected features and selection metadata
        """
        logger.info("Performing intelligent feature selection...")
        
        selection_metadata = {
            "original_features": X.shape[1],
            "selection_method": "intelligent",
            "features_removed": 0,
            "feature_scores": {}
        }
        
        try:
            # Only apply to numeric features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return X, selection_metadata
            
            # Calculate feature scores
            if task_type == "classification":
                selector = SelectKBest(score_func=f_classif, k='all')
            else:
                selector = SelectKBest(score_func=f_regression, k='all')
            
            selector.fit(X[numeric_cols], y)
            feature_scores = dict(zip(numeric_cols, selector.scores_))
            selection_metadata["feature_scores"] = feature_scores
            
            # Store feature importances for future use
            self._last_feature_importances = feature_scores.copy()
            
            # Determine number of features to keep
            n_features_to_keep = max(2, int(len(numeric_cols) * self.feature_selection_ratio))  # Keep at least 2 features
            
            # Select top features
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_features[:n_features_to_keep]]
            
            # Also keep features that meet minimum importance threshold
            important_features = [f for f, score in feature_scores.items() if score >= self.min_feature_importance]
            
            # Combine selections
            selected_numeric_features = list(set(top_features + important_features))
            
            # Keep all non-numeric features
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            selected_features = selected_numeric_features + list(non_numeric_cols)
            
            # Create result DataFrame
            X_result = X[selected_features]
            
            # Update metadata
            selection_metadata["features_removed"] = X.shape[1] - X_result.shape[1]
            selection_metadata["final_features"] = X_result.shape[1]
            
            logger.info(f"Selected {X_result.shape[1]} features from {X.shape[1]} original features")
            
            return X_result, selection_metadata
            
        except Exception as e:
            logger.warning(f"Failed to perform feature selection: {e}")
            return X, selection_metadata
    
    def _score_features(self, 
                       X: pd.DataFrame,
                       y: Union[np.ndarray, pd.Series],
                       task_type: str) -> Dict[str, float]:
        """
        Score individual features
        
        Args:
            X: Input features
            y: Target variable
            task_type: Type of ML task
            
        Returns:
            Feature scores
        """
        feature_scores = {}
        
        try:
            # Only score numeric features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                score = self._get_baseline_score(X[[col]], y)
                feature_scores[col] = score
            
        except Exception as e:
            logger.warning(f"Failed to score features: {e}")
        
        return feature_scores
    
    def get_feature_importance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive feature importance report
        
        Returns:
            Feature importance report
        """
        return {
            "original_features": self.original_features,
            "engineered_features": self.engineered_features,
            "selected_features": self.selected_features,
            "feature_metadata": self.feature_metadata
        }


def engineer_features_smart(X, 
                           y, 
                           task_type: str = "classification",
                           **kwargs) -> Tuple[Union[np.ndarray, pd.DataFrame], Dict[str, Any]]:
    """
    Convenience function for smart feature engineering
    
    Args:
        X: Input features
        y: Target variable
        task_type: Type of ML task
        **kwargs: Additional arguments for SmartFeatureEngineering
        
    Returns:
        Engineered features and metadata
    """
    engineer = SmartFeatureEngineering(**kwargs)
    return engineer.engineer_features(X, y, task_type)
