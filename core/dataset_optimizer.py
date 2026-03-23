"""
Dataset-Aware Optimization Engine
Adaptive compute usage based on dataset characteristics
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings

logger = logging.getLogger(__name__)


class DatasetOptimizer:
    """
    Adaptive dataset optimization for large-scale AutoML
    """
    
    def __init__(self, 
                 large_dataset_threshold: int = 100_000,
                 sample_size: int = 20_000,
                 min_sample_size: int = 5_000,
                 max_sample_size: int = 50_000):
        """
        Initialize dataset optimizer
        
        Args:
            large_dataset_threshold: Dataset size to trigger sampling
            sample_size: Default sample size for large datasets
            min_sample_size: Minimum sample size to maintain
            max_sample_size: Maximum sample size allowed
        """
        self.large_dataset_threshold = large_dataset_threshold
        self.sample_size = sample_size
        self.min_sample_size = min_sample_size
        self.max_sample_size = max_sample_size
        
    def optimize_dataset(self, X, y, task_type: str = "classification") -> Tuple[Tuple, str, Dict[str, Any]]:
        """
        Optimize dataset based on size and characteristics
        
        Args:
            X: Features
            y: Target
            task_type: Type of ML task
            
        Returns:
            Tuple of (X_optimized, y_optimized), strategy_used, metadata
        """
        n_samples, n_features = X.shape
        
        logger.info(f"Analyzing dataset: {n_samples:,} samples × {n_features:,} features")
        
        # Determine optimization strategy
        if n_samples > self.large_dataset_threshold:
            return self._sample_large_dataset(X, y, task_type)
        else:
            return self._use_full_dataset(X, y, task_type)
    
    def _sample_large_dataset(self, X, y, task_type: str) -> Tuple[Tuple, str, Dict[str, Any]]:
        """
        Sample large dataset for efficient training
        
        Args:
            X: Features
            y: Target
            task_type: Type of ML task
            
        Returns:
            Tuple of (X_sampled, y_sampled), strategy, metadata
        """
        n_samples = X.shape[0]
        
        # Adaptive sample size based on dataset size
        if n_samples > 1_000_000:
            sample_size = min(self.max_sample_size, n_samples // 50)
        elif n_samples > 500_000:
            sample_size = min(self.sample_size * 2, n_samples // 25)
        else:
            sample_size = self.sample_size
            
        sample_size = max(sample_size, self.min_sample_size)
        
        logger.info(f"Large dataset detected ({n_samples:,} samples)")
        logger.info(f"Using adaptive sampling strategy: {sample_size:,} samples")
        
        # Stratified sampling for classification
        if task_type == "classification":
            try:
                # Convert to pandas for stratified sampling
                if not isinstance(X, pd.DataFrame):
                    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
                else:
                    X_df = X.copy()
                    
                y_series = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()
                
                # Stratified sampling
                X_sampled, y_sampled = self._stratified_sample(X_df, y_series, sample_size)
                
                # Convert back to numpy if needed
                if not isinstance(X, pd.DataFrame):
                    X_sampled = X_sampled.values
                    
            except Exception as e:
                logger.warning(f"Stratified sampling failed: {e}. Using random sampling.")
                X_sampled, y_sampled = self._random_sample(X, y, sample_size)
        else:
            # For regression, use random sampling
            X_sampled, y_sampled = self._random_sample(X, y, sample_size)
        
        metadata = {
            "original_size": n_samples,
            "sampled_size": len(X_sampled),
            "sampling_ratio": len(X_sampled) / n_samples,
            "strategy": "adaptive_sampling",
            "task_type": task_type
        }
        
        return (X_sampled, y_sampled), "sampled", metadata
    
    def _stratified_sample(self, X: pd.DataFrame, y: pd.Series, sample_size: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Perform stratified sampling maintaining class distribution
        
        Args:
            X: Features DataFrame
            y: Target Series
            sample_size: Target sample size
            
        Returns:
            Sampled X and y
        """
        # Calculate sample fraction
        sample_fraction = sample_size / len(X)
        
        # Ensure we don't sample more than available
        sample_fraction = min(sample_fraction, 1.0)
        
        # Perform stratified sampling
        X_sampled, _, y_sampled, _ = train_test_split(
            X, y, 
            test_size=1-sample_fraction,
            stratify=y,
            random_state=42
        )
        
        return X_sampled, y_sampled
    
    def _random_sample(self, X, y, sample_size: int) -> Tuple:
        """
        Perform random sampling
        
        Args:
            X: Features
            y: Target
            sample_size: Target sample size
            
        Returns:
            Sampled X and y
        """
        indices = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
        
        if isinstance(X, pd.DataFrame):
            X_sampled = X.iloc[indices].copy()
        else:
            X_sampled = X[indices].copy()
            
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_sampled = y.iloc[indices].copy()
        else:
            y_sampled = y[indices].copy()
            
        return X_sampled, y_sampled
    
    def _use_full_dataset(self, X, y, task_type: str) -> Tuple[Tuple, str, Dict[str, Any]]:
        """
        Use full dataset for smaller problems
        
        Args:
            X: Features
            y: Target
            task_type: Type of ML task
            
        Returns:
            Tuple of (X, y), strategy, metadata
        """
        n_samples = X.shape[0]
        
        logger.info(f"Using full dataset ({n_samples:,} samples) - no sampling needed")
        
        metadata = {
            "original_size": n_samples,
            "sampled_size": n_samples,
            "sampling_ratio": 1.0,
            "strategy": "full_dataset",
            "task_type": task_type
        }
        
        return (X, y), "full", metadata
    
    def analyze_dataset_characteristics(self, X, y) -> Dict[str, Any]:
        """
        Analyze dataset characteristics for optimization decisions
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary of dataset characteristics
        """
        n_samples, n_features = X.shape
        
        # Basic statistics
        characteristics = {
            "n_samples": n_samples,
            "n_features": n_features,
            "samples_per_feature": n_samples / n_features,
            "is_large_dataset": n_samples > self.large_dataset_threshold,
            "memory_estimate_mb": self._estimate_memory_usage(X, y)
        }
        
        # Feature analysis
        if isinstance(X, pd.DataFrame):
            characteristics.update({
                "numeric_features": len(X.select_dtypes(include=[np.number]).columns),
                "categorical_features": len(X.select_dtypes(include=['object', 'category']).columns),
                "missing_ratio": X.isnull().sum().sum() / X.size,
                "feature_names": list(X.columns)
            })
        else:
            characteristics.update({
                "numeric_features": n_features,
                "categorical_features": 0,
                "missing_ratio": np.isnan(X).sum() / X.size if X.dtype.kind in 'fc' else 0,
                "feature_names": [f"feature_{i}" for i in range(n_features)]
            })
        
        # Target analysis
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_values = y.values.flatten()
        else:
            y_values = y.flatten()
            
        characteristics.update({
            "target_unique_values": len(np.unique(y_values[~np.isnan(y_values)])),
            "target_dtype": str(y_values.dtype),
            "target_missing_ratio": np.isnan(y_values).sum() / len(y_values)
        })
        
        return characteristics
    
    def _estimate_memory_usage(self, X, y) -> float:
        """
        Estimate memory usage in MB
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Estimated memory usage in MB
        """
        try:
            # Calculate memory usage
            if isinstance(X, pd.DataFrame):
                x_memory = X.memory_usage(deep=True).sum()
            else:
                x_memory = X.nbytes
                
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_memory = y.memory_usage(deep=True).sum()
            else:
                y_memory = y.nbytes
                
            total_bytes = x_memory + y_memory
            total_mb = total_bytes / (1024 * 1024)
            
            return total_mb
            
        except Exception:
            # Fallback estimation
            return (X.shape[0] * X.shape[1] * 8 + len(y) * 8) / (1024 * 1024)
    
    def get_optimization_recommendations(self, X, y) -> Dict[str, Any]:
        """
        Get optimization recommendations based on dataset analysis
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Optimization recommendations
        """
        characteristics = self.analyze_dataset_characteristics(X, y)
        recommendations = {
            "dataset_characteristics": characteristics,
            "optimization_strategy": "full",
            "recommended_trials": 50,
            "recommended_cv_folds": 5,
            "performance_tips": []
        }
        
        # Large dataset recommendations
        if characteristics["is_large_dataset"]:
            recommendations["optimization_strategy"] = "sampled"
            recommendations["recommended_trials"] = 30  # Fewer trials for sampled data
            recommendations["performance_tips"].append(
                f"Consider using sampled dataset ({self.sample_size:,} samples) for faster training"
            )
        
        # High-dimensional data recommendations
        if characteristics["samples_per_feature"] < 10:
            recommendations["recommended_trials"] = 30
            recommendations["performance_tips"].append(
                "High-dimensional data detected - consider feature selection"
            )
        
        # Memory usage recommendations
        if characteristics["memory_estimate_mb"] > 1000:  # > 1GB
            recommendations["performance_tips"].append(
                f"High memory usage ({characteristics['memory_estimate_mb']:.1f} MB) - consider sampling"
            )
        
        # Missing data recommendations
        if characteristics["missing_ratio"] > 0.1:
            recommendations["performance_tips"].append(
                "High missing value ratio - consider imputation strategies"
            )
        
        return recommendations


def optimize_dataset(X, y, task_type: str = "classification", **kwargs) -> Tuple[Tuple, str, Dict[str, Any]]:
    """
    Convenience function for dataset optimization
    
    Args:
        X: Features
        y: Target
        task_type: Type of ML task
        **kwargs: Additional arguments for DatasetOptimizer
        
    Returns:
        Tuple of (X_optimized, y_optimized), strategy_used, metadata
    """
    optimizer = DatasetOptimizer(**kwargs)
    return optimizer.optimize_dataset(X, y, task_type)
