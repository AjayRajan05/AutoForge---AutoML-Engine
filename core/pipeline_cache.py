"""
Pipeline-Level Caching with Joblib
Massive speedup in search loops through intelligent caching
"""

import os
import hashlib
import pickle
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from joblib import Memory
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)


class PipelineCache:
    """
    Intelligent caching for pipeline components and transformations
    """
    
    def __init__(self, 
                 cache_dir: str = "cache/pipeline",
                 memory_limit: int = 2048,  # MB
                 verbose: int = 0):
        """
        Initialize pipeline cache
        
        Args:
            cache_dir: Directory for cache storage
            memory_limit: Memory limit in MB
            verbose: Verbosity level
        """
        self.cache_dir = cache_dir
        self.memory_limit = memory_limit
        self.verbose = verbose
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize joblib memory
        self.memory = Memory(location=cache_dir, verbose=verbose, mmap_mode='r')
        
        # Cache statistics
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "evictions": 0,
            "total_time_saved": 0.0
        }
        
        logger.info(f"Pipeline cache initialized: {cache_dir} (limit: {memory_limit}MB)")
    
    def get_cache_key(self, 
                     X_shape: Tuple[int, int], 
                     X_dtype: str,
                     params: Dict[str, Any],
                     component_type: str) -> str:
        """
        Generate cache key for pipeline component
        
        Args:
            X_shape: Shape of input data
            X_dtype: Data type of input
            params: Component parameters
            component_type: Type of component
            
        Returns:
            Cache key string
        """
        # Create deterministic key from input characteristics
        key_data = {
            "shape": X_shape,
            "dtype": str(X_dtype),
            "params": self._normalize_params(params),
            "type": component_type
        }
        
        # Generate hash
        key_str = str(sorted(key_data.items()))
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"{component_type}_{key_hash}"
    
    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameters for consistent hashing
        
        Args:
            params: Raw parameters
            
        Returns:
            Normalized parameters
        """
        normalized = {}
        
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                normalized[key] = value
            elif isinstance(value, (list, tuple)):
                normalized[key] = tuple(value)
            elif isinstance(value, np.ndarray):
                normalized[key] = tuple(value.shape)
            elif value is None:
                normalized[key] = None
            else:
                # For complex objects, use string representation
                normalized[key] = str(type(value))
        
        return normalized
    
    def cached_fit_transform(self, 
                           X: np.ndarray,
                           component_type: str,
                           component_params: Dict[str, Any],
                           cache_key: str) -> Tuple[np.ndarray, BaseEstimator]:
        """
        Cached fit and transform operation
        
        Args:
            X: Input data
            component_type: Type of component
            component_params: Component parameters
            cache_key: Cache key
            
        Returns:
            Transformed data and fitted component
        """
        # Use the memory cache properly
        cached_func = self.memory.cache(self._fit_transform_impl)
        
        return cached_func(X, component_type, component_params, cache_key)
    
    def _fit_transform_impl(self, 
                           X: np.ndarray,
                           component_type: str,
                           component_params: Dict[str, Any],
                           cache_key: str) -> Tuple[np.ndarray, BaseEstimator]:
        """Implementation of fit and transform for caching"""
        start_time = time.time()
        
        # Create component
        component = self._create_component(component_type, component_params)
        
        # Handle None component (no transformation)
        if component is None:
            return X, None
        
        # Fit and transform
        if hasattr(component, 'fit_transform'):
            X_transformed = component.fit_transform(X)
        else:
            X_transformed = component.fit(X).transform(X)
        
        fit_time = time.time() - start_time
        
        if self.verbose > 0:
            logger.debug(f"Cached fit_transform ({component_type}): {fit_time:.3f}s")
        
        return X_transformed, component
    
    def cached_transform(self, 
                        X: np.ndarray,
                        component: BaseEstimator,
                        cache_key: str) -> np.ndarray:
        """
        Cached transform operation
        
        Args:
            X: Input data
            component: Fitted component
            cache_key: Cache key
            
        Returns:
            Transformed data
        """
        # Use the memory cache properly
        cached_func = self.memory.cache(self._transform_impl)
        
        return cached_func(X, component, cache_key)
    
    def _transform_impl(self, 
                       X: np.ndarray,
                       component: BaseEstimator,
                       cache_key: str) -> np.ndarray:
        """Implementation of transform for caching"""
        start_time = time.time()
        
        if hasattr(component, 'transform'):
            X_transformed = component.transform(X)
        else:
            X_transformed = component.predict(X)
        
        transform_time = time.time() - start_time
        
        if self.verbose > 0:
            logger.debug(f"Cached transform: {transform_time:.3f}s")
        
        return X_transformed
    
    def _create_component(self, component_type: str, params: Dict[str, Any]) -> BaseEstimator:
        """
        Create pipeline component based on type and parameters
        
        Args:
            component_type: Type of component
            params: Component parameters
            
        Returns:
            Created component
        """
        component_map = {
            "scaler_standard": StandardScaler,
            "scaler_minmax": MinMaxScaler,
            "scaler_robust": RobustScaler,
            "scaler_none": lambda: None,  # No scaling
            "imputer_mean": lambda: SimpleImputer(strategy='mean'),
            "imputer_median": lambda: SimpleImputer(strategy='median'),
            "imputer_most_frequent": lambda: SimpleImputer(strategy='most_frequent'),
            "imputer_knn": KNNImputer,
            "selector_kbest": SelectKBest,
            "selector_model": SelectFromModel,
            "selector_rfe": RFE,
            "pca": PCA
        }
        
        if component_type not in component_map:
            raise ValueError(f"Unknown component type: {component_type}")
        
        component_class = component_map[component_type]
        
        # Handle components that need parameters
        if component_type == "imputer_knn":
            return component_class(n_neighbors=params.get('n_neighbors', 5))
        elif component_type == "selector_kbest":
            return component_class(k=params.get('k', 10))
        elif component_type == "selector_model":
            # Use RandomForest as base selector
            selector = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                random_state=42
            )
            return component_class(selector, threshold=params.get('threshold', 'median'))
        elif component_type == "selector_rfe":
            # Use RandomForest as base estimator
            estimator = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                random_state=42
            )
            return component_class(estimator, n_features_to_select=params.get('n_features', 10))
        elif component_type == "pca":
            return component_class(n_components=params.get('n_components', 0.95))
        else:
            return component_class()
    
    def get_cached_pipeline(self, 
                           X: np.ndarray,
                           y: np.ndarray,
                           pipeline_config: Dict[str, Any]) -> Tuple[np.ndarray, List[BaseEstimator]]:
        """
        Get cached pipeline or create and cache new one
        
        Args:
            X: Input data
            y: Target data
            pipeline_config: Pipeline configuration
            
        Returns:
            Transformed data and list of fitted components
        """
        start_time = time.time()
        X_transformed = X.copy()
        components = []
        
        # Process each pipeline step
        steps = pipeline_config.get('steps', [])
        
        for step_config in steps:
            step_type = step_config['type']
            step_params = step_config.get('params', {})
            
            # Generate cache key
            cache_key = self.get_cache_key(
                X_transformed.shape, 
                str(X_transformed.dtype),
                step_params,
                step_type
            )
            
            # Try to get from cache
            try:
                X_transformed, component = self.cached_fit_transform(
                    X_transformed, step_type, step_params, cache_key
                )
                components.append(component)
                self.cache_stats["hits"] += 1
                
                if self.verbose > 1:
                    logger.debug(f"Cache hit: {step_type}")
                    
            except Exception as e:
                # Cache miss, create component directly
                logger.debug(f"Cache miss: {step_type} - {e}")
                self.cache_stats["misses"] += 1
                
                component = self._create_component(step_type, step_params)
                
                # Handle None component (no transformation)
                if component is None:
                    components.append(None)
                    continue
                
                if hasattr(component, 'fit_transform'):
                    # Check if this is a feature selection component that needs y
                    if 'selector' in step_type or 'feature_selection' in step_type:
                        X_transformed = component.fit_transform(X_transformed, y)
                    else:
                        X_transformed = component.fit_transform(X_transformed)
                else:
                    X_transformed = component.fit(X_transformed).transform(X_transformed)
                
                components.append(component)
        
        total_time = time.time() - start_time
        
        if self.verbose > 0:
            logger.info(f"Pipeline processed in {total_time:.3f}s")
        
        return X_transformed, components
    
    def transform_with_cache(self, 
                           X: np.ndarray,
                           components: List[BaseEstimator],
                           pipeline_config: Dict[str, Any]) -> np.ndarray:
        """
        Transform data using cached components
        
        Args:
            X: Input data
            components: List of fitted components
            pipeline_config: Pipeline configuration
            
        Returns:
            Transformed data
        """
        start_time = time.time()
        X_transformed = X.copy()
        
        for i, component in enumerate(components):
            step_config = pipeline_config.get('steps', [])[i] if i < len(pipeline_config.get('steps', [])) else {}
            step_type = step_config.get('type', f'component_{i}')
            
            # Handle None component (no transformation)
            if component is None:
                continue
            
            if hasattr(component, 'transform'):
                X_transformed = component.transform(X_transformed)
            else:
                # Component might be a fitted transformer without explicit transform
                X_transformed = component.fit_transform(X_transformed)
        
        total_time = time.time() - start_time
        
        if self.verbose > 0:
            logger.info(f"Pipeline transform completed in {total_time:.3f}s")
        
        return X_transformed
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            self.memory.clear()
            self.cache_stats = {
                "hits": 0,
                "misses": 0,
                "saves": 0,
                "evictions": 0,
                "total_time_saved": 0.0
            }
            logger.info("Pipeline cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics
        
        Returns:
            Cache statistics dictionary
        """
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        stats = self.cache_stats.copy()
        stats.update({
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_dir": self.cache_dir,
            "cache_size_mb": self._get_cache_size()
        })
        
        return stats
    
    def _get_cache_size(self) -> float:
        """
        Get current cache size in MB
        
        Returns:
            Cache size in MB
        """
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    def optimize_cache(self):
        """Optimize cache by removing old or large entries"""
        try:
            # Get cache directory size
            current_size = self._get_cache_size()
            
            if current_size > self.memory_limit * 0.9:  # 90% of limit
                logger.warning(f"Cache size ({current_size:.1f}MB) approaching limit ({self.memory_limit}MB)")
                
                # Clear cache and let it rebuild with frequently used items
                self.clear_cache()
                self.cache_stats["evictions"] += 1
                
                logger.info("Cache optimized - cleared for rebuilding")
        
        except Exception as e:
            logger.error(f"Failed to optimize cache: {e}")


class CachedPipelineBuilder:
    """
    Pipeline builder with intelligent caching
    """
    
    def __init__(self, cache: Optional[PipelineCache] = None):
        """
        Initialize cached pipeline builder
        
        Args:
            cache: Pipeline cache instance
        """
        self.cache = cache or PipelineCache()
    
    def build_cached_pipeline(self, 
                             X: np.ndarray,
                             y: np.ndarray,
                             config: Dict[str, Any]) -> Tuple[np.ndarray, List[BaseEstimator]]:
        """
        Build pipeline with caching
        
        Args:
            X: Input data
            y: Target data
            config: Pipeline configuration
            
        Returns:
            Transformed data and fitted components
        """
        return self.cache.get_cached_pipeline(X, y, config)
    
    def transform_with_cache(self, 
                           X: np.ndarray,
                           components: List[BaseEstimator],
                           config: Dict[str, Any]) -> np.ndarray:
        """
        Transform data using cached components
        
        Args:
            X: Input data
            components: Fitted components
            config: Pipeline configuration
            
        Returns:
            Transformed data
        """
        return self.cache.transform_with_cache(X, components, config)


# Global cache instance
_global_cache = None

def get_global_cache() -> PipelineCache:
    """Get global pipeline cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = PipelineCache()
    return _global_cache

def clear_global_cache():
    """Clear global pipeline cache"""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear_cache()
