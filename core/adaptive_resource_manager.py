"""
🔥 ADAPTIVE RESOURCE MANAGER
Automatically adjust to ANY resource constraints
"""

import logging
import psutil
import gc
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdaptiveResourceManager:
    """
    🔥 Adaptive Resource Manager - Automatically adjust to ANY resource constraints
    """
    
    def __init__(self):
        """Initialize resource management"""
        self.resource_profiles = {
            'minimal': {
                'n_trials': 1,
                'cv_folds': 2,
                'timeout': 30,
                'max_depth': 3,
                'n_estimators': 10,
                'max_features': 'sqrt',
                'memory_limit_mb': 512,
                'cpu_cores': 1,
                'simple_models_only': True,
                'disable_feature_engineering': True,
            },
            'low': {
                'n_trials': 3,
                'cv_folds': 2,
                'timeout': 60,
                'max_depth': 5,
                'n_estimators': 50,
                'max_features': 'sqrt',
                'memory_limit_mb': 1024,
                'cpu_cores': 2,
                'simple_models_only': False,
                'disable_feature_engineering': False,
            },
            'medium': {
                'n_trials': 10,
                'cv_folds': 3,
                'timeout': 120,
                'max_depth': 8,
                'n_estimators': 100,
                'max_features': None,
                'memory_limit_mb': 2048,
                'cpu_cores': 4,
                'simple_models_only': False,
                'disable_feature_engineering': False,
            },
            'high': {
                'n_trials': 50,
                'cv_folds': 5,
                'timeout': 300,
                'max_depth': 15,
                'n_estimators': 300,
                'max_features': None,
                'memory_limit_mb': 4096,
                'cpu_cores': 8,
                'simple_models_only': False,
                'disable_feature_engineering': False,
            }
        }
        
        self.current_profile = None
        self.system_info = self.get_system_info()
        
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information
        
        Returns:
            System resource information
        """
        try:
            # Memory information
            memory = psutil.virtual_memory()
            memory_info = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent_used': memory.percent
            }
            
            # CPU information
            cpu_info = {
                'cores': psutil.cpu_count(),
                'percent_used': psutil.cpu_percent(interval=1)
            }
            
            # Disk information
            disk = psutil.disk_usage('/')
            disk_info = {
                'total_gb': disk.total / (1024**3),
                'free_gb': disk.free / (1024**3),
                'used_gb': disk.used / (1024**3)
            }
            
            return {
                'memory': memory_info,
                'cpu': cpu_info,
                'disk': disk_info,
                'timestamp': pd.Timestamp.now()
            }
            
        except Exception as e:
            logger.warning(f"Failed to get system info: {str(e)}")
            return {}
    
    def detect_resource_level(self) -> str:
        """
        Auto-detect available resource level
        
        Returns:
            Resource level string
        """
        try:
            memory_gb = self.system_info.get('memory', {}).get('total_gb', 4)
            cpu_cores = self.system_info.get('cpu', {}).get('cores', 2)
            available_memory_gb = self.system_info.get('memory', {}).get('available_gb', 2)
            
            # Determine resource level based on available resources
            if available_memory_gb < 1 or cpu_cores < 2:
                return 'minimal'
            elif memory_gb < 4 or cpu_cores < 4:
                return 'low'
            elif memory_gb < 8 or cpu_cores < 8:
                return 'medium'
            else:
                return 'high'
                
        except Exception as e:
            logger.warning(f"Failed to detect resource level: {str(e)}")
            return 'low'  # Safe default
    
    def auto_configure(self, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Auto-configure based on available resources and constraints
        
        Args:
            constraints: Optional user constraints
            
        Returns:
            Configuration dictionary
        """
        # Detect resource level
        resource_level = self.detect_resource_level()
        
        # Get base configuration
        config = self.resource_profiles[resource_level].copy()
        
        # Apply user constraints if provided
        if constraints:
            config = self.apply_constraints(config, constraints)
        
        # Apply dynamic adjustments
        config = self.apply_dynamic_adjustments(config)
        
        self.current_profile = resource_level
        
        logger.info(f"Auto-configured for {resource_level} resources")
        logger.info(f"Configuration: {config}")
        
        return config
    
    def apply_constraints(self, config: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply user constraints to configuration
        
        Args:
            config: Base configuration
            constraints: User constraints
            
        Returns:
            Updated configuration
        """
        # Apply time constraint
        if 'time_limit' in constraints:
            time_limit = constraints['time_limit']
            if time_limit < 30:
                config.update(self.resource_profiles['minimal'])
            elif time_limit < 60:
                config.update(self.resource_profiles['low'])
            elif time_limit < 120:
                config.update(self.resource_profiles['medium'])
            
            config['timeout'] = min(config['timeout'], time_limit)
        
        # Apply memory constraint
        if 'memory_limit' in constraints:
            memory_limit_mb = constraints['memory_limit'] * 1024  # Convert GB to MB
            config['memory_limit_mb'] = min(config['memory_limit_mb'], memory_limit_mb)
            
            # Adjust complexity based on memory
            if memory_limit_mb < 1024:  # < 1GB
                config['n_trials'] = min(config['n_trials'], 3)
                config['cv_folds'] = min(config['cv_folds'], 2)
                config['simple_models_only'] = True
        
        # Apply CPU constraint
        if 'cpu_limit' in constraints:
            cpu_limit = constraints['cpu_limit']
            config['cpu_cores'] = min(config['cpu_cores'], cpu_limit)
            
            # Adjust trials based on CPU
            if cpu_limit < 2:
                config['n_trials'] = min(config['n_trials'], 5)
        
        # Apply accuracy constraint
        if 'accuracy_priority' in constraints and constraints['accuracy_priority']:
            # Favor accuracy over speed
            config['n_trials'] = int(config['n_trials'] * 1.5)
            config['cv_folds'] = min(config['cv_folds'] + 1, 5)
            config['simple_models_only'] = False
        
        # Apply speed constraint
        if 'speed_priority' in constraints and constraints['speed_priority']:
            # Favor speed over accuracy
            config['n_trials'] = max(config['n_trials'] // 2, 1)
            config['cv_folds'] = max(config['cv_folds'] - 1, 2)
            config['simple_models_only'] = True
            config['disable_feature_engineering'] = True
        
        return config
    
    def apply_dynamic_adjustments(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply dynamic adjustments based on current system state
        
        Args:
            config: Current configuration
            
        Returns:
            Adjusted configuration
        """
        # Check current memory usage
        memory_percent = self.system_info.get('memory', {}).get('percent_used', 50)
        
        if memory_percent > 80:
            # High memory usage - reduce complexity
            config['n_trials'] = max(config['n_trials'] // 2, 1)
            config['cv_folds'] = max(config['cv_folds'] - 1, 2)
            config['disable_feature_engineering'] = True
            logger.warning("High memory usage detected, reducing complexity")
        
        # Check current CPU usage
        cpu_percent = self.system_info.get('cpu', {}).get('percent_used', 50)
        
        if cpu_percent > 80:
            # High CPU usage - reduce parallelism
            config['cpu_cores'] = max(config['cpu_cores'] // 2, 1)
            config['n_trials'] = max(config['n_trials'] // 2, 1)
            logger.warning("High CPU usage detected, reducing parallelism")
        
        # Check available memory
        available_memory_gb = self.system_info.get('memory', {}).get('available_gb', 2)
        
        if available_memory_gb < 1:
            # Low available memory - use minimal config
            config.update(self.resource_profiles['minimal'])
            logger.warning("Low available memory, using minimal configuration")
        
        return config
    
    def monitor_resources(self) -> Dict[str, Any]:
        """
        Monitor current resource usage
        
        Returns:
            Current resource usage information
        """
        try:
            # Update system info
            self.system_info = self.get_system_info()
            
            # Get process info
            process = psutil.Process()
            process_info = {
                'memory_mb': process.memory_info().rss / (1024**2),
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
            }
            
            return {
                'system': self.system_info,
                'process': process_info,
                'profile': self.current_profile,
                'timestamp': pd.Timestamp.now()
            }
            
        except Exception as e:
            logger.warning(f"Failed to monitor resources: {str(e)}")
            return {}
    
    def optimize_memory(self, aggressive: bool = False) -> bool:
        """
        Optimize memory usage
        
        Args:
            aggressive: Whether to use aggressive optimization
            
        Returns:
            True if optimization was successful
        """
        try:
            # Run garbage collection
            gc.collect()
            
            if aggressive:
                # More aggressive optimization
                for _ in range(3):
                    gc.collect()
                
                # Clear large objects if possible
                import sys
                if hasattr(sys, 'getobjects'):
                    # This is CPython specific
                    objects = sys.getobjects()
                    # Find and clear large objects
                    for obj in objects:
                        if isinstance(obj, (list, dict, set)) and len(obj) > 1000:
                            obj.clear()
            
            logger.info("Memory optimization completed")
            return True
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {str(e)}")
            return False
    
    def get_optimal_batch_size(self, data_size: int) -> int:
        """
        Get optimal batch size for processing
        
        Args:
            data_size: Size of dataset
            
        Returns:
            Optimal batch size
        """
        available_memory_mb = self.system_info.get('memory', {}).get('available_gb', 2) * 1024
        
        # Estimate memory per sample (rough estimate)
        memory_per_sample_mb = 0.001  # 1KB per sample
        
        # Calculate batch size
        batch_size = min(
            int(available_memory_mb / memory_per_sample_mb),
            data_size,
            10000  # Maximum batch size
        )
        
        return max(batch_size, 1)  # At least 1
    
    def should_use_parallel(self, data_size: int) -> bool:
        """
        Determine if parallel processing should be used
        
        Args:
            data_size: Size of dataset
            
        Returns:
            True if parallel processing should be used
        """
        cpu_cores = self.system_info.get('cpu', {}).get('cores', 2)
        available_memory_gb = self.system_info.get('memory', {}).get('available_gb', 2)
        
        # Use parallel if we have multiple cores and enough memory
        return cpu_cores > 1 and available_memory_gb > 2 and data_size > 1000
    
    def get_resource_recommendations(self) -> Dict[str, Any]:
        """
        Get resource optimization recommendations
        
        Returns:
            Recommendations dictionary
        """
        recommendations = []
        
        memory_percent = self.system_info.get('memory', {}).get('percent_used', 50)
        cpu_percent = self.system_info.get('cpu', {}).get('percent_used', 50)
        available_memory_gb = self.system_info.get('memory', {}).get('available_gb', 2)
        
        if memory_percent > 80:
            recommendations.append({
                'type': 'memory',
                'priority': 'high',
                'message': 'High memory usage detected. Consider reducing model complexity or increasing memory.',
                'action': 'reduce_complexity'
            })
        
        if cpu_percent > 80:
            recommendations.append({
                'type': 'cpu',
                'priority': 'high',
                'message': 'High CPU usage detected. Consider reducing parallelism or model complexity.',
                'action': 'reduce_parallelism'
            })
        
        if available_memory_gb < 1:
            recommendations.append({
                'type': 'memory',
                'priority': 'critical',
                'message': 'Very low available memory. Use minimal configuration.',
                'action': 'use_minimal_config'
            })
        
        if self.system_info.get('cpu', {}).get('cores', 2) < 2:
            recommendations.append({
                'type': 'cpu',
                'priority': 'medium',
                'message': 'Limited CPU cores available. Consider reducing parallel processing.',
                'action': 'reduce_parallelism'
            })
        
        return {
            'recommendations': recommendations,
            'current_profile': self.current_profile,
            'system_info': self.system_info
        }


# Global instance
adaptive_resource_manager = AdaptiveResourceManager()
