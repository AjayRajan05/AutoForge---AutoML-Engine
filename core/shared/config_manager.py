"""
⚙️ Dynamic Configuration Manager
Intelligent configuration management with adaptation
"""

import logging
import json
import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AutoMLConfig:
    """AutoML configuration data class"""
    # System settings
    max_time_seconds: Optional[float] = None
    max_trials: int = 100
    random_state: int = 42
    verbose: bool = True
    
    # Resource settings
    max_memory_mb: float = 4096
    max_cpu_percent: float = 80.0
    n_jobs: int = -1
    
    # Model settings
    enable_ensemble: bool = True
    enable_feature_selection: bool = True
    enable_hyperparameter_tuning: bool = True
    
    # Data settings
    handle_missing: str = "auto"  # auto, drop, impute
    handle_categorical: str = "auto"  # auto, encode, drop
    handle_outliers: str = "auto"  # auto, remove, cap
    
    # Validation settings
    cv_folds: int = 5
    test_size: float = 0.2
    stratify: bool = True
    
    # Advanced settings
    enable_meta_learning: bool = True
    enable_explainability: bool = True
    enable_early_stopping: bool = True
    
    # Performance settings
    cache_pipelines: bool = True
    use_incremental: bool = False
    batch_size: Optional[int] = None


class ConfigManager:
    """
    Dynamic Configuration Manager
    
    Manages AutoML configuration with intelligent adaptation
    based on system resources and dataset characteristics.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "automl_config.json"
        self.config = AutoMLConfig()
        self.load_config()
        
    def load_config(self) -> AutoMLConfig:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_dict = json.load(f)
                
                # Update config with loaded values
                for key, value in config_dict.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                logger.info(f"⚙️ Configuration loaded from {self.config_file}")
            else:
                logger.info("⚙️ No config file found, using defaults")
                
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
        
        return self.config
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_dict = asdict(self.config)
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"⚙️ Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                old_value = getattr(self.config, key)
                setattr(self.config, key, value)
                logger.info(f"⚙️ Updated {key}: {old_value} → {value}")
            else:
                logger.warning(f"Unknown config key: {key}")
    
    def get_config(self) -> AutoMLConfig:
        """Get current configuration"""
        return self.config
    
    def adapt_config(self, dataset_profile: Dict[str, Any]) -> AutoMLConfig:
        """Adapt configuration based on dataset characteristics"""
        adapted_config = AutoMLConfig(**asdict(self.config))
        
        # Size-based adaptations
        size_profile = dataset_profile.get("size_profile", {})
        size_category = size_profile.get("category", "medium")
        
        if size_category == "large":
            adapted_config.max_trials = min(adapted_config.max_trials, 50)
            adapted_config.cv_folds = min(adapted_config.cv_folds, 3)
            adapted_config.use_incremental = True
            adapted_config.enable_hyperparameter_tuning = False
            
        elif size_category == "small":
            adapted_config.cv_folds = min(adapted_config.cv_folds + 2, 10)
            adapted_config.enable_hyperparameter_tuning = True
        
        # Quality-based adaptations
        quality_profile = dataset_profile.get("quality_profile", {})
        quality_category = quality_profile.get("category", "good")
        
        if quality_category == "poor":
            adapted_config.handle_missing = "impute"
            adapted_config.handle_outliers = "cap"
            adapted_config.enable_feature_selection = True
        
        # Type-based adaptations
        type_profile = dataset_profile.get("type_profile", {})
        
        if type_profile.get("has_categorical", False):
            adapted_config.handle_categorical = "encode"
        
        # Complexity-based adaptations
        complexity_profile = dataset_profile.get("complexity_profile", {})
        
        if complexity_profile.get("is_imbalanced", False):
            adapted_config.stratify = True
            adapted_config.enable_ensemble = True
        
        # Resource-based adaptations
        adapted_config = self._adapt_for_resources(adapted_config)
        
        logger.info("⚙️ Configuration adapted based on dataset characteristics")
        return adapted_config
    
    def _adapt_for_resources(self, config: AutoMLConfig) -> AutoMLConfig:
        """Adapt configuration based on available resources"""
        try:
            import psutil
            
            # Memory adaptation
            available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
            
            if available_memory < 2048:  # Less than 2GB
                config.max_memory_mb = available_memory * 0.8
                config.n_jobs = 1
                config.enable_ensemble = False
                
            elif available_memory < 4096:  # Less than 4GB
                config.max_memory_mb = available_memory * 0.7
                config.n_jobs = min(config.n_jobs, 2)
            
            # CPU adaptation
            cpu_count = psutil.cpu_count()
            if config.n_jobs == -1:
                config.n_jobs = max(1, cpu_count - 1)
            
        except ImportError:
            logger.warning("psutil not available, skipping resource adaptation")
        
        return config
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return issues"""
        issues = []
        warnings = []
        
        # Time validation
        if self.config.max_time_seconds is not None and self.config.max_time_seconds <= 0:
            issues.append("max_time_seconds must be positive")
        
        # Trial validation
        if self.config.max_trials <= 0:
            issues.append("max_trials must be positive")
        
        # CV folds validation
        if self.config.cv_folds < 2:
            issues.append("cv_folds must be at least 2")
        
        # Test size validation
        if not 0 < self.config.test_size < 1:
            issues.append("test_size must be between 0 and 1")
        
        # Memory validation
        if self.config.max_memory_mb <= 0:
            issues.append("max_memory_mb must be positive")
        
        # CPU validation
        if self.config.n_jobs == 0:
            warnings.append("n_jobs=0 may cause issues, recommend using -1 or positive integer")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization-specific configuration"""
        return {
            "max_trials": self.config.max_trials,
            "max_time": self.config.max_time_seconds,
            "enable_hyperparameter_tuning": self.config.enable_hyperparameter_tuning,
            "enable_early_stopping": self.config.enable_early_stopping,
            "random_state": self.config.random_state
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration"""
        return {
            "handle_missing": self.config.handle_missing,
            "handle_categorical": self.config.handle_categorical,
            "handle_outliers": self.config.handle_outliers,
            "enable_feature_selection": self.config.enable_feature_selection,
            "enable_ensemble": self.config.enable_ensemble
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration"""
        return {
            "cv_folds": self.config.cv_folds,
            "test_size": self.config.test_size,
            "stratify": self.config.stratify
        }
    
    def get_resource_config(self) -> Dict[str, Any]:
        """Get resource configuration"""
        return {
            "max_memory_mb": self.config.max_memory_mb,
            "max_cpu_percent": self.config.max_cpu_percent,
            "n_jobs": self.config.n_jobs,
            "use_incremental": self.config.use_incremental,
            "batch_size": self.config.batch_size
        }
    
    def create_profile_specific_config(self, profile_name: str) -> AutoMLConfig:
        """Create configuration for specific profile"""
        profiles = {
            "fast": AutoMLConfig(
                max_trials=20,
                cv_folds=3,
                enable_hyperparameter_tuning=False,
                enable_feature_selection=False,
                enable_ensemble=False
            ),
            "accurate": AutoMLConfig(
                max_trials=200,
                cv_folds=10,
                enable_hyperparameter_tuning=True,
                enable_feature_selection=True,
                enable_ensemble=True
            ),
            "robust": AutoMLConfig(
                max_trials=50,
                cv_folds=5,
                handle_missing="impute",
                handle_outliers="cap",
                enable_feature_selection=True,
                enable_ensemble=True
            ),
            "research": AutoMLConfig(
                max_trials=500,
                cv_folds=5,
                enable_meta_learning=True,
                enable_explainability=True,
                enable_hyperparameter_tuning=True
            ),
            "production": AutoMLConfig(
                max_trials=30,
                cv_folds=3,
                enable_early_stopping=True,
                cache_pipelines=True,
                use_incremental=True,
                verbose=False
            )
        }
        
        if profile_name in profiles:
            return profiles[profile_name]
        else:
            logger.warning(f"Unknown profile: {profile_name}, using default")
            return AutoMLConfig()
    
    def __str__(self) -> str:
        """String representation of current config"""
        return f"AutoMLConfig(max_trials={self.config.max_trials}, cv_folds={self.config.cv_folds})"


# Global config manager instance
global_config_manager = ConfigManager()


def get_config() -> AutoMLConfig:
    """Get global configuration"""
    return global_config_manager.get_config()


def update_config(updates: Dict[str, Any]):
    """Update global configuration"""
    global_config_manager.update_config(updates)


def adapt_config(dataset_profile: Dict[str, Any]) -> AutoMLConfig:
    """Adapt global configuration based on dataset"""
    return global_config_manager.adapt_config(dataset_profile)
