"""
🏭 Engine Factory - Creates and manages AutoML execution engines
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class EngineFactory:
    """
    Factory for creating AutoML execution engines
    
    This factory creates the appropriate engine based on the requested mode.
    Each engine has different capabilities and trade-offs.
    """
    
    @staticmethod
    def create(mode: str):
        """
        Create an AutoML engine
        
        Args:
            mode: Engine mode ("adaptive", "bulletproof", "research", "production")
            
        Returns:
            Engine instance
        """
        engines = {
            "adaptive": AdaptiveEngine,
            "bulletproof": BulletproofEngine,
            "research": ResearchEngine,
            "production": ProductionEngine
        }
        
        engine_class = engines.get(mode, AdaptiveEngine)
        
        try:
            engine = engine_class()
            logger.info(f"🏭 Created {mode} engine: {engine_class.__name__}")
            return engine
            
        except Exception as e:
            logger.error(f"❌ Failed to create {mode} engine: {e}")
            logger.warning("🛡️ Falling back to bulletproof engine")
            return BulletproofEngine()


class BaseEngine:
    """Base class for all AutoML engines"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.capabilities = []
        self.is_fallback = False
        
    def fit_with_strategy(self, X, y, strategy):
        """Fit model using provided strategy"""
        raise NotImplementedError("Subclasses must implement fit_with_strategy")
    
    def predict(self, X):
        """Make predictions"""
        raise NotImplementedError("Subclasses must implement predict")
    
    def get_capabilities(self):
        """Get engine capabilities"""
        return self.capabilities


class AdaptiveEngine(BaseEngine):
    """
    Adaptive Engine - Main execution engine with intelligent adaptation
    
    Features:
    - Strategy-based execution
    - Performance monitoring
    - Dynamic adaptation
    - Intelligent fallbacks
    """
    
    def __init__(self):
        super().__init__()
        self.name = "AdaptiveEngine"
        self.capabilities = [
            "strategy_execution",
            "performance_monitoring", 
            "dynamic_adaptation",
            "intelligent_fallbacks"
        ]
        
        # Import components with error handling
        try:
            from ..truly_bulletproof_automl import TrulyBulletproofAutoML
            self.bulletproof_fallback = TrulyBulletproofAutoML()
        except ImportError:
            logger.warning("Bulletproof engine not available")
            self.bulletproof_fallback = None
    
    def fit_with_strategy(self, X, y, strategy):
        """
        Fit model using adaptive strategy execution
        
        Args:
            X: Feature data
            y: Target data
            strategy: Execution strategy
            
        Returns:
            Training results
        """
        try:
            logger.info(f"🔧 Adaptive engine executing strategy: {strategy.get('primary_strategy', 'unknown')}")
            
            # Apply preprocessing strategy
            X_processed = self._apply_preprocessing(X, strategy)
            
            # Apply feature engineering strategy
            X_featured = self._apply_feature_engineering(X_processed, strategy)
            
            # Select and optimize models based on strategy
            model_results = self._execute_model_strategy(X_featured, y, strategy)
            
            # Return best result
            best_result = max(model_results, key=lambda x: x.get('score', 0))
            
            return {
                "best_model": best_result.get('model', 'unknown'),
                "best_score": best_result.get('score', 0),
                "models_tried": [r.get('model') for r in model_results],
                "training_time": best_result.get('time', 0),
                "strategy_used": strategy,
                "engine": self.name
            }
            
        except Exception as e:
            logger.error(f"❌ Adaptive engine failed: {e}")
            return self._fallback_to_bulletproof(X, y)
    
    def _apply_preprocessing(self, X, strategy):
        """Apply preprocessing based on strategy"""
        import pandas as pd
        import numpy as np
        
        X_processed = X.copy() if hasattr(X, 'copy') else X
        
        # Handle missing values
        if "robust_imputation" in strategy.get("preprocessing", []):
            if isinstance(X_processed, pd.DataFrame):
                X_processed = X_processed.fillna(X_processed.median())
            else:
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X_processed = imputer.fit_transform(X_processed)
        
        # Handle outliers
        if "outlier_removal" in strategy.get("preprocessing", []):
            if isinstance(X_processed, pd.DataFrame):
                # Simple outlier removal using IQR
                for col in X_processed.select_dtypes(include=[np.number]).columns:
                    Q1 = X_processed[col].quantile(0.25)
                    Q3 = X_processed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    X_processed[col] = X_processed[col].clip(lower_bound, upper_bound)
        
        return X_processed
    
    def _apply_feature_engineering(self, X, strategy):
        """Apply feature engineering based on strategy"""
        X_featured = X.copy() if hasattr(X, 'copy') else X
        
        # Add polynomial features if requested
        if "polynomial_features" in strategy.get("feature_engineering", []):
            try:
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=2, include_bias=False)
                if hasattr(X_featured, 'values'):
                    X_featured = poly.fit_transform(X_featured.values)
                else:
                    X_featured = poly.fit_transform(X_featured)
            except ImportError:
                logger.warning("Polynomial features not available")
        
        return X_featured
    
    def _execute_model_strategy(self, X, y, strategy):
        """Execute model selection and optimization"""
        import time
        from sklearn.metrics import accuracy_score, r2_score
        from sklearn.model_selection import train_test_split
        
        # Get models from strategy
        models = strategy.get("models", ["random_forest"])
        max_trials = strategy.get("optimization", {}).get("max_trials", 10)
        
        results = []
        
        # Train/test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        for model_name in models:
            try:
                start_time = time.time()
                
                # Get and train model
                model = self._get_model(model_name)
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                
                if len(np.unique(y)) < 20:  # Classification
                    score = accuracy_score(y_test, y_pred)
                else:  # Regression
                    score = r2_score(y_test, y_pred)
                
                training_time = time.time() - start_time
                
                results.append({
                    'model': model_name,
                    'score': score,
                    'time': training_time,
                    'model_object': model
                })
                
                logger.info(f"✅ {model_name}: {score:.3f} in {training_time:.2f}s")
                
            except Exception as e:
                logger.warning(f"❌ {model_name} failed: {e}")
                continue
        
        return results
    
    def _get_model(self, model_name):
        """Get model instance by name"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.svm import SVC, SVR
        
        models = {
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=100),
            'svm': SVC(random_state=42),
            'linear_regression': LinearRegression(),
            'random_forest_reg': RandomForestRegressor(n_estimators=50, random_state=42),
            'svr': SVR()
        }
        
        return models.get(model_name, RandomForestClassifier(n_estimators=50, random_state=42))
    
    def _fallback_to_bulletproof(self, X, y):
        """Fallback to bulletproof engine"""
        if self.bulletproof_fallback:
            logger.warning("🛡️ Falling back to bulletproof engine")
            try:
                self.bulletproof_fallback.fit(X, y)
                stats = self.bulletproof_fallback.get_performance_stats()
                return {
                    "best_model": stats.get("best_model", "bulletproof_fallback"),
                    "best_score": stats.get("best_score", 0),
                    "models_tried": ["bulletproof_fallback"],
                    "training_time": stats.get("training_time", 0),
                    "strategy_used": {"fallback": True},
                    "engine": "BulletproofFallback"
                }
            except Exception as e:
                logger.error(f"❌ Even bulletproof fallback failed: {e}")
        
        # Final fallback
        return {
            "best_model": "none",
            "best_score": 0,
            "models_tried": [],
            "training_time": 0,
            "strategy_used": {"fallback": True, "failed": True},
            "engine": "Failed"
        }
    
    def predict(self, X):
        """Make predictions"""
        if hasattr(self, 'best_model_object'):
            return self.best_model_object.predict(X)
        elif self.bulletproof_fallback:
            return self.bulletproof_fallback.predict(X)
        else:
            raise ValueError("No trained model available")


class BulletproofEngine(BaseEngine):
    """
    Bulletproof Engine - Ultra-reliable fallback engine
    
    Uses the proven TrulyBulletproofAutoML system as fallback.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "BulletproofEngine"
        self.capabilities = ["bulletproof_reliability", "universal_compatibility"]
        self.is_fallback = True
        
        try:
            from ..truly_bulletproof_automl import TrulyBulletproofAutoML
            self.bulletproof_automl = TrulyBulletproofAutoML()
        except ImportError:
            logger.error("❌ TrulyBulletproofAutoML not available")
            self.bulletproof_automl = None
    
    def fit_with_strategy(self, X, y, strategy):
        """Fit using bulletproof approach"""
        if not self.bulletproof_automl:
            raise ImportError("Bulletproof AutoML not available")
        
        # Use bulletproof system with strategy hints
        max_time = strategy.get("optimization", {}).get("max_time", 60)
        max_trials = strategy.get("optimization", {}).get("max_trials", 10)
        
        self.bulletproof_automl.max_time = max_time
        self.bulletproof_automl.max_trials = max_trials
        
        self.bulletproof_automl.fit(X, y)
        stats = self.bulletproof_automl.get_performance_stats()
        
        return {
            "best_model": stats.get("best_model", "bulletproof"),
            "best_score": stats.get("best_score", 0),
            "models_tried": ["bulletproof"],
            "training_time": stats.get("training_time", 0),
            "strategy_used": strategy,
            "engine": self.name
        }
    
    def predict(self, X):
        """Make predictions"""
        if not self.bulletproof_automl:
            raise ValueError("Bulletproof AutoML not trained")
        return self.bulletproof_automl.predict(X)


class ResearchEngine(BaseEngine):
    """
    Research Engine - Advanced features for experimentation
    
    Includes NAS, multimodal, and other advanced capabilities.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "ResearchEngine"
        self.capabilities = ["nas", "multimodal", "advanced_optimization"]
        
        # Try to import advanced features
        self.has_advanced_features = self._check_advanced_features()
    
    def _check_advanced_features(self):
        """Check if advanced features are available"""
        try:
            from ..revolutionary_automl import AdvancedAutoML
            self.advanced_automl = AdvancedAutoML(enable_all_advanced_features=False)
            return True
        except ImportError:
            logger.warning("⚠️ Advanced features not available, using adaptive engine")
            self.adaptive_engine = AdaptiveEngine()
            return False
    
    def fit_with_strategy(self, X, y, strategy):
        """Fit using research capabilities"""
        if self.has_advanced_features:
            return self._fit_with_advanced(X, y, strategy)
        else:
            logger.info("🔧 Using adaptive engine as fallback")
            return self.adaptive_engine.fit_with_strategy(X, y, strategy)
    
    def _fit_with_advanced(self, X, y, strategy):
        """Fit with advanced features"""
        # Use advanced AutoML with strategy hints
        max_trials = strategy.get("optimization", {}).get("max_trials", 50)
        
        self.advanced_automl.n_trials = max_trials
        self.advanced_automl.fit(X, y)
        
        return {
            "best_model": getattr(self.advanced_automl, 'best_model_name', 'advanced'),
            "best_score": getattr(self.advanced_automl, 'best_score', 0),
            "models_tried": ["advanced"],
            "training_time": 0,  # Not tracked in advanced automl
            "strategy_used": strategy,
            "engine": self.name
        }
    
    def predict(self, X):
        """Make predictions"""
        if self.has_advanced_features:
            return self.advanced_automl.predict(X)
        else:
            return self.adaptive_engine.predict(X)


class ProductionEngine(BaseEngine):
    """
    Production Engine - Optimized for production deployment
    
    Focuses on speed, reliability, and resource efficiency.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "ProductionEngine"
        self.capabilities = ["fast_execution", "resource_efficiency", "monitoring"]
        self.adaptive_engine = AdaptiveEngine()
    
    def fit_with_strategy(self, X, y, strategy):
        """Fit optimized for production"""
        # Optimize strategy for production
        production_strategy = self._optimize_for_production(strategy)
        
        # Use adaptive engine with production optimizations
        result = self.adaptive_engine.fit_with_strategy(X, y, production_strategy)
        result["engine"] = self.name
        result["production_optimized"] = True
        
        return result
    
    def _optimize_for_production(self, strategy):
        """Optimize strategy for production constraints"""
        prod_strategy = strategy.copy()
        
        # Reduce trials for speed
        if "optimization" in prod_strategy:
            prod_strategy["optimization"]["max_trials"] = min(
                prod_strategy["optimization"].get("max_trials", 50), 20
            )
        
        # Prioritize fast models
        if "models" in prod_strategy:
            fast_models = ["random_forest", "logistic_regression"]
            prod_strategy["models"] = [
                m for m in prod_strategy["models"] if m in fast_models
            ] or fast_models
        
        return prod_strategy
    
    def predict(self, X):
        """Make predictions"""
        return self.adaptive_engine.predict(X)
