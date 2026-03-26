"""
🚀 Unified AutoML - One Interface to Rule Them All
Category-Defining Adaptive AutoML System
"""

import logging
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np

# Import core components
from ..core.engine_factory import EngineFactory
from ..intelligence.dataset_analyzer import DatasetIntelligence
from ..intelligence.strategy_selector import StrategySelector

logger = logging.getLogger(__name__)


class UnifiedAutoML:
    """
    Unified AutoML System with Intelligent Decision Engine
    
    This is the main entry point for the category-defining AutoML system.
    It automatically selects the best strategy based on data characteristics.
    
    Features:
    - 🧠 Intelligent strategy selection
    - 🔄 Adaptive execution
    - 🛡️ Bulletproof fallbacks
    - 📈 Self-improving capabilities
    - 💡 Explainable decisions
    """
    
    def __init__(self, 
                 mode: str = "adaptive",
                 strategy: str = "auto",
                 max_time: Optional[float] = None,
                 max_trials: Optional[int] = None,
                 enable_explainability: bool = True,
                 enable_learning: bool = True):
        """
        Initialize Unified AutoML
        
        Args:
            mode: Engine mode ("adaptive", "bulletproof", "research", "production")
            strategy: Strategy selection ("auto", "fast", "accurate", "robust")
            max_time: Maximum time in seconds
            max_trials: Maximum number of trials
            enable_explainability: Enable explainable decisions
            enable_learning: Enable self-improving capabilities
        """
        self.mode = mode
        self.strategy = strategy
        self.max_time = max_time
        self.max_trials = max_trials
        self.enable_explainability = enable_explainability
        self.enable_learning = enable_learning
        
        # Initialize core components
        self.engine = EngineFactory.create(mode)
        self.dataset_analyzer = DatasetIntelligence()
        self.strategy_selector = StrategySelector()
        
        # Results storage
        self.dataset_profile = None
        self.selected_strategy = None
        self.execution_results = None
        self.explanations = {}
        
        logger.info(f"🚀 Unified AutoML initialized with mode='{mode}', strategy='{strategy}'")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray]) -> 'UnifiedAutoML':
        """
        Fit the AutoML system with intelligent strategy selection
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Self for method chaining
        """
        try:
            logger.info("🧠 Starting intelligent AutoML workflow...")
            
            # Step 1: Analyze dataset characteristics
            logger.info("📊 Analyzing dataset characteristics...")
            self.dataset_profile = self.dataset_analyzer.analyze(X, y)
            
            # Step 2: Select optimal strategy
            logger.info("🎯 Selecting optimal strategy...")
            self.selected_strategy = self.strategy_selector.select_strategy(
                self.dataset_profile, 
                self.strategy,
                self.max_time,
                self.max_trials
            )
            
            # Step 3: Explain decision (if enabled)
            if self.enable_explainability:
                self.explanations = self._explain_strategy_selection()
                logger.info(f"💡 Strategy selected: {self.selected_strategy.get('primary_strategy', 'unknown')}")
            
            # Step 4: Execute with adaptive configuration
            logger.info("⚡ Executing with adaptive configuration...")
            self.execution_results = self.engine.fit_with_strategy(
                X, y, self.selected_strategy
            )
            
            # Step 5: Learn from results (if enabled)
            if self.enable_learning:
                self._learn_from_results()
            
            logger.info("✅ Unified AutoML training completed successfully!")
            return self
            
        except Exception as e:
            logger.error(f"❌ Unified AutoML failed: {e}")
            # Fallback to bulletproof
            logger.warning("🛡️ Falling back to bulletproof mode...")
            self.engine = EngineFactory.create("bulletproof")
            self.execution_results = self.engine.fit(X, y)
            return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Feature data
            
        Returns:
            Predictions
        """
        if self.execution_results is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.engine.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make probability predictions (classification only)
        
        Args:
            X: Feature data
            
        Returns:
            Probability predictions
        """
        if self.execution_results is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if hasattr(self.engine, 'predict_proba'):
            return self.engine.predict_proba(X)
        else:
            logger.warning("Probability predictions not available for this model")
            return None
    
    def explain(self) -> Dict[str, Any]:
        """
        Get explanations for AutoML decisions
        
        Returns:
            Dictionary containing explanations
        """
        if not self.enable_explainability:
            return {"error": "Explainability not enabled"}
        
        explanations = {
            "dataset_profile": self.dataset_profile,
            "selected_strategy": self.selected_strategy,
            "decision_explanations": self.explanations,
            "execution_summary": self._get_execution_summary()
        }
        
        return explanations
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Performance statistics dictionary
        """
        if self.execution_results is None:
            return {"error": "No training results available"}
        
        stats = {
            "best_score": self.execution_results.get("best_score", 0),
            "best_model": self.execution_results.get("best_model", "unknown"),
            "training_time": self.execution_results.get("training_time", 0),
            "trials_used": self.execution_results.get("trials_used", 0),
            "dataset_profile": self.dataset_profile,
            "selected_strategy": self.selected_strategy,
            "mode": self.mode
        }
        
        return stats
    
    def _explain_strategy_selection(self) -> Dict[str, Any]:
        """Explain why a particular strategy was selected"""
        explanations = []
        
        profile = self.dataset_profile
        strategy = self.selected_strategy
        
        # Size-based explanations
        if profile.get("size_profile") == "large":
            explanations.append({
                "factor": "dataset_size",
                "observation": "Large dataset detected",
                "decision": "Used sampling and fast models",
                "reasoning": "Optimizes for speed with large datasets"
            })
        
        # Quality-based explanations
        if profile.get("quality_profile") == "poor":
            explanations.append({
                "factor": "data_quality",
                "observation": "High missing values detected",
                "decision": "Applied robust preprocessing",
                "reasoning": "Ensures model reliability with poor data"
            })
        
        # Model-based explanations
        if "xgboost" in strategy.get("models", []):
            explanations.append({
                "factor": "model_selection",
                "observation": "XGBoost selected",
                "decision": "Primary gradient boosting model",
                "reasoning": "Best performance for tabular data"
            })
        
        return {
            "primary_factors": [exp["factor"] for exp in explanations],
            "explanations": explanations,
            "confidence": strategy.get("confidence", 0.8)
        }
    
    def _get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution results"""
        return {
            "status": "success" if self.execution_results else "failed",
            "models_tried": self.execution_results.get("models_tried", []) if self.execution_results else [],
            "preprocessing_applied": self.selected_strategy.get("preprocessing", []) if self.selected_strategy else [],
            "feature_engineering_applied": self.selected_strategy.get("feature_engineering", []) if self.selected_strategy else []
        }
    
    def _learn_from_results(self):
        """Learn from this execution for future improvements"""
        if self.dataset_profile and self.selected_strategy and self.execution_results:
            # Store in knowledge base for future learning
            self.strategy_selector.learn_from_result(
                self.dataset_profile,
                self.selected_strategy,
                self.execution_results.get("best_score", 0)
            )
            logger.info("🧠 Learned from this execution for future improvements")
    
    def __repr__(self):
        return f"UnifiedAutoML(mode='{self.mode}', strategy='{self.strategy}')"


# Convenience function for quick usage
def create_automl(mode: str = "adaptive", **kwargs) -> UnifiedAutoML:
    """
    Convenience function to create UnifiedAutoML instance
    
    Args:
        mode: Engine mode
        **kwargs: Additional arguments
        
    Returns:
        UnifiedAutoML instance
    """
    return UnifiedAutoML(mode=mode, **kwargs)


# Quick fit function for rapid prototyping
def quick_fit(X, y, mode: str = "adaptive", max_time: float = 60) -> UnifiedAutoML:
    """
    Quick fit function for rapid prototyping
    
    Args:
        X: Feature data
        y: Target data  
        mode: Engine mode
        max_time: Maximum time in seconds
        
    Returns:
        Trained UnifiedAutoML instance
    """
    automl = UnifiedAutoML(mode=mode, max_time=max_time)
    return automl.fit(X, y)
