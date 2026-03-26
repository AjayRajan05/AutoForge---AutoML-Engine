"""
🔥 PRODUCTION-GRADE: Failure Memory System
Learn from failures to avoid repeating mistakes
"""

import logging
import json
import time
from typing import Dict, List, Any
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class FailureMemory:
    """
    🔥 PRODUCTION-GRADE: Intelligent failure tracking and avoidance system
    """
    
    def __init__(self, memory_file: str = "core/failure_memory.json"):
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.failure_log = self._load_memory()
        
    def _load_memory(self) -> List[Dict[str, Any]]:
        """Load failure memory from file"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load failure memory: {e}")
        return []
    
    def _save_memory(self) -> None:
        """Save failure memory to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.failure_log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save failure memory: {e}")
    
    def log_failure(self, model_name: str, params: Dict[str, Any], error: str, dataset_info: Dict[str, Any] = None) -> None:
        """
        Log a failure for future learning
        
        Args:
            model_name: Name of the model that failed
            params: Parameters that caused failure
            error: Error message
            dataset_info: Dataset characteristics
        """
        failure_entry = {
            "model": model_name,
            "params": params,
            "error": error,
            "dataset_info": dataset_info or {},
            "timestamp": time.time(),
            "param_hash": self._hash_params(params)
        }
        
        self.failure_log.append(failure_entry)
        self._save_memory()
        
        logger.info(f"Logged failure for {model_name}: {error}")
    
    def _hash_params(self, params: Dict[str, Any]) -> str:
        """Create hash of parameters for similarity detection"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def is_similar_to_past_failure(self, model_name: str, params: Dict[str, Any], similarity_threshold: float = 0.8) -> bool:
        """
        Check if current params are similar to past failures
        
        Args:
            model_name: Current model name
            params: Current parameters
            similarity_threshold: Threshold for similarity (0-1)
            
        Returns:
            True if similar to past failure
        """
        model_failures = [f for f in self.failure_log if f["model"] == model_name]
        
        for failure in model_failures:
            similarity = self._calculate_similarity(params, failure["params"])
            if similarity >= similarity_threshold:
                logger.warning(f"Params similar to past failure: {failure['error']}")
                return True
        
        return False
    
    def _calculate_similarity(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate similarity between two parameter sets"""
        if not params1 or not params2:
            return 0.0
        
        # Get common keys
        common_keys = set(params1.keys()) & set(params2.keys())
        if not common_keys:
            return 0.0
        
        # Calculate similarity for common parameters
        matches = 0
        total = len(common_keys)
        
        for key in common_keys:
            if params1[key] == params2[key]:
                matches += 1
            elif isinstance(params1[key], (int, float)) and isinstance(params2[key], (int, float)):
                # For numeric values, check if they're close
                if abs(params1[key] - params2[key]) / max(abs(params1[key]), abs(params2[key]), 1) < 0.1:
                    matches += 1
        
        return matches / total
    
    def get_safe_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get safe parameters based on failure history
        
        Args:
            model_name: Model name
            params: Original parameters
            
        Returns:
            Safe parameters avoiding past failures
        """
        if self.is_similar_to_past_failure(model_name, params):
            # Find safe alternatives
            model_failures = [f for f in self.failure_log if f["model"] == model_name]
            
            # Get most common safe values
            safe_params = params.copy()
            
            for failure in model_failures:
                for key, value in failure["params"].items():
                    if key in safe_params and safe_params[key] == value:
                        # Try to find a safe alternative
                        safe_params[key] = self._get_safe_alternative(key, value, model_name)
            
            logger.info(f"Adjusted params to avoid past failures for {model_name}")
            return safe_params
        
        return params
    
    def _get_safe_alternative(self, param_name: str, bad_value: Any, model_name: str) -> Any:
        """Get safe alternative parameter value"""
        # Default safe alternatives based on parameter type
        if param_name == "solver":
            safe_options = ["lbfgs", "liblinear", "saga"]
            return safe_options[0] if bad_value in safe_options else "lbfgs"
        
        elif param_name == "optimizer":
            safe_options = ["adam", "sgd"]
            return safe_options[0] if bad_value in safe_options else "adam"
        
        elif param_name == "learning_rate":
            if isinstance(bad_value, (int, float)):
                return 0.001  # Safe default learning rate
        
        elif param_name == "C":
            if isinstance(bad_value, (int, float)):
                return 1.0  # Safe default C value
        
        # Return bad_value if no safe alternative found
        return bad_value
    
    def get_failure_stats(self) -> Dict[str, Any]:
        """Get statistics about failures"""
        if not self.failure_log:
            return {"total_failures": 0}
        
        model_stats = {}
        for failure in self.failure_log:
            model = failure["model"]
            if model not in model_stats:
                model_stats[model] = {"count": 0, "errors": []}
            model_stats[model]["count"] += 1
            model_stats[model]["errors"].append(failure["error"])
        
        return {
            "total_failures": len(self.failure_log),
            "models": model_stats,
            "most_common_errors": self._get_most_common_errors()
        }
    
    def _get_most_common_errors(self) -> List[str]:
        """Get most common error messages"""
        error_counts = {}
        for failure in self.failure_log:
            error = failure["error"]
            error_counts[error] = error_counts.get(error, 0) + 1
        
        return sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]


# Global failure memory instance
failure_memory = FailureMemory()
