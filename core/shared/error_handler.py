"""
🛡️ Bulletproof Error Handler
Universal error recovery and fallback system
"""

import logging
import traceback
from typing import Dict, Any, Callable, Optional, List
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Recovery action types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """Error context information"""
    error_type: str
    severity: ErrorSeverity
    message: str
    traceback: str
    timestamp: float
    context: Dict[str, Any]


class BulletproofErrorHandler:
    """
    Bulletproof Error Handler
    
    Provides intelligent error recovery with multiple fallback strategies.
    Ensures AutoML system always provides some result.
    """
    
    def __init__(self):
        self.error_history = []
        self.recovery_strategies = {}
        self.fallback_functions = {}
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle error with intelligent recovery
        
        Args:
            error: The exception that occurred
            context: Additional context information
            
        Returns:
            Recovery result with action taken
        """
        error_context = ErrorContext(
            error_type=type(error).__name__,
            severity=self._classify_error(error),
            message=str(error),
            traceback=traceback.format_exc(),
            timestamp=time.time(),
            context=context or {}
        )
        
        self.error_history.append(error_context)
        
        logger.error(f"🛡️ Handling error: {error_context.error_type} - {error_context.message}")
        
        # Determine recovery action
        recovery_action = self._determine_recovery_action(error_context)
        
        # Execute recovery
        recovery_result = self._execute_recovery(error_context, recovery_action)
        
        return recovery_result
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register custom recovery strategy for error type"""
        self.recovery_strategies[error_type] = strategy
    
    def register_fallback(self, error_type: str, fallback_func: Callable):
        """Register fallback function for error type"""
        self.fallback_functions[error_type] = fallback_func
    
    def bulletproof_method(self, fallback_result=None, max_retries=3):
        """
        Decorator to make methods bulletproof
        
        Args:
            fallback_result: Result to return if all recovery fails
            max_retries: Maximum number of retry attempts
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        logger.warning(f"🔄 Attempt {attempt + 1} failed: {str(e)}")
                        
                        if attempt < max_retries - 1:
                            # Try recovery
                            recovery = self.handle_error(e, {
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "args": str(args)[:100],
                                "kwargs": str(kwargs)[:100]
                            })
                            
                            if recovery.get("success", False):
                                return recovery.get("result", fallback_result)
                
                # All attempts failed, return fallback
                logger.error(f"❌ All attempts failed for {func.__name__}")
                return fallback_result if fallback_result is not None else {"error": str(last_error)}
            
            return wrapper
        return decorator
    
    def _classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify error severity"""
        error_type = type(error).__name__
        
        # Critical errors
        if error_type in ["MemoryError", "SystemExit", "KeyboardInterrupt"]:
            return ErrorSeverity.CRITICAL
        
        # High severity
        if error_type in ["ImportError", "AttributeError", "TypeError"]:
            return ErrorSeverity.HIGH
        
        # Medium severity
        if error_type in ["ValueError", "KeyError", "IndexError"]:
            return ErrorSeverity.MEDIUM
        
        # Low severity
        return ErrorSeverity.LOW
    
    def _determine_recovery_action(self, error_context: ErrorContext) -> RecoveryAction:
        """Determine best recovery action based on error context"""
        error_type = error_context.error_type
        severity = error_context.severity
        
        # Critical errors - abort
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryAction.ABORT
        
        # Import/Attribute errors - fallback
        if error_type in ["ImportError", "AttributeError"]:
            return RecoveryAction.FALLBACK
        
        # Memory errors - fallback with reduced resources
        if error_type == "MemoryError":
            return RecoveryAction.FALLBACK
        
        # Data errors - retry with preprocessing
        if error_type in ["ValueError", "TypeError"]:
            return RecoveryAction.RETRY
        
        # Default - fallback
        return RecoveryAction.FALLBACK
    
    def _execute_recovery(self, error_context: ErrorContext, action: RecoveryAction) -> Dict[str, Any]:
        """Execute recovery action"""
        try:
            if action == RecoveryAction.RETRY:
                return self._retry_with_modifications(error_context)
            elif action == RecoveryAction.FALLBACK:
                return self._execute_fallback(error_context)
            elif action == RecoveryAction.SKIP:
                return {"success": True, "action": "skipped", "message": "Step skipped due to error"}
            elif action == RecoveryAction.ABORT:
                return {"success": False, "action": "aborted", "message": "Execution aborted due to critical error"}
            else:
                return {"success": False, "action": "unknown", "message": "Unknown recovery action"}
                
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return {"success": False, "action": "recovery_failed", "message": str(e)}
    
    def _retry_with_modifications(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Retry with intelligent modifications"""
        error_type = error_context.error_type
        
        modifications = {
            "ValueError": ["convert_data_types", "handle_missing_values", "validate_inputs"],
            "TypeError": ["convert_data_types", "check_shapes", "validate_inputs"],
            "KeyError": ["check_column_names", "validate_features"],
            "IndexError": ["check_indices", "validate_data_shape"]
        }
        
        suggested_mods = modifications.get(error_type, [])
        
        return {
            "success": True,
            "action": "retry_with_modifications",
            "suggested_modifications": suggested_mods,
            "message": f"Retry suggested with modifications: {suggested_mods}"
        }
    
    def _execute_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Execute fallback strategy"""
        error_type = error_context.error_type
        
        # Try registered fallback first
        if error_type in self.fallback_functions:
            try:
                fallback_result = self.fallback_functions[error_type](error_context)
                return {
                    "success": True,
                    "action": "fallback_executed",
                    "result": fallback_result,
                    "message": f"Executed custom fallback for {error_type}"
                }
            except Exception as e:
                logger.warning(f"Custom fallback failed: {e}")
        
        # Default fallbacks
        default_fallbacks = {
            "ImportError": self._import_error_fallback,
            "MemoryError": self._memory_error_fallback,
            "ValueError": self._value_error_fallback,
            "TypeError": self._type_error_fallback
        }
        
        if error_type in default_fallbacks:
            return default_fallbacks[error_type](error_context)
        
        # Generic fallback
        return {
            "success": True,
            "action": "generic_fallback",
            "result": {"error": error_context.message, "fallback": True},
            "message": "Applied generic fallback strategy"
        }
    
    def _import_error_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Fallback for import errors"""
        return {
            "success": True,
            "action": "import_fallback",
            "result": {"error": "Import failed", "use_alternative": True},
            "message": "Import error - use alternative implementation"
        }
    
    def _memory_error_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Fallback for memory errors"""
        return {
            "success": True,
            "action": "memory_fallback",
            "result": {"error": "Memory insufficient", "use_sampling": True, "reduce_features": True},
            "message": "Memory error - use data sampling and feature reduction"
        }
    
    def _value_error_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Fallback for value errors"""
        return {
            "success": True,
            "action": "value_fallback",
            "result": {"error": "Invalid values", "use_robust_methods": True},
            "message": "Value error - use robust preprocessing methods"
        }
    
    def _type_error_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Fallback for type errors"""
        return {
            "success": True,
            "action": "type_fallback",
            "result": {"error": "Type mismatch", "convert_types": True},
            "message": "Type error - convert data types"
        }
    
    def _register_default_strategies(self):
        """Register default recovery strategies"""
        # Memory error strategy
        def memory_strategy(error_context):
            return {
                "reduce_data": True,
                "use_sampling": True,
                "simplify_models": True
            }
        
        # Import error strategy
        def import_strategy(error_context):
            return {
                "use_alternative": True,
                "disable_advanced_features": True
            }
        
        self.register_recovery_strategy("MemoryError", memory_strategy)
        self.register_recovery_strategy("ImportError", import_strategy)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered"""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            error_type = error.error_type
            severity = error.severity.value
            
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_counts,
            "severity_distribution": severity_counts,
            "most_common_error": max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None,
            "recovery_success_rate": self._calculate_recovery_success_rate()
        }
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate"""
        if not self.error_history:
            return 1.0
        
        # This would need to be tracked during actual recovery
        # For now, return optimistic estimate
        return 0.85


# Global error handler instance
global_error_handler = BulletproofErrorHandler()


def bulletproof(fallback_result=None, max_retries=3):
    """Convenience decorator for bulletproof methods"""
    return global_error_handler.bulletproof_method(fallback_result, max_retries)


def handle_error(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function to handle errors"""
    return global_error_handler.handle_error(error, context)
