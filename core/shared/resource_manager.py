"""
⚡ Adaptive Resource Manager
Intelligent CPU, memory, and time resource management
"""

import logging
import psutil
import time
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits configuration"""
    max_memory_mb: float = 4096  # 4GB default
    max_cpu_percent: float = 80.0
    max_time_seconds: Optional[float] = None
    memory_safety_margin: float = 0.8  # Use 80% of available


class ResourceManager:
    """
    Adaptive Resource Manager
    
    Monitors and manages system resources during AutoML execution.
    Adapts strategy based on available resources.
    """
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.start_time = None
        self.monitoring_active = False
        self.monitor_thread = None
        self.resource_history = []
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.start_time = time.time()
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        logger.info("⚡ Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("⚡ Resource monitoring stopped")
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        return {
            "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "elapsed_time": time.time() - self.start_time if self.start_time else 0
        }
    
    def should_adapt(self) -> Dict[str, Any]:
        """Check if strategy adaptation is needed"""
        current = self.get_current_usage()
        
        adaptations = {}
        
        # Memory pressure
        if current["memory_mb"] > self.limits.max_memory_mb * self.limits.memory_safety_margin:
            adaptations["memory_pressure"] = True
            adaptations["suggested_actions"] = ["reduce_features", "use_sampling", "simplify_models"]
        
        # Time pressure
        if self.limits.max_time_seconds and current["elapsed_time"] > self.limits.max_time_seconds * 0.8:
            adaptations["time_pressure"] = True
            adaptations["suggested_actions"] = ["reduce_trials", "use_fast_models", "early_stopping"]
        
        # CPU pressure
        if current["cpu_percent"] > self.limits.max_cpu_percent:
            adaptations["cpu_pressure"] = True
            adaptations["suggested_actions"] = ["reduce_parallelism", "use_efficient_models"]
        
        return adaptations
    
    def get_optimal_config(self, dataset_size: int) -> Dict[str, Any]:
        """Get optimal configuration based on resources and dataset size"""
        available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
        
        config = {
            "max_samples": min(dataset_size, available_memory * 10),  # Rough estimate
            "batch_size": self._calculate_optimal_batch_size(available_memory),
            "n_jobs": self._calculate_optimal_jobs(),
            "use_incremental": dataset_size > 100000 or available_memory < 2048
        }
        
        return config
    
    def _monitor_resources(self):
        """Monitor resources in background thread"""
        while self.monitoring_active:
            try:
                current = self.get_current_usage()
                self.resource_history.append({
                    "timestamp": time.time(),
                    **current
                })
                
                # Keep only last 100 entries
                if len(self.resource_history) > 100:
                    self.resource_history = self.resource_history[-100:]
                
                # Check for critical conditions
                if current["memory_percent"] > 90:
                    logger.warning("⚠️ High memory usage detected")
                
                if current["cpu_percent"] > 95:
                    logger.warning("⚠️ High CPU usage detected")
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
    
    def _calculate_optimal_batch_size(self, available_memory_mb: float) -> int:
        """Calculate optimal batch size based on available memory"""
        # Rough estimate: 1MB per 1000 samples with 10 features
        return max(1000, int(available_memory_mb * 100))
    
    def _calculate_optimal_jobs(self) -> int:
        """Calculate optimal number of parallel jobs"""
        cpu_count = psutil.cpu_count()
        # Leave one CPU free for system
        return max(1, cpu_count - 1)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage"""
        if not self.resource_history:
            return {"error": "No resource history available"}
        
        memory_usage = [entry["memory_mb"] for entry in self.resource_history]
        cpu_usage = [entry["cpu_percent"] for entry in self.resource_history]
        
        return {
            "peak_memory_mb": max(memory_usage),
            "avg_memory_mb": sum(memory_usage) / len(memory_usage),
            "peak_cpu_percent": max(cpu_usage),
            "avg_cpu_percent": sum(cpu_usage) / len(cpu_usage),
            "total_time": self.resource_history[-1]["elapsed_time"] if self.resource_history else 0,
            "samples_monitored": len(self.resource_history)
        }
