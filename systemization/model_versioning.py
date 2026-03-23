"""
Model Versioning System
Production model management with metadata tracking
"""

import os
import json
import pickle
import datetime
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ModelVersioning:
    """
    Comprehensive model versioning and management system
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 metadata_file: str = "models/metadata.json",
                 max_versions: int = 10):
        """
        Initialize model versioning system
        
        Args:
            models_dir: Directory to store model files
            metadata_file: Path to metadata file
            max_versions: Maximum number of versions to keep
        """
        self.models_dir = Path(models_dir)
        self.metadata_file = Path(metadata_file)
        self.max_versions = max_versions
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_file.parent.mkdir(exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
        
        logger.info(f"Model versioning system initialized: {self.models_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            else:
                return {"versions": [], "latest_version": None}
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return {"versions": [], "latest_version": None}
    
    def _save_metadata(self) -> None:
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def _generate_version_id(self, model_name: str) -> str:
        """Generate unique version ID"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_obj = hashlib.md5(f"{model_name}{timestamp}".encode())
        hash_suffix = hash_obj.hexdigest()[:6]
        return f"v{timestamp}_{hash_suffix}"
    
    def _calculate_model_hash(self, model: BaseEstimator) -> str:
        """Calculate hash of model for integrity checking"""
        try:
            # Serialize model to bytes
            model_bytes = pickle.dumps(model)
            return hashlib.sha256(model_bytes).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate model hash: {e}")
            return "unknown"
    
    def save_model(self, 
                   model: BaseEstimator,
                   model_name: str,
                   metrics: Dict[str, float],
                   dataset_info: Dict[str, Any],
                   task_type: str = "classification",
                   description: str = "",
                   tags: List[str] = None) -> str:
        """
        Save model with comprehensive metadata
        
        Args:
            model: Trained model
            model_name: Name of the model
            metrics: Performance metrics
            dataset_info: Dataset characteristics
            task_type: Type of ML task
            description: Model description
            tags: List of tags for categorization
            
        Returns:
            Version ID
        """
        try:
            # Generate version ID
            version_id = self._generate_version_id(model_name)
            
            # Create model file path
            model_file = self.models_dir / f"{version_id}.pkl"
            
            # Calculate model hash
            model_hash = self._calculate_model_hash(model)
            
            # Save model
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # Create version metadata
            version_metadata = {
                "version": version_id,
                "model_name": model_name,
                "task_type": task_type,
                "description": description,
                "tags": tags or [],
                "metrics": metrics,
                "dataset_info": dataset_info,
                "model_hash": model_hash,
                "file_path": str(model_file),
                "created_at": datetime.datetime.now().isoformat(),
                "file_size": model_file.stat().st_size
            }
            
            # Add to metadata
            self.metadata["versions"].append(version_metadata)
            self.metadata["latest_version"] = version_id
            
            # Clean up old versions if needed
            self._cleanup_old_versions(model_name)
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Model saved: {model_name} v{version_id}")
            logger.info(f"Metrics: {metrics}")
            
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, version_id: str = None, model_name: str = None) -> tuple:
        """
        Load model from version
        
        Args:
            version_id: Specific version ID (if None, loads latest)
            model_name: Model name (used if version_id is None)
            
        Returns:
            Tuple of (model, metadata)
        """
        try:
            # Find version to load
            if version_id is None:
                if model_name:
                    # Find latest version for this model
                    model_versions = [v for v in self.metadata["versions"] 
                                    if v["model_name"] == model_name]
                    if not model_versions:
                        raise ValueError(f"No versions found for model: {model_name}")
                    
                    # Sort by creation time and get latest
                    latest_version = max(model_versions, 
                                       key=lambda x: x["created_at"])
                    version_id = latest_version["version"]
                else:
                    # Load overall latest version
                    version_id = self.metadata.get("latest_version")
                    if version_id is None:
                        raise ValueError("No models available")
            
            # Find version metadata
            version_metadata = None
            for version in self.metadata["versions"]:
                if version["version"] == version_id:
                    version_metadata = version
                    break
            
            if version_metadata is None:
                raise ValueError(f"Version not found: {version_id}")
            
            # Load model
            model_file = Path(version_metadata["file_path"])
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Verify model hash if available
            if "model_hash" in version_metadata:
                current_hash = self._calculate_model_hash(model)
                if current_hash != version_metadata["model_hash"]:
                    logger.warning("Model hash mismatch - file may be corrupted")
            
            logger.info(f"Model loaded: {version_metadata['model_name']} v{version_id}")
            
            return model, version_metadata
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def list_versions(self, model_name: str = None) -> List[Dict[str, Any]]:
        """
        List all versions or versions for specific model
        
        Args:
            model_name: Filter by model name
            
        Returns:
            List of version metadata
        """
        versions = self.metadata["versions"]
        
        if model_name:
            versions = [v for v in versions if v["model_name"] == model_name]
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        
        return versions
    
    def get_version_info(self, version_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific version
        
        Args:
            version_id: Version ID
            
        Returns:
            Version metadata
        """
        for version in self.metadata["versions"]:
            if version["version"] == version_id:
                return version.copy()
        
        raise ValueError(f"Version not found: {version_id}")
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        Compare two versions
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            Comparison results
        """
        try:
            v1_info = self.get_version_info(version_id1)
            v2_info = self.get_version_info(version_id2)
            
            comparison = {
                "version1": v1_info["version"],
                "version2": v2_info["version"],
                "model_name": v1_info["model_name"],
                "comparison_date": datetime.datetime.now().isoformat(),
                "metrics_comparison": {},
                "metadata_comparison": {}
            }
            
            # Compare metrics
            metrics1 = v1_info.get("metrics", {})
            metrics2 = v2_info.get("metrics", {})
            
            all_metrics = set(metrics1.keys()) | set(metrics2.keys())
            
            for metric in all_metrics:
                val1 = metrics1.get(metric, None)
                val2 = metrics2.get(metric, None)
                
                if val1 is not None and val2 is not None:
                    diff = val2 - val1
                    pct_change = (diff / val1 * 100) if val1 != 0 else 0
                    
                    comparison["metrics_comparison"][metric] = {
                        "version1": val1,
                        "version2": val2,
                        "difference": diff,
                        "percent_change": pct_change,
                        "improvement": diff > 0 if "accuracy" in metric.lower() or "r2" in metric.lower() else diff < 0
                    }
            
            # Compare metadata
            metadata_fields = ["task_type", "dataset_info", "tags", "description"]
            
            for field in metadata_fields:
                val1 = v1_info.get(field)
                val2 = v2_info.get(field)
                
                comparison["metadata_comparison"][field] = {
                    "version1": val1,
                    "version2": v2,
                    "same": val1 == val2
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            raise
    
    def delete_version(self, version_id: str) -> bool:
        """
        Delete a specific version
        
        Args:
            version_id: Version ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            # Find version metadata
            version_metadata = None
            version_index = -1
            
            for i, version in enumerate(self.metadata["versions"]):
                if version["version"] == version_id:
                    version_metadata = version
                    version_index = i
                    break
            
            if version_metadata is None:
                raise ValueError(f"Version not found: {version_id}")
            
            # Delete model file
            model_file = Path(version_metadata["file_path"])
            if model_file.exists():
                model_file.unlink()
            
            # Remove from metadata
            self.metadata["versions"].pop(version_index)
            
            # Update latest version if needed
            if self.metadata.get("latest_version") == version_id:
                if self.metadata["versions"]:
                    # Set latest to most recent
                    latest = max(self.metadata["versions"], 
                              key=lambda x: x["created_at"])
                    self.metadata["latest_version"] = latest["version"]
                else:
                    self.metadata["latest_version"] = None
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Version deleted: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete version: {e}")
            return False
    
    def _cleanup_old_versions(self, model_name: str) -> None:
        """Clean up old versions, keeping only the most recent ones"""
        try:
            # Get versions for this model
            model_versions = [v for v in self.metadata["versions"] 
                            if v["model_name"] == model_name]
            
            # Sort by creation time (newest first)
            model_versions.sort(key=lambda x: x["created_at"], reverse=True)
            
            # Keep only the most recent versions
            versions_to_keep = model_versions[:self.max_versions]
            versions_to_delete = model_versions[self.max_versions:]
            
            # Delete old versions
            for version in versions_to_delete:
                self.delete_version(version["version"])
            
            if versions_to_delete:
                logger.info(f"Cleaned up {len(versions_to_delete)} old versions for {model_name}")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup old versions: {e}")
    
    def get_best_model(self, 
                      model_name: str = None,
                      metric: str = "accuracy",
                      higher_is_better: bool = True) -> tuple:
        """
        Get the best performing model
        
        Args:
            model_name: Filter by model name
            metric: Metric to optimize
            higher_is_better: Whether higher metric values are better
            
        Returns:
            Tuple of (version_id, score)
        """
        try:
            versions = self.list_versions(model_name)
            
            if not versions:
                raise ValueError("No versions available")
            
            best_version = None
            best_score = None
            
            for version in versions:
                metrics = version.get("metrics", {})
                if metric in metrics:
                    score = metrics[metric]
                    
                    if best_score is None:
                        best_score = score
                        best_version = version
                    elif higher_is_better and score > best_score:
                        best_score = score
                        best_version = version
                    elif not higher_is_better and score < best_score:
                        best_score = score
                        best_version = version
            
            if best_version is None:
                raise ValueError(f"Metric '{metric}' not found in any version")
            
            return best_version["version"], best_score
            
        except Exception as e:
            logger.error(f"Failed to get best model: {e}")
            raise
    
    def export_metadata(self, filepath: str = None) -> Dict[str, Any]:
        """
        Export all metadata
        
        Args:
            filepath: Path to save export file
            
        Returns:
            Complete metadata
        """
        try:
            metadata_copy = {
                "versions": self.metadata["versions"].copy(),
                "latest_version": self.metadata["latest_version"],
                "export_date": datetime.datetime.now().isoformat(),
                "total_versions": len(self.metadata["versions"])
            }
            
            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(metadata_copy, f, indent=2, default=str)
                logger.info(f"Metadata exported to: {filepath}")
            
            return metadata_copy
            
        except Exception as e:
            logger.error(f"Failed to export metadata: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get versioning statistics
        
        Returns:
            Statistics about stored models
        """
        try:
            versions = self.metadata["versions"]
            
            if not versions:
                return {"total_versions": 0, "models": []}
            
            # Calculate statistics
            model_names = [v["model_name"] for v in versions]
            task_types = [v.get("task_type", "unknown") for v in versions]
            
            model_counts = {}
            for name in model_names:
                model_counts[name] = model_counts.get(name, 0) + 1
            
            task_counts = {}
            for task in task_types:
                task_counts[task] = task_counts.get(task, 0) + 1
            
            # File sizes
            total_size = sum(v.get("file_size", 0) for v in versions)
            
            # Date range
            dates = [v.get("created_at") for v in versions if v.get("created_at")]
            if dates:
                oldest = min(dates)
                newest = max(dates)
            else:
                oldest = newest = None
            
            return {
                "total_versions": len(versions),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "model_counts": model_counts,
                "task_type_counts": task_counts,
                "oldest_version": oldest,
                "newest_version": newest,
                "latest_version": self.metadata.get("latest_version")
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}


# Convenience functions
def save_model_version(model: BaseEstimator,
                      model_name: str,
                      metrics: Dict[str, float],
                      dataset_info: Dict[str, Any],
                      **kwargs) -> str:
    """Convenience function for saving model version"""
    versioning = ModelVersioning()
    return versioning.save_model(model, model_name, metrics, dataset_info, **kwargs)


def load_model_version(version_id: str = None, model_name: str = None) -> tuple:
    """Convenience function for loading model version"""
    versioning = ModelVersioning()
    return versioning.load_model(version_id, model_name)


def list_model_versions(model_name: str = None) -> List[Dict[str, Any]]:
    """Convenience function for listing model versions"""
    versioning = ModelVersioning()
    return versioning.list_versions(model_name)
