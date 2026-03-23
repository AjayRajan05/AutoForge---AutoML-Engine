"""
Lightweight Monitoring System
Accuracy tracking and data drift detection
"""

import numpy as np
import pandas as pd
import logging
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict, deque
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
import warnings

logger = logging.getLogger(__name__)


class LightweightMonitor:
    """
    Lightweight monitoring for accuracy and data drift
    """
    
    def __init__(self, 
                 monitor_dir: str = "monitoring",
                 max_history: int = 1000,
                 drift_threshold: float = 0.1,
                 accuracy_window: int = 50):
        """
        Initialize monitoring system
        
        Args:
            monitor_dir: Directory to store monitoring data
            max_history: Maximum number of records to keep in memory
            drift_threshold: Threshold for data drift detection
            accuracy_window: Window size for accuracy tracking
        """
        self.monitor_dir = Path(monitor_dir)
        self.max_history = max_history
        self.drift_threshold = drift_threshold
        self.accuracy_window = accuracy_window
        
        # Create directories
        self.monitor_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.accuracy_history = deque(maxlen=max_history)
        self.data_profiles = deque(maxlen=max_history)
        self.alerts = deque(maxlen=100)
        
        # Load existing data
        self._load_monitoring_data()
        
        logger.info(f"Lightweight monitoring initialized: {self.monitor_dir}")
    
    def _load_monitoring_data(self) -> None:
        """Load existing monitoring data"""
        try:
            accuracy_file = self.monitor_dir / "accuracy_history.json"
            if accuracy_file.exists():
                with open(accuracy_file, 'r') as f:
                    data = json.load(f)
                    self.accuracy_history = deque(data, maxlen=self.max_history)
            
            profile_file = self.monitor_dir / "data_profiles.json"
            if profile_file.exists():
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                    self.data_profiles = deque(data, maxlen=self.max_history)
            
            alerts_file = self.monitor_dir / "alerts.json"
            if alerts_file.exists():
                with open(alerts_file, 'r') as f:
                    data = json.load(f)
                    self.alerts = deque(data, maxlen=100)
                    
        except Exception as e:
            logger.warning(f"Failed to load monitoring data: {e}")
    
    def _save_monitoring_data(self) -> None:
        """Save monitoring data to files"""
        try:
            # Save accuracy history
            accuracy_file = self.monitor_dir / "accuracy_history.json"
            with open(accuracy_file, 'w') as f:
                json.dump(list(self.accuracy_history), f, indent=2, default=str)
            
            # Save data profiles
            profile_file = self.monitor_dir / "data_profiles.json"
            with open(profile_file, 'w') as f:
                json.dump(list(self.data_profiles), f, indent=2, default=str)
            
            # Save alerts
            alerts_file = self.monitor_dir / "alerts.json"
            with open(alerts_file, 'w') as f:
                json.dump(list(self.alerts), f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save monitoring data: {e}")
    
    def log_prediction(self, 
                      model_name: str,
                      X: Union[np.ndarray, pd.DataFrame],
                      y_true: Union[np.ndarray, pd.Series],
                      y_pred: Union[np.ndarray, pd.Series],
                      timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Log prediction and calculate accuracy
        
        Args:
            model_name: Name of the model
            X: Input features
            y_true: True labels
            y_pred: Predicted labels
            timestamp: Timestamp (auto-generated if None)
            
        Returns:
            Accuracy metrics
        """
        try:
            if timestamp is None:
                timestamp = datetime.datetime.now().isoformat()
            
            # Calculate metrics based on task type
            metrics = self._calculate_metrics(y_true, y_pred)
            
            # Create accuracy record
            accuracy_record = {
                "timestamp": timestamp,
                "model_name": model_name,
                "metrics": metrics,
                "sample_size": len(y_true)
            }
            
            # Add to history
            self.accuracy_history.append(accuracy_record)
            
            # Check for accuracy alerts
            self._check_accuracy_alerts(model_name, metrics)
            
            # Save data
            self._save_monitoring_data()
            
            logger.info(f"Prediction logged: {model_name} - {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            raise
    
    def log_data_profile(self, 
                        X: Union[np.ndarray, pd.DataFrame],
                        dataset_name: str = "production",
                        timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Log data profile for drift detection
        
        Args:
            X: Input features
            dataset_name: Name of the dataset
            timestamp: Timestamp (auto-generated if None)
            
        Returns:
            Data profile
        """
        try:
            if timestamp is None:
                timestamp = datetime.datetime.now().isoformat()
            
            # Calculate data profile
            profile = self._calculate_data_profile(X)
            
            # Create profile record
            profile_record = {
                "timestamp": timestamp,
                "dataset_name": dataset_name,
                "profile": profile,
                "sample_size": len(X)
            }
            
            # Add to history
            self.data_profiles.append(profile_record)
            
            # Check for data drift
            self._check_data_drift(dataset_name, profile)
            
            # Save data
            self._save_monitoring_data()
            
            logger.info(f"Data profile logged: {dataset_name} - {len(X)} samples")
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to log data profile: {e}")
            raise
    
    def _calculate_metrics(self, 
                          y_true: Union[np.ndarray, pd.Series],
                          y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            # Convert to numpy arrays
            if hasattr(y_true, 'values'):
                y_true = y_true.values
            if hasattr(y_pred, 'values'):
                y_pred = y_pred.values
            
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Determine task type
            unique_targets = len(np.unique(y_true))
            is_classification = unique_targets <= 20 and np.issubdtype(y_true.dtype, np.integer)
            
            metrics = {}
            
            if is_classification:
                metrics["accuracy"] = accuracy_score(y_true, y_pred)
                metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted')
                metrics["task_type"] = "classification"
            else:
                metrics["mse"] = mean_squared_error(y_true, y_pred)
                metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics["r2_score"] = r2_score(y_true, y_pred)
                metrics["task_type"] = "regression"
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_data_profile(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate data profile for drift detection"""
        try:
            # Convert to DataFrame if needed
            if isinstance(X, np.ndarray):
                df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            else:
                df = X.copy()
            
            profile = {
                "n_samples": len(df),
                "n_features": df.shape[1],
                "missing_ratio": df.isnull().sum().sum() / df.size,
                "feature_stats": {}
            }
            
            # Calculate statistics for each feature
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    # Numeric features
                    stats = {
                        "mean": df[col].mean(),
                        "std": df[col].std(),
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "q25": df[col].quantile(0.25),
                        "q50": df[col].quantile(0.50),
                        "q75": df[col].quantile(0.75),
                        "missing_ratio": df[col].isnull().sum() / len(df)
                    }
                else:
                    # Categorical features
                    stats = {
                        "unique_count": df[col].nunique(),
                        "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                        "missing_ratio": df[col].isnull().sum() / len(df)
                    }
                
                profile["feature_stats"][col] = stats
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to calculate data profile: {e}")
            return {"error": str(e)}
    
    def _check_accuracy_alerts(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Check for accuracy alerts"""
        try:
            # Get recent accuracy history for this model
            recent_history = [record for record in self.accuracy_history 
                            if record.get("model_name") == model_name][-self.accuracy_window:]
            
            if len(recent_history) < 2:
                return
            
            # Check for accuracy degradation
            task_type = metrics.get("task_type", "classification")
            
            if task_type == "classification":
                current_acc = metrics.get("accuracy", 0)
                recent_accs = [record["metrics"].get("accuracy", 0) for record in recent_history]
                
                if len(recent_accs) > 1:
                    avg_recent = np.mean(recent_accs[:-1])  # Exclude current
                    if current_acc < avg_recent - 0.1:  # 10% drop
                        self._create_alert("accuracy_degradation", model_name, {
                            "current_accuracy": current_acc,
                            "recent_average": avg_recent,
                            "drop_percentage": (avg_recent - current_acc) / avg_recent * 100
                        })
            
            elif task_type == "regression":
                current_r2 = metrics.get("r2_score", 0)
                recent_r2s = [record["metrics"].get("r2_score", 0) for record in recent_history]
                
                if len(recent_r2s) > 1:
                    avg_recent = np.mean(recent_r2s[:-1])
                    if current_r2 < avg_recent - 0.1:  # 0.1 drop in R2
                        self._create_alert("r2_degradation", model_name, {
                            "current_r2": current_r2,
                            "recent_average": avg_recent,
                            "drop_amount": avg_recent - current_r2
                        })
            
        except Exception as e:
            logger.warning(f"Failed to check accuracy alerts: {e}")
    
    def _check_data_drift(self, dataset_name: str, current_profile: Dict[str, Any]) -> None:
        """Check for data drift"""
        try:
            # Get previous profiles for this dataset
            previous_profiles = [record for record in self.data_profiles 
                              if record.get("dataset_name") == dataset_name]
            
            if len(previous_profiles) < 2:
                return
            
            # Compare with most recent previous profile
            previous_profile = previous_profiles[-2]["profile"]
            
            drift_detected = False
            drift_details = {}
            
            # Compare feature statistics
            for feature_name, current_stats in current_profile.get("feature_stats", {}).items():
                if feature_name in previous_profile.get("feature_stats", {}):
                    prev_stats = previous_profile["feature_stats"][feature_name]
                    
                    # Compare numeric features
                    if "mean" in current_stats and "mean" in prev_stats:
                        mean_diff = abs(current_stats["mean"] - prev_stats["mean"])
                        std_diff = abs(current_stats["std"] - prev_stats["std"])
                        
                        # Check for significant drift
                        if prev_stats["std"] > 0:
                            mean_z_score = mean_diff / prev_stats["std"]
                            if mean_z_score > 2:  # 2 standard deviations
                                drift_detected = True
                                drift_details[feature_name] = {
                                    "type": "mean_drift",
                                    "z_score": mean_z_score,
                                    "current_mean": current_stats["mean"],
                                    "previous_mean": prev_stats["mean"]
                                }
                    
                    # Compare categorical features
                    elif "unique_count" in current_stats and "unique_count" in prev_stats:
                        unique_diff = abs(current_stats["unique_count"] - prev_stats["unique_count"])
                        if unique_diff > prev_stats["unique_count"] * 0.2:  # 20% change
                            drift_detected = True
                            drift_details[feature_name] = {
                                "type": "cardinality_drift",
                                "current_unique": current_stats["unique_count"],
                                "previous_unique": prev_stats["unique_count"],
                                "change_percentage": unique_diff / prev_stats["unique_count"] * 100
                            }
            
            # Create alert if drift detected
            if drift_detected:
                self._create_alert("data_drift", dataset_name, {
                    "drift_details": drift_details,
                    "drift_threshold": self.drift_threshold,
                    "current_sample_size": current_profile.get("n_samples"),
                    "previous_sample_size": previous_profile.get("n_samples")
                })
            
        except Exception as e:
            logger.warning(f"Failed to check data drift: {e}")
    
    def _create_alert(self, alert_type: str, source: str, details: Dict[str, Any]) -> None:
        """Create an alert"""
        alert = {
            "timestamp": datetime.datetime.now().isoformat(),
            "alert_type": alert_type,
            "source": source,
            "details": details,
            "severity": self._get_alert_severity(alert_type)
        }
        
        self.alerts.append(alert)
        
        logger.warning(f"Alert created: {alert_type} - {source}")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity level"""
        severity_map = {
            "accuracy_degradation": "high",
            "r2_degradation": "high",
            "data_drift": "medium",
            "missing_data": "low"
        }
        return severity_map.get(alert_type, "medium")
    
    def get_accuracy_trend(self, 
                          model_name: str = None,
                          metric: str = "accuracy",
                          last_n: int = 50) -> List[Dict[str, Any]]:
        """
        Get accuracy trend for a model
        
        Args:
            model_name: Filter by model name
            metric: Metric to track
            last_n: Number of recent records to return
            
        Returns:
            List of accuracy records
        """
        try:
            # Filter records
            records = self.accuracy_history
            
            if model_name:
                records = [r for r in records if r.get("model_name") == model_name]
            
            # Get last N records
            records = records[-last_n:]
            
            # Extract metric values
            trend = []
            for record in records:
                metrics = record.get("metrics", {})
                if metric in metrics:
                    trend.append({
                        "timestamp": record["timestamp"],
                        "value": metrics[metric],
                        "sample_size": record.get("sample_size", 0)
                    })
            
            return trend
            
        except Exception as e:
            logger.error(f"Failed to get accuracy trend: {e}")
            return []
    
    def get_data_drift_report(self, dataset_name: str = None) -> Dict[str, Any]:
        """
        Get data drift report
        
        Args:
            dataset_name: Filter by dataset name
            
        Returns:
            Drift report
        """
        try:
            # Filter profiles
            profiles = list(self.data_profiles)
            
            if dataset_name:
                profiles = [p for p in profiles if p.get("dataset_name") == dataset_name]
            
            if len(profiles) < 2:
                return {"error": "Insufficient data for drift analysis"}
            
            # Get most recent and baseline profiles
            current_profile = profiles[-1]["profile"]
            baseline_profile = profiles[0]["profile"]
            
            # Calculate drift metrics
            drift_report = {
                "dataset_name": dataset_name or "all",
                "baseline_date": profiles[0]["timestamp"],
                "current_date": profiles[-1]["timestamp"],
                "drift_analysis": {},
                "overall_drift_score": 0
            }
            
            drift_scores = []
            
            for feature_name, current_stats in current_profile.get("feature_stats", {}).items():
                if feature_name in baseline_profile.get("feature_stats", {}):
                    prev_stats = baseline_profile["feature_stats"][feature_name]
                    
                    feature_drift = {
                        "feature": feature_name,
                        "drift_metrics": {}
                    }
                    
                    # Numeric feature drift
                    if "mean" in current_stats and "mean" in prev_stats:
                        mean_diff = abs(current_stats["mean"] - prev_stats["mean"])
                        if prev_stats["std"] > 0:
                            z_score = mean_diff / prev_stats["std"]
                            feature_drift["drift_metrics"]["mean_z_score"] = z_score
                            drift_scores.append(abs(z_score))
                    
                    # Categorical feature drift
                    elif "unique_count" in current_stats and "unique_count" in prev_stats:
                        unique_diff = abs(current_stats["unique_count"] - prev_stats["unique_count"])
                        if prev_stats["unique_count"] > 0:
                            pct_change = unique_diff / prev_stats["unique_count"]
                            feature_drift["drift_metrics"]["cardinality_change"] = pct_change
                            drift_scores.append(abs(pct_change))
                    
                    drift_report["drift_analysis"][feature_name] = feature_drift
            
            # Calculate overall drift score
            if drift_scores:
                drift_report["overall_drift_score"] = np.mean(drift_scores)
            
            return drift_report
            
        except Exception as e:
            logger.error(f"Failed to get data drift report: {e}")
            return {"error": str(e)}
    
    def get_alerts(self, 
                   alert_type: str = None,
                   severity: str = None,
                   last_n: int = 20) -> List[Dict[str, Any]]:
        """
        Get alerts
        
        Args:
            alert_type: Filter by alert type
            severity: Filter by severity
            last_n: Number of recent alerts to return
            
        Returns:
            List of alerts
        """
        try:
            alerts = list(self.alerts)
            
            # Apply filters
            if alert_type:
                alerts = [a for a in alerts if a.get("alert_type") == alert_type]
            
            if severity:
                alerts = [a for a in alerts if a.get("severity") == severity]
            
            # Return most recent
            return alerts[-last_n:]
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring summary
        
        Returns:
            Monitoring summary
        """
        try:
            summary = {
                "summary_date": datetime.datetime.now().isoformat(),
                "accuracy_tracking": {
                    "total_predictions": len(self.accuracy_history),
                    "models_monitored": len(set(r.get("model_name", "unknown") for r in self.accuracy_history)),
                    "latest_prediction": self.accuracy_history[-1]["timestamp"] if self.accuracy_history else None
                },
                "data_monitoring": {
                    "total_profiles": len(self.data_profiles),
                    "datasets_monitored": len(set(r.get("dataset_name", "unknown") for r in self.data_profiles)),
                    "latest_profile": self.data_profiles[-1]["timestamp"] if self.data_profiles else None
                },
                "alerts": {
                    "total_alerts": len(self.alerts),
                    "recent_alerts": len([a for a in self.alerts if self._is_recent_alert(a)]),
                    "alert_types": {}
                }
            }
            
            # Count alert types
            for alert in self.alerts:
                alert_type = alert.get("alert_type", "unknown")
                summary["alerts"]["alert_types"][alert_type] = summary["alerts"]["alert_types"].get(alert_type, 0) + 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get monitoring summary: {e}")
            return {"error": str(e)}
    
    def _is_recent_alert(self, alert: Dict[str, Any], hours: int = 24) -> bool:
        """Check if alert is recent"""
        try:
            alert_time = datetime.datetime.fromisoformat(alert["timestamp"])
            current_time = datetime.datetime.now()
            return (current_time - alert_time).total_seconds() < hours * 3600
        except:
            return False
    
    def clear_old_data(self, days_to_keep: int = 30) -> None:
        """Clear old monitoring data"""
        try:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
            cutoff_iso = cutoff_time.isoformat()
            
            # Filter accuracy history
            original_size = len(self.accuracy_history)
            self.accuracy_history = deque(
                [r for r in self.accuracy_history if r.get("timestamp", "") >= cutoff_iso],
                maxlen=self.max_history
            )
            
            # Filter data profiles
            original_profiles = len(self.data_profiles)
            self.data_profiles = deque(
                [r for r in self.data_profiles if r.get("timestamp", "") >= cutoff_iso],
                maxlen=self.max_history
            )
            
            # Filter alerts
            original_alerts = len(self.alerts)
            self.alerts = deque(
                [a for a in self.alerts if a.get("timestamp", "") >= cutoff_iso],
                maxlen=100
            )
            
            # Save cleaned data
            self._save_monitoring_data()
            
            logger.info(f"Cleaned old data: Accuracy {original_size}→{len(self.accuracy_history)}, "
                       f"Profiles {original_profiles}→{len(self.data_profiles)}, "
                       f"Alerts {original_alerts}→{len(self.alerts)}")
            
        except Exception as e:
            logger.error(f"Failed to clear old data: {e}")


# Convenience functions
def log_model_performance(model_name: str, X, y_true, y_pred, **kwargs) -> Dict[str, Any]:
    """Convenience function for logging model performance"""
    monitor = LightweightMonitor()
    return monitor.log_prediction(model_name, X, y_true, y_pred, **kwargs)


def log_data_drift(X, dataset_name: str = "production", **kwargs) -> Dict[str, Any]:
    """Convenience function for logging data profile"""
    monitor = LightweightMonitor()
    return monitor.log_data_profile(X, dataset_name, **kwargs)


def get_monitoring_dashboard() -> Dict[str, Any]:
    """Convenience function for monitoring dashboard"""
    monitor = LightweightMonitor()
    return monitor.get_monitoring_summary()
