"""
📊 Performance Tracker
Real-time performance monitoring and metrics collection
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    timestamp: float
    metric_name: str
    value: float
    unit: str
    context: Dict[str, Any]


@dataclass
class ModelPerformance:
    """Model performance data"""
    model_name: str
    training_time: float
    prediction_time: float
    score: float
    metrics: Dict[str, float]
    parameters: Dict[str, Any]


class PerformanceTracker:
    """
    Performance Tracker
    
    Tracks real-time performance metrics during AutoML execution.
    Provides insights for optimization and debugging.
    """
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.model_performance: List[ModelPerformance] = []
        self.start_time = None
        self.current_phase = None
        self.phase_start_time = None
        
    def start_tracking(self):
        """Start performance tracking"""
        self.start_time = time.time()
        logger.info("📊 Performance tracking started")
    
    def start_phase(self, phase_name: str):
        """Start tracking a new phase"""
        self.current_phase = phase_name
        self.phase_start_time = time.time()
        logger.info(f"📊 Started phase: {phase_name}")
    
    def end_phase(self, phase_name: str = None):
        """End current phase and record duration"""
        if self.phase_start_time and self.current_phase:
            phase = phase_name or self.current_phase
            duration = time.time() - self.phase_start_time
            
            self.record_metric(
                metric_name=f"phase_duration_{phase}",
                value=duration,
                unit="seconds",
                context={"phase": phase}
            )
            
            logger.info(f"📊 Phase {phase} completed in {duration:.2f}s")
            
            self.current_phase = None
            self.phase_start_time = None
    
    def record_metric(self, metric_name: str, value: float, 
                    unit: str = "", context: Dict[str, Any] = None):
        """Record a performance metric"""
        metric = PerformanceMetrics(
            timestamp=time.time(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            context=context or {}
        )
        
        self.metrics_history.append(metric)
        
        # Keep only last 1000 metrics to prevent memory issues
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def record_model_performance(self, model_name: str, training_time: float,
                            prediction_time: float, score: float,
                            metrics: Dict[str, float] = None,
                            parameters: Dict[str, Any] = None):
        """Record model performance"""
        model_perf = ModelPerformance(
            model_name=model_name,
            training_time=training_time,
            prediction_time=prediction_time,
            score=score,
            metrics=metrics or {},
            parameters=parameters or {}
        )
        
        self.model_performance.append(model_perf)
        logger.info(f"📊 Recorded performance for {model_name}: score={score:.3f}")
    
    def get_current_duration(self) -> float:
        """Get duration since tracking started"""
        if self.start_time:
            return time.time() - self.start_time
        return 0
    
    def get_phase_duration(self) -> float:
        """Get duration of current phase"""
        if self.phase_start_time:
            return time.time() - self.phase_start_time
        return 0
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded metrics"""
        if not self.metrics_history:
            return {"error": "No metrics recorded"}
        
        # Group metrics by name
        metric_groups = {}
        for metric in self.metrics_history:
            name = metric.metric_name
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(metric.value)
        
        # Calculate statistics for each metric
        summary = {}
        for name, values in metric_groups.items():
            summary[name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1]
            }
        
        return {
            "total_metrics": len(self.metrics_history),
            "metrics_summary": summary,
            "tracking_duration": self.get_current_duration()
        }
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance"""
        if not self.model_performance:
            return {"error": "No model performance recorded"}
        
        # Sort by score
        sorted_models = sorted(self.model_performance, key=lambda x: x.score, reverse=True)
        
        best_model = sorted_models[0]
        worst_model = sorted_models[-1]
        
        # Calculate averages
        avg_training_time = sum(m.training_time for m in self.model_performance) / len(self.model_performance)
        avg_score = sum(m.score for m in self.model_performance) / len(self.model_performance)
        
        return {
            "total_models": len(self.model_performance),
            "best_model": {
                "name": best_model.model_name,
                "score": best_model.score,
                "training_time": best_model.training_time
            },
            "worst_model": {
                "name": worst_model.model_name,
                "score": worst_model.score,
                "training_time": worst_model.training_time
            },
            "averages": {
                "training_time": avg_training_time,
                "score": avg_score
            },
            "all_models": [
                {
                    "name": m.model_name,
                    "score": m.score,
                    "training_time": m.training_time
                }
                for m in sorted_models
            ]
        }
    
    def get_performance_report(self) -> str:
        """Generate human-readable performance report"""
        report_lines = []
        
        report_lines.append("📊 PERFORMANCE REPORT")
        report_lines.append("=" * 50)
        
        # Overall timing
        total_duration = self.get_current_duration()
        report_lines.append(f"⏱️ Total Duration: {total_duration:.2f} seconds")
        
        # Metrics summary
        metrics_summary = self.get_metrics_summary()
        if "metrics_summary" in metrics_summary:
            report_lines.append("\n📈 Key Metrics:")
            for metric_name, stats in metrics_summary["metrics_summary"].items():
                report_lines.append(f"  • {metric_name}: {stats['latest']:.3f} (avg: {stats['avg']:.3f})")
        
        # Model performance
        model_summary = self.get_model_performance_summary()
        if "best_model" in model_summary:
            report_lines.append("\n🏆 Model Performance:")
            best = model_summary["best_model"]
            report_lines.append(f"  • Best Model: {best['name']} (score: {best['score']:.3f})")
            report_lines.append(f"  • Total Models Tried: {model_summary['total_models']}")
            
            if "averages" in model_summary:
                avg = model_summary["averages"]
                report_lines.append(f"  • Average Score: {avg['score']:.3f}")
        
        report_lines.append("=" * 50)
        
        return "\n".join(report_lines)
    
    export_formats = ["json", "csv", "html"]
    
    def export_metrics(self, filename: str, format_type: str = "json"):
        """Export metrics to file"""
        try:
            if format_type == "json":
                self._export_json(filename)
            elif format_type == "csv":
                self._export_csv(filename)
            elif format_type == "html":
                self._export_html(filename)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
            logger.info(f"📊 Metrics exported to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def _export_json(self, filename: str):
        """Export metrics as JSON"""
        data = {
            "metrics": [asdict(m) for m in self.metrics_history],
            "model_performance": [asdict(m) for m in self.model_performance],
            "summary": {
                "metrics": self.get_metrics_summary(),
                "models": self.get_model_performance_summary()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _export_csv(self, filename: str):
        """Export metrics as CSV"""
        import csv
        
        # Export metrics
        with open(filename.replace('.csv', '_metrics.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'metric_name', 'value', 'unit', 'context'])
            
            for metric in self.metrics_history:
                writer.writerow([
                    metric.timestamp,
                    metric.metric_name,
                    metric.value,
                    metric.unit,
                    json.dumps(metric.context)
                ])
        
        # Export model performance
        with open(filename.replace('.csv', '_models.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model_name', 'training_time', 'prediction_time', 'score', 'metrics', 'parameters'])
            
            for model in self.model_performance:
                writer.writerow([
                    model.model_name,
                    model.training_time,
                    model.prediction_time,
                    model.score,
                    json.dumps(model.metrics),
                    json.dumps(model.parameters)
                ])
    
    def _export_html(self, filename: str):
        """Export metrics as HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AutoML Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 AutoML Performance Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📈 Performance Metrics</h2>
                <div class="metric">
                    <strong>Total Duration:</strong> {self.get_current_duration():.2f} seconds
                </div>
            </div>
            
            <div class="section">
                <h2>🏆 Model Performance</h2>
                <table>
                    <tr><th>Model</th><th>Score</th><th>Training Time</th></tr>
        """
        
        model_summary = self.get_model_performance_summary()
        if "all_models" in model_summary:
            for model in model_summary["all_models"]:
                html_content += f"""
                    <tr>
                        <td>{model['name']}</td>
                        <td>{model['score']:.3f}</td>
                        <td>{model['training_time']:.2f}s</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
    
    def reset(self):
        """Reset all tracking data"""
        self.metrics_history.clear()
        self.model_performance.clear()
        self.start_time = None
        self.current_phase = None
        self.phase_start_time = None
        logger.info("📊 Performance tracking reset")


# Global performance tracker instance
global_performance_tracker = PerformanceTracker()


def start_tracking():
    """Start global performance tracking"""
    global_performance_tracker.start_tracking()


def start_phase(phase_name: str):
    """Start new phase in global tracker"""
    global_performance_tracker.start_phase(phase_name)


def end_phase(phase_name: str = None):
    """End current phase in global tracker"""
    global_performance_tracker.end_phase(phase_name)


def record_metric(metric_name: str, value: float, unit: str = "", context: Dict[str, Any] = None):
    """Record metric in global tracker"""
    global_performance_tracker.record_metric(metric_name, value, unit, context)


def record_model_performance(model_name: str, training_time: float, prediction_time: float,
                        score: float, metrics: Dict[str, float] = None,
                        parameters: Dict[str, Any] = None):
    """Record model performance in global tracker"""
    global_performance_tracker.record_model_performance(
        model_name, training_time, prediction_time, score, metrics, parameters
    )
