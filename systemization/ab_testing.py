"""
A/B Testing Framework
Model comparison and statistical significance testing
"""

import numpy as np
import pandas as pd
import logging
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class ABTestingFramework:
    """
    A/B testing framework for model comparison
    """
    
    def __init__(self, 
                 results_dir: str = "ab_testing",
                 significance_level: float = 0.05,
                 min_sample_size: int = 30):
        """
        Initialize A/B testing framework
        
        Args:
            results_dir: Directory to store test results
            significance_level: Alpha level for statistical tests
            min_sample_size: Minimum sample size for testing
        """
        self.results_dir = Path(results_dir)
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        
        # Test results storage
        self.test_results = {}
        
        logger.info(f"A/B testing framework initialized: {self.results_dir}")
    
    def compare_models(self, 
                      model1_name: str,
                      model2_name: str,
                      X_test: Union[np.ndarray, pd.DataFrame],
                      y_test: Union[np.ndarray, pd.Series],
                      y_pred1: Union[np.ndarray, pd.Series],
                      y_pred2: Union[np.ndarray, pd.Series],
                      task_type: str = "classification",
                      test_name: str = None) -> Dict[str, Any]:
        """
        Compare two models with statistical significance testing
        
        Args:
            model1_name: Name of first model
            model2_name: Name of second model
            X_test: Test features
            y_test: True labels
            y_pred1: Predictions from first model
            y_pred2: Predictions from second model
            task_type: Type of ML task
            test_name: Optional name for the test
            
        Returns:
            Comparison results
        """
        try:
            if test_name is None:
                test_name = f"{model1_name}_vs_{model2_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Validate inputs
            if len(y_test) < self.min_sample_size:
                raise ValueError(f"Insufficient sample size: {len(y_test)} < {self.min_sample_size}")
            
            # Calculate metrics for both models
            metrics1 = self._calculate_metrics(y_test, y_pred1, task_type)
            metrics2 = self._calculate_metrics(y_test, y_pred2, task_type)
            
            # Perform statistical tests
            statistical_tests = self._perform_statistical_tests(
                y_test, y_pred1, y_pred2, task_type
            )
            
            # Determine winner
            winner, improvement = self._determine_winner(metrics1, metrics2, task_type)
            
            # Create test result
            test_result = {
                "test_name": test_name,
                "test_date": datetime.datetime.now().isoformat(),
                "model1": {
                    "name": model1_name,
                    "metrics": metrics1
                },
                "model2": {
                    "name": model2_name,
                    "metrics": metrics2
                },
                "statistical_tests": statistical_tests,
                "comparison": {
                    "winner": winner,
                    "improvement": improvement,
                    "significant": statistical_tests.get("significant", False)
                },
                "task_type": task_type,
                "sample_size": len(y_test),
                "significance_level": self.significance_level
            }
            
            # Store result
            self.test_results[test_name] = test_result
            
            # Save to file
            self._save_test_result(test_result)
            
            # Log results
            self._log_comparison_results(test_result)
            
            return test_result
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise
    
    def _calculate_metrics(self, 
                          y_true: Union[np.ndarray, pd.Series],
                          y_pred: Union[np.ndarray, pd.Series],
                          task_type: str) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            # Convert to numpy arrays
            if hasattr(y_true, 'values'):
                y_true = y_true.values
            if hasattr(y_pred, 'values'):
                y_pred = y_pred.values
            
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            metrics = {}
            
            if task_type == "classification":
                metrics["accuracy"] = accuracy_score(y_true, y_pred)
                metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted')
                
                # Calculate per-class accuracy
                unique_classes = np.unique(y_true)
                per_class_accuracy = {}
                for cls in unique_classes:
                    cls_mask = y_true == cls
                    if cls_mask.sum() > 0:
                        per_class_accuracy[f"class_{cls}"] = accuracy_score(
                            y_true[cls_mask], y_pred[cls_mask]
                        )
                metrics["per_class_accuracy"] = per_class_accuracy
                
            elif task_type == "regression":
                metrics["mse"] = mean_squared_error(y_true, y_pred)
                metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics["r2_score"] = r2_score(y_true, y_pred)
                metrics["mae"] = np.mean(np.abs(y_true - y_pred))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {"error": str(e)}
    
    def _perform_statistical_tests(self, 
                                   y_true: Union[np.ndarray, pd.Series],
                                   y_pred1: Union[np.ndarray, pd.Series],
                                   y_pred2: Union[np.ndarray, pd.Series],
                                   task_type: str) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        try:
            # Convert to numpy arrays
            if hasattr(y_true, 'values'):
                y_true = y_true.values
            if hasattr(y_pred1, 'values'):
                y_pred1 = y_pred1.values
            if hasattr(y_pred2, 'values'):
                y_pred2 = y_pred2.values
            
            y_true = np.array(y_true)
            y_pred1 = np.array(y_pred1)
            y_pred2 = np.array(y_pred2)
            
            test_results = {}
            
            if task_type == "classification":
                # McNemar's test for classification
                test_results = self._mcnemar_test(y_true, y_pred1, y_pred2)
                
            elif task_type == "regression":
                # Paired t-test for regression
                test_results = self._paired_t_test(y_true, y_pred1, y_pred2)
            
            # Bootstrap test for both classification and regression
            bootstrap_results = self._bootstrap_test(y_true, y_pred1, y_pred2, task_type)
            test_results["bootstrap"] = bootstrap_results
            
            # Overall significance
            test_results["significant"] = any(
                result.get("p_value", 1.0) < self.significance_level
                for result in test_results.values()
                if isinstance(result, dict) and "p_value" in result
            )
            
            return test_results
            
        except Exception as e:
            logger.error(f"Failed to perform statistical tests: {e}")
            return {"error": str(e)}
    
    def _mcnemar_test(self, 
                      y_true: np.ndarray,
                      y_pred1: np.ndarray,
                      y_pred2: np.ndarray) -> Dict[str, Any]:
        """Perform McNemar's test for classification"""
        try:
            # Create contingency table
            correct1 = (y_pred1 == y_true)
            correct2 = (y_pred2 == y_true)
            
            # Contingency table:
            #           Model2 Correct
            #           Yes    No
            # Model1 Yes  a      b
            #         No   c      d
            
            a = np.sum(correct1 & correct2)  # Both correct
            b = np.sum(correct1 & ~correct2)  # Model1 correct, Model2 wrong
            c = np.sum(~correct1 & correct2)  # Model1 wrong, Model2 correct
            d = np.sum(~correct1 & ~correct2)  # Both wrong
            
            # McNemar's test statistic (with continuity correction)
            if b + c == 0:
                return {"test": "mcnemar", "p_value": 1.0, "statistic": 0.0}
            
            statistic = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
            
            return {
                "test": "mcnemar",
                "statistic": statistic,
                "p_value": p_value,
                "contingency_table": {"a": int(a), "b": int(b), "c": int(c), "d": int(d)},
                "significant": p_value < self.significance_level
            }
            
        except Exception as e:
            logger.error(f"Failed to perform McNemar's test: {e}")
            return {"test": "mcnemar", "error": str(e)}
    
    def _paired_t_test(self, 
                      y_true: np.ndarray,
                      y_pred1: np.ndarray,
                      y_pred2: np.ndarray) -> Dict[str, Any]:
        """Perform paired t-test for regression"""
        try:
            # Calculate absolute errors
            errors1 = np.abs(y_true - y_pred1)
            errors2 = np.abs(y_true - y_pred2)
            
            # Paired t-test
            statistic, p_value = stats.ttest_rel(errors1, errors2)
            
            # Calculate effect size (Cohen's d)
            diff = errors1 - errors2
            pooled_std = np.sqrt((np.var(errors1) + np.var(errors2)) / 2)
            effect_size = np.mean(diff) / pooled_std if pooled_std > 0 else 0
            
            return {
                "test": "paired_t_test",
                "statistic": statistic,
                "p_value": p_value,
                "effect_size": effect_size,
                "mean_diff": np.mean(diff),
                "significant": p_value < self.significance_level
            }
            
        except Exception as e:
            logger.error(f"Failed to perform paired t-test: {e}")
            return {"test": "paired_t_test", "error": str(e)}
    
    def _bootstrap_test(self, 
                       y_true: np.ndarray,
                       y_pred1: np.ndarray,
                       y_pred2: np.ndarray,
                       task_type: str,
                       n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Perform bootstrap test"""
        try:
            n_samples = len(y_true)
            bootstrap_diffs = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                
                y_true_boot = y_true[indices]
                y_pred1_boot = y_pred1[indices]
                y_pred2_boot = y_pred2[indices]
                
                # Calculate metrics
                if task_type == "classification":
                    metric1 = accuracy_score(y_true_boot, y_pred1_boot)
                    metric2 = accuracy_score(y_true_boot, y_pred2_boot)
                else:  # regression
                    metric1 = r2_score(y_true_boot, y_pred1_boot)
                    metric2 = r2_score(y_true_boot, y_pred2_boot)
                
                bootstrap_diffs.append(metric2 - metric1)
            
            bootstrap_diffs = np.array(bootstrap_diffs)
            
            # Calculate confidence interval
            ci_lower = np.percentile(bootstrap_diffs, 2.5)
            ci_upper = np.percentile(bootstrap_diffs, 97.5)
            
            # Calculate p-value (two-sided)
            observed_diff = bootstrap_diffs.mean()
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
            
            return {
                "test": "bootstrap",
                "mean_difference": observed_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p_value,
                "n_bootstrap": n_bootstrap,
                "significant": p_value < self.significance_level
            }
            
        except Exception as e:
            logger.error(f"Failed to perform bootstrap test: {e}")
            return {"test": "bootstrap", "error": str(e)}
    
    def _determine_winner(self, 
                         metrics1: Dict[str, float],
                         metrics2: Dict[str, float],
                         task_type: str) -> Tuple[str, Dict[str, float]]:
        """Determine which model performed better"""
        try:
            if task_type == "classification":
                primary_metric = "accuracy"
                higher_is_better = True
            else:  # regression
                primary_metric = "r2_score"
                higher_is_better = True
            
            if primary_metric not in metrics1 or primary_metric not in metrics2:
                return "unknown", {}
            
            score1 = metrics1[primary_metric]
            score2 = metrics2[primary_metric]
            
            if higher_is_better:
                if score2 > score1:
                    winner = "model2"
                    improvement = score2 - score1
                    pct_improvement = (improvement / score1) * 100 if score1 != 0 else 0
                else:
                    winner = "model1"
                    improvement = score1 - score2
                    pct_improvement = (improvement / score2) * 100 if score2 != 0 else 0
            else:
                if score2 < score1:
                    winner = "model2"
                    improvement = score1 - score2
                    pct_improvement = (improvement / score1) * 100 if score1 != 0 else 0
                else:
                    winner = "model1"
                    improvement = score2 - score1
                    pct_improvement = (improvement / score2) * 100 if score2 != 0 else 0
            
            improvement_info = {
                "absolute_improvement": improvement,
                "percentage_improvement": pct_improvement,
                "primary_metric": primary_metric,
                "model1_score": score1,
                "model2_score": score2
            }
            
            return winner, improvement_info
            
        except Exception as e:
            logger.error(f"Failed to determine winner: {e}")
            return "unknown", {}
    
    def _save_test_result(self, test_result: Dict[str, Any]) -> None:
        """Save test result to file"""
        try:
            filename = f"{test_result['test_name']}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(test_result, f, indent=2, default=str)
            
            logger.info(f"Test result saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save test result: {e}")
    
    def _log_comparison_results(self, test_result: Dict[str, Any]) -> None:
        """Log comparison results"""
        try:
            comparison = test_result["comparison"]
            model1_name = test_result["model1"]["name"]
            model2_name = test_result["model2"]["name"]
            
            logger.info(f"A/B Test Results: {model1_name} vs {model2_name}")
            logger.info(f"Winner: {comparison['winner']}")
            logger.info(f"Improvement: {comparison['improvement']['percentage_improvement']:.2f}%")
            logger.info(f"Statistically Significant: {comparison['significant']}")
            
            # Log detailed metrics
            task_type = test_result["task_type"]
            if task_type == "classification":
                acc1 = test_result["model1"]["metrics"]["accuracy"]
                acc2 = test_result["model2"]["metrics"]["accuracy"]
                logger.info(f"Accuracy: {model1_name}={acc1:.4f}, {model2_name}={acc2:.4f}")
            else:
                r2_1 = test_result["model1"]["metrics"]["r2_score"]
                r2_2 = test_result["model2"]["metrics"]["r2_score"]
                logger.info(f"R2 Score: {model1_name}={r2_1:.4f}, {model2_name}={r2_2:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to log comparison results: {e}")
    
    def get_test_result(self, test_name: str) -> Dict[str, Any]:
        """Get specific test result"""
        try:
            if test_name in self.test_results:
                return self.test_results[test_name]
            
            # Try to load from file
            filepath = self.results_dir / f"{test_name}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    result = json.load(f)
                self.test_results[test_name] = result
                return result
            
            raise ValueError(f"Test result not found: {test_name}")
            
        except Exception as e:
            logger.error(f"Failed to get test result: {e}")
            raise
    
    def list_tests(self, model_name: str = None) -> List[Dict[str, Any]]:
        """List all tests or tests for specific model"""
        try:
            # Load all test files
            test_files = list(self.results_dir.glob("*.json"))
            
            tests = []
            for test_file in test_files:
                try:
                    with open(test_file, 'r') as f:
                        test_result = json.load(f)
                    
                    # Filter by model name if specified
                    if model_name:
                        model1 = test_result.get("model1", {}).get("name", "")
                        model2 = test_result.get("model2", {}).get("name", "")
                        if model_name not in [model1, model2]:
                            continue
                    
                    tests.append(test_result)
                    
                except Exception as e:
                    logger.warning(f"Failed to load test file {test_file}: {e}")
            
            # Sort by date (newest first)
            tests.sort(key=lambda x: x.get("test_date", ""), reverse=True)
            
            return tests
            
        except Exception as e:
            logger.error(f"Failed to list tests: {e}")
            return []
    
    def get_test_summary(self, test_name: str) -> str:
        """Get human-readable test summary"""
        try:
            test_result = self.get_test_result(test_name)
            
            model1_name = test_result["model1"]["name"]
            model2_name = test_result["model2"]["name"]
            comparison = test_result["comparison"]
            task_type = test_result["task_type"]
            
            summary = []
            summary.append(f"🧪 A/B Test Summary: {model1_name} vs {model2_name}")
            summary.append("=" * 60)
            summary.append(f"Task Type: {task_type}")
            summary.append(f"Sample Size: {test_result['sample_size']:,}")
            summary.append(f"Test Date: {test_result['test_date']}")
            summary.append("")
            
            # Winner announcement
            winner = comparison["winner"]
            if winner == "model1":
                winner_name = model1_name
            elif winner == "model2":
                winner_name = model2_name
            else:
                winner_name = "Tie/Unknown"
            
            summary.append(f"🏆 Winner: {winner_name}")
            summary.append(f"📈 Improvement: {comparison['improvement']['percentage_improvement']:.2f}%")
            summary.append(f"🔬 Statistically Significant: {comparison['significant']}")
            summary.append("")
            
            # Detailed metrics
            summary.append("📊 Performance Metrics:")
            if task_type == "classification":
                acc1 = test_result["model1"]["metrics"]["accuracy"]
                acc2 = test_result["model2"]["metrics"]["accuracy"]
                f1_1 = test_result["model1"]["metrics"]["f1_score"]
                f1_2 = test_result["model2"]["metrics"]["f1_score"]
                
                summary.append(f"  {model1_name}:")
                summary.append(f"    Accuracy: {acc1:.4f}")
                summary.append(f"    F1 Score: {f1_1:.4f}")
                summary.append(f"  {model2_name}:")
                summary.append(f"    Accuracy: {acc2:.4f}")
                summary.append(f"    F1 Score: {f1_2:.4f}")
                
            else:  # regression
                r2_1 = test_result["model1"]["metrics"]["r2_score"]
                r2_2 = test_result["model2"]["metrics"]["r2_score"]
                mse1 = test_result["model1"]["metrics"]["mse"]
                mse2 = test_result["model2"]["metrics"]["mse"]
                
                summary.append(f"  {model1_name}:")
                summary.append(f"    R2 Score: {r2_1:.4f}")
                summary.append(f"    MSE: {mse1:.4f}")
                summary.append(f"  {model2_name}:")
                summary.append(f"    R2 Score: {r2_2:.4f}")
                summary.append(f"    MSE: {mse2:.4f}")
            
            # Statistical test results
            summary.append("")
            summary.append("🔬 Statistical Tests:")
            stat_tests = test_result["statistical_tests"]
            
            for test_name, test_result in stat_tests.items():
                if isinstance(test_result, dict) and "p_value" in test_result:
                    p_val = test_result["p_value"]
                    significant = test_result.get("significant", False)
                    status = "✅ Significant" if significant else "❌ Not Significant"
                    summary.append(f"  {test_name}: p={p_val:.4f} {status}")
            
            return "\n".join(summary)
            
        except Exception as e:
            logger.error(f"Failed to generate test summary: {e}")
            return f"Error generating summary: {e}"
    
    def get_leaderboard(self, task_type: str = None) -> List[Dict[str, Any]]:
        """Get leaderboard of model performance"""
        try:
            # Get all tests
            tests = self.list_tests()
            
            if task_type:
                tests = [t for t in tests if t.get("task_type") == task_type]
            
            # Collect model performance
            model_stats = {}
            
            for test in tests:
                model1_name = test["model1"]["name"]
                model2_name = test["model2"]["name"]
                
                # Update model1 stats
                if model1_name not in model_stats:
                    model_stats[model1_name] = {
                        "name": model1_name,
                        "wins": 0,
                        "losses": 0,
                        "tests": 0,
                        "avg_improvement": 0,
                        "task_type": test["task_type"]
                    }
                
                # Update model2 stats
                if model2_name not in model_stats:
                    model_stats[model2_name] = {
                        "name": model2_name,
                        "wins": 0,
                        "losses": 0,
                        "tests": 0,
                        "avg_improvement": 0,
                        "task_type": test["task_type"]
                    }
                
                # Update stats based on winner
                winner = test["comparison"]["winner"]
                improvement = test["comparison"]["improvement"]["percentage_improvement"]
                
                if winner == "model1":
                    model_stats[model1_name]["wins"] += 1
                    model_stats[model2_name]["losses"] += 1
                    model_stats[model1_name]["avg_improvement"] += improvement
                elif winner == "model2":
                    model_stats[model2_name]["wins"] += 1
                    model_stats[model1_name]["losses"] += 1
                    model_stats[model2_name]["avg_improvement"] += improvement
                
                model_stats[model1_name]["tests"] += 1
                model_stats[model2_name]["tests"] += 1
            
            # Calculate average improvements
            for model in model_stats.values():
                if model["tests"] > 0:
                    model["avg_improvement"] /= model["tests"]
                model["win_rate"] = model["wins"] / model["tests"] if model["tests"] > 0 else 0
            
            # Sort by win rate, then by average improvement
            leaderboard = sorted(
                model_stats.values(),
                key=lambda x: (x["win_rate"], x["avg_improvement"]),
                reverse=True
            )
            
            return leaderboard
            
        except Exception as e:
            logger.error(f"Failed to generate leaderboard: {e}")
            return []


# Convenience functions
def compare_models(model1_name: str, model2_name: str, X_test, y_test, y_pred1, y_pred2, **kwargs) -> Dict[str, Any]:
    """Convenience function for model comparison"""
    ab_test = ABTestingFramework()
    return ab_test.compare_models(model1_name, model2_name, X_test, y_test, y_pred1, y_pred2, **kwargs)


def get_ab_test_leaderboard(task_type: str = None) -> List[Dict[str, Any]]:
    """Convenience function for getting A/B test leaderboard"""
    ab_test = ABTestingFramework()
    return ab_test.get_leaderboard(task_type)
