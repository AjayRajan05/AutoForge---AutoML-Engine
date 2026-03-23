"""
Test suite for Phase 1: Core Performance Engine features
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import time
from unittest.mock import patch, MagicMock

# Import the new performance features
from core.dataset_optimizer import DatasetOptimizer, optimize_dataset
from optimizer.adaptive_optimizer import AdaptiveOptimizer, create_adaptive_optimizer
from core.pipeline_cache import PipelineCache, CachedPipelineBuilder
from core.progress_tracker import ProgressTracker, create_progress_tracker


class TestDatasetOptimizer:
    """Test dataset-aware optimization engine"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        
        # Small dataset
        self.X_small = np.random.rand(1000, 10)
        self.y_small = np.random.randint(0, 2, 1000)
        
        # Large dataset
        self.X_large = np.random.rand(150_000, 20)
        self.y_large = np.random.randint(0, 3, 150_000)
        
        # Very large dataset
        self.X_very_large = np.random.rand(1_200_000, 15)
        self.y_very_large = np.random.randint(0, 2, 1_200_000)
        
        self.optimizer = DatasetOptimizer()
    
    def test_small_dataset_no_sampling(self):
        """Test that small datasets use full data"""
        (X_opt, y_opt), strategy, metadata = self.optimizer.optimize_dataset(
            self.X_small, self.y_small, "classification"
        )
        
        assert strategy == "full"
        assert X_opt.shape == self.X_small.shape
        assert y_opt.shape == self.y_small.shape
        assert metadata["sampling_ratio"] == 1.0
    
    def test_large_dataset_sampling(self):
        """Test that large datasets get sampled"""
        (X_opt, y_opt), strategy, metadata = self.optimizer.optimize_dataset(
            self.X_large, self.y_large, "classification"
        )
        
        assert strategy == "sampled"
        assert X_opt.shape[0] < self.X_large.shape[0]
        assert y_opt.shape[0] < self.y_large.shape[0]
        assert metadata["sampling_ratio"] < 1.0
        assert metadata["original_size"] == self.X_large.shape[0]
        assert metadata["sampled_size"] == X_opt.shape[0]
    
    def test_very_large_dataset_adaptive_sampling(self):
        """Test adaptive sampling for very large datasets"""
        (X_opt, y_opt), strategy, metadata = self.optimizer.optimize_dataset(
            self.X_very_large, self.y_very_large, "classification"
        )
        
        assert strategy == "sampled"
        # Should use smaller sample for very large dataset
        assert X_opt.shape[0] <= 50_000  # max_sample_size
        assert metadata["sampling_ratio"] < 0.1  # Very small ratio
    
    def test_stratified_sampling_classification(self):
        """Test stratified sampling for classification"""
        # Create imbalanced dataset
        X_imbalanced = np.random.rand(10_000, 5)
        y_imbalanced = np.array([0] * 9000 + [1] * 1000)  # 90:10 ratio
        
        (X_opt, y_opt), strategy, metadata = self.optimizer.optimize_dataset(
            X_imbalanced, y_imbalanced, "classification"
        )
        
        # Check that class distribution is preserved
        unique_opt, counts_opt = np.unique(y_opt, return_counts=True)
        ratio_opt = counts_opt[1] / counts_opt[0] if len(counts_opt) > 1 else 0
        
        unique_orig, counts_orig = np.unique(y_imbalanced, return_counts=True)
        ratio_orig = counts_orig[1] / counts_orig[0]
        
        # Ratios should be similar (within 20% tolerance)
        assert abs(ratio_opt - ratio_orig) / ratio_orig < 0.2
    
    def test_regression_random_sampling(self):
        """Test random sampling for regression"""
        y_regression = np.random.rand(50_000)
        
        (X_opt, y_opt), strategy, metadata = self.optimizer.optimize_dataset(
            self.X_large[:50_000], y_regression, "regression"
        )
        
        assert strategy == "sampled"
        assert len(y_opt) < len(y_regression)
        # Check that we have a reasonable range of values
        assert y_opt.min() >= y_regression.min()
        assert y_opt.max() <= y_regression.max()
    
    def test_dataset_characteristics_analysis(self):
        """Test dataset characteristics analysis"""
        characteristics = self.optimizer.analyze_dataset_characteristics(
            self.X_large, self.y_large
        )
        
        assert "n_samples" in characteristics
        assert "n_features" in characteristics
        assert "is_large_dataset" in characteristics
        assert "memory_estimate_mb" in characteristics
        assert characteristics["n_samples"] == self.X_large.shape[0]
        assert characteristics["n_features"] == self.X_large.shape[1]
        assert characteristics["is_large_dataset"] == True
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations"""
        recommendations = self.optimizer.get_optimization_recommendations(
            self.X_large, self.y_large
        )
        
        assert "dataset_characteristics" in recommendations
        assert "optimization_strategy" in recommendations
        assert "recommended_trials" in recommendations
        assert "performance_tips" in recommendations
        
        # Should recommend sampling for large dataset
        assert recommendations["optimization_strategy"] == "sampled"


class TestAdaptiveOptimizer:
    """Test adaptive hyperparameter optimization"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.X = np.random.rand(1000, 10)
        self.y = np.random.randint(0, 2, 1000)
        
        self.optimizer = AdaptiveOptimizer(
            initial_trials=5,
            max_trials=20,
            time_budget=30  # 30 seconds
        )
    
    def test_study_creation(self):
        """Test Optuna study creation"""
        study = self.optimizer.create_study("test_study")
        
        assert study is not None
        assert study.study_name == "test_study"
        assert study.direction.name == "MAXIMIZE"
    
    def test_objective_function_with_pruning(self):
        """Test objective function with pruning support"""
        self.optimizer.create_study()
        
        # Mock objective function
        def mock_objective(trial, X, y, task_type):
            # Simulate a trial that gets pruned
            trial.report(0.5, 1)
            if trial.should_prune():
                raise optuna.TrialPruned()
            trial.report(0.8, 2)
            return 0.8
        
        # Test the wrapped objective
        with patch('optuna.trial.Trial.should_prune', return_value=True):
            with pytest.raises(optuna.TrialPruned):
                score = self.optimizer._adaptive_objective(
                    self.optimizer.study.trials[0],
                    mock_objective,
                    self.X,
                    self.y,
                    "classification"
                )
    
    def test_progress_analysis(self):
        """Test optimization progress analysis"""
        self.optimizer.create_study()
        
        # Simulate some trials
        for i in range(10):
            self.optimizer.study.optimize(lambda t: np.random.rand(), n_trials=1)
            self.optimizer.best_scores.append(self.optimizer.study.best_value)
        
        should_continue, next_trials = self.optimizer._analyze_progress()
        
        assert isinstance(should_continue, bool)
        assert isinstance(next_trials, int)
        assert next_trials <= self.optimizer.max_trials
    
    def test_optimization_metadata(self):
        """Test optimization metadata generation"""
        self.optimizer.create_study()
        
        # Run some trials
        def simple_objective(trial):
            return trial.suggest_float("x", 0, 1)
        
        self.optimizer.study.optimize(simple_objective, n_trials=5)
        
        report = self.optimizer.get_optimization_report()
        
        assert "total_trials" in report
        assert "completed_trials" in report
        assert "best_score" in report
        assert "pruning_efficiency" in report
        assert "model_performance" in report


class TestPipelineCache:
    """Test pipeline-level caching"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.X = np.random.rand(1000, 10)
        self.cache_dir = tempfile.mkdtemp()
        self.cache = PipelineCache(cache_dir=self.cache_dir)
    
    def teardown_method(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        key1 = self.cache.get_cache_key((1000, 10), "float64", {"param1": 1.0}, "scaler")
        key2 = self.cache.get_cache_key((1000, 10), "float64", {"param1": 1.0}, "scaler")
        key3 = self.cache.get_cache_key((1000, 10), "float64", {"param1": 2.0}, "scaler")
        
        assert key1 == key2  # Same params should generate same key
        assert key1 != key3  # Different params should generate different key
    
    def test_cached_fit_transform(self):
        """Test cached fit and transform"""
        # First call should compute and cache
        X1, component1 = self.cache.cached_fit_transform(
            self.X, "scaler_standard", {}, "test_key"
        )
        
        # Second call should use cache
        X2, component2 = self.cache.cached_fit_transform(
            self.X, "scaler_standard", {}, "test_key"
        )
        
        # Results should be identical
        np.testing.assert_array_almost_equal(X1, X2)
        assert type(component1) == type(component2)
    
    def test_cache_statistics(self):
        """Test cache statistics"""
        # Perform some operations
        self.cache.cached_fit_transform(self.X, "scaler_standard", {}, "test_key1")
        self.cache.cached_fit_transform(self.X, "scaler_minmax", {}, "test_key2")
        
        # Try to access same key again (cache hit)
        self.cache.cached_fit_transform(self.X, "scaler_standard", {}, "test_key1")
        
        stats = self.cache.get_cache_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "cache_size_mb" in stats
        assert stats["hits"] >= 1
        assert stats["misses"] >= 2
    
    def test_cached_pipeline_builder(self):
        """Test cached pipeline builder"""
        builder = CachedPipelineBuilder(self.cache)
        
        config = {
            "steps": [
                {"type": "scaler_standard", "params": {}},
                {"type": "imputer_mean", "params": {}}
            ]
        }
        
        X_transformed, components = builder.build_cached_pipeline(self.X, config)
        
        assert X_transformed.shape[0] == self.X.shape[0]
        assert len(components) == 2
        
        # Test transform with cache
        X_transformed2 = builder.transform_with_cache(self.X, components, config)
        
        assert X_transformed2.shape == X_transformed.shape


class TestProgressTracker:
    """Test progress tracking and developer experience"""
    
    def setup_method(self):
        """Setup"""
        self.tracker = ProgressTracker(show_progress=False)  # Disable rich for testing
    
    def test_trial_result_tracking(self):
        """Test trial result tracking"""
        self.tracker.add_trial_result(
            trial_number=1,
            model_name="random_forest",
            score=0.85,
            params={"n_estimators": 100}
        )
        
        assert len(self.tracker.trials) == 1
        assert self.tracker.trials[0].model_name == "random_forest"
        assert self.tracker.trials[0].score == 0.85
    
    def test_model_performance_tracking(self):
        """Test model performance tracking"""
        # Add multiple trials for same model
        self.tracker.add_trial_result(1, "random_forest", 0.85, {})
        self.tracker.add_trial_result(2, "random_forest", 0.87, {})
        self.tracker.add_trial_result(3, "logistic_regression", 0.82, {})
        
        perf = self.tracker.model_performances["random_forest"]
        
        assert perf.best_score == 0.87
        assert perf.avg_score == 0.86
        assert perf.trial_count == 2
    
    def test_optimization_summary(self):
        """Test optimization summary generation"""
        # Add some trials
        self.tracker.add_trial_result(1, "random_forest", 0.85, {})
        self.tracker.add_trial_result(2, "logistic_regression", 0.82, {})
        self.tracker.add_trial_result(3, "svm", 0.88, {})
        
        summary = self.tracker.get_optimization_summary()
        
        assert "total_trials" in summary
        assert "best_score" in summary
        assert "best_model" in summary
        assert "models_tested" in summary
        assert summary["best_score"] == 0.88
        assert summary["best_model"] == "svm"
    
    def test_progress_tracker_creation(self):
        """Test progress tracker convenience function"""
        tracker = create_progress_tracker(show_progress=False)
        
        assert isinstance(tracker, ProgressTracker)
        assert tracker.show_progress == False


class TestIntegratedPerformanceFeatures:
    """Test integration of all performance features"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.X = np.random.rand(50_000, 15)  # Medium-sized dataset
        self.y = np.random.randint(0, 2, 50_000)
    
    def test_complete_performance_pipeline(self):
        """Test complete pipeline with all performance features"""
        # Dataset optimization
        optimizer = DatasetOptimizer()
        (X_opt, y_opt), strategy, metadata = optimizer.optimize_dataset(
            self.X, self.y, "classification"
        )
        
        # Should use sampling for this size
        assert strategy == "sampled"
        assert X_opt.shape[0] < self.X.shape[0]
        
        # Progress tracking
        tracker = create_progress_tracker(show_progress=False)
        tracker.start_optimization(10, "classification")
        
        # Simulate some trials
        models = ["random_forest", "logistic_regression", "svm"]
        for i, model in enumerate(models):
            score = np.random.rand() * 0.3 + 0.7  # Scores between 0.7-1.0
            tracker.add_trial_result(i, model, score, {})
        
        # Check results
        summary = tracker.get_optimization_summary()
        assert summary["total_trials"] == 3
        assert summary["models_tested"] == 3
        
        tracker.finish_optimization()
        
        # Pipeline caching
        cache = PipelineCache()
        X_cached, component = cache.cached_fit_transform(
            X_opt[:1000], "scaler_standard", {}, "test_integration"
        )
        
        assert X_cached.shape[0] == 1000
        
        stats = cache.get_cache_stats()
        assert stats["hits"] >= 0
        assert stats["misses"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
