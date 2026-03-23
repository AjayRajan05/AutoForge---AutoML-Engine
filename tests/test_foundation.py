import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from api.automl import AutoML, validate_input, detect_task_type
from models.registry import MODEL_REGISTRY, TASK_TYPES
from optimizer.optuna_search import OptunaOptimizer
from meta_learning.knowledge_base import KnowledgeBase
from meta_learning.self_improver import SelfImprover


class TestFoundationFixes:
    """Test all foundation stabilization fixes"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create test data
        np.random.seed(42)
        self.X_classification = np.random.rand(100, 5)
        self.y_classification = np.random.randint(0, 2, 100)
        
        self.X_regression = np.random.rand(100, 5)
        self.y_regression = np.random.rand(100)
        
        # Create temporary directory for logs
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "test_logs.json")
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_input_success(self):
        """Test successful input validation"""
        X_valid, y_valid = validate_input(self.X_classification, self.y_classification)
        
        assert X_valid.shape == self.X_classification.shape
        assert y_valid.shape == self.y_classification.shape
        assert isinstance(X_valid, np.ndarray)
        assert isinstance(y_valid, np.ndarray)
    
    def test_validate_input_pandas(self):
        """Test input validation with pandas DataFrame"""
        X_df = pd.DataFrame(self.X_classification)
        y_series = pd.Series(self.y_classification)
        
        X_valid, y_valid = validate_input(X_df, y_series)
        
        assert X_valid.shape == X_df.shape
        assert y_valid.shape == y_series.shape
        assert isinstance(X_valid, np.ndarray)
        assert isinstance(y_valid, np.ndarray)
    
    def test_validate_input_mismatched_dimensions(self):
        """Test validation with mismatched dimensions"""
        X_wrong = np.random.rand(50, 5)  # Wrong number of samples
        
        with pytest.raises(ValueError, match="mismatched dimensions"):
            validate_input(X_wrong, self.y_classification)
    
    def test_validate_input_too_small(self):
        """Test validation with too small dataset"""
        X_small = np.random.rand(5, 5)
        y_small = np.random.randint(0, 2, 5)
        
        with pytest.raises(ValueError, match="too small"):
            validate_input(X_small, y_small)
    
    def test_validate_input_no_features(self):
        """Test validation with no features"""
        X_no_features = np.random.rand(100, 0)
        
        with pytest.raises(ValueError, match="no features"):
            validate_input(X_no_features, self.y_classification)
    
    def test_validate_input_too_many_missing(self):
        """Test validation with too many missing values"""
        X_missing = np.full((100, 5), np.nan)
        
        with pytest.raises(ValueError, match="Too many missing values"):
            validate_input(X_missing, self.y_classification)
    
    def test_validate_target_insufficient_classes(self):
        """Test validation with insufficient target classes"""
        y_single_class = np.ones(100)  # All same value
        
        with pytest.raises(ValueError, match="less than 2 unique classes"):
            validate_input(self.X_classification, y_single_class)
    
    def test_detect_task_type_classification(self):
        """Test classification task detection"""
        task_type = detect_task_type(self.y_classification)
        assert task_type == "classification"
    
    def test_detect_task_type_regression(self):
        """Test regression task detection"""
        task_type = detect_task_type(self.y_regression)
        assert task_type == "regression"
    
    def test_detect_task_type_multiclass(self):
        """Test multi-class classification detection"""
        y_multiclass = np.random.randint(0, 5, 100)
        task_type = detect_task_type(y_multiclass)
        assert task_type == "classification"
    
    def test_automl_initialization(self):
        """Test AutoML initialization"""
        automl = AutoML(n_trials=10, timeout=300, cv=5)
        
        assert automl.n_trials == 10
        assert automl.timeout == 300
        assert automl.cv == 5
        assert automl.best_pipeline is None
        assert automl.task_type is None
        assert automl.ensemble is None
    
    def test_automl_fit_classification(self):
        """Test AutoML fitting for classification"""
        # Mock all dependencies to speed up test and avoid external dependencies
        with patch('api.automl.OptunaOptimizer') as mock_optimizer, \
             patch('api.automl.profile_dataset') as mock_profile, \
             patch('api.automl.KnowledgeBase') as mock_kb, \
             patch('api.automl.MetaRecommender') as mock_recommender, \
             patch('api.automl.ExperimentLogger') as mock_logger, \
             patch('api.automl.build_pipeline') as mock_build_pipeline:
            
            # Mock dataset profiling
            mock_profile.return_value = {
                "num_rows": 100,
                "num_cols": 5,
                "numeric_cols": 5,
                "categorical_cols": 0,
                "missing_ratio": 0.0,
                "target_classes": 2
            }
            
            # Mock optimization
            mock_study = MagicMock()
            mock_study.best_value = 0.85
            mock_study.best_params = {
                "model": "classification_random_forest",
                "n_estimators": 100,
                "max_depth": 10
            }
            mock_study.trials = [MagicMock() for _ in range(5)]
            # Make best_trial.value work with formatting
            mock_study.best_trial = mock_study
            mock_study.value = 0.85
            # Mock params.copy() and pop() properly
            mock_params_dict = MagicMock()
            mock_params_dict.copy.return_value = {
                "model": "classification_random_forest",
                "n_estimators": 100,
                "max_depth": 10
            }
            mock_params_dict.copy.return_value.pop = MagicMock(return_value="classification_random_forest")
            mock_study.params = mock_params_dict
            
            mock_opt_instance = MagicMock()
            mock_opt_instance.study = mock_study
            mock_opt_instance.optimize.return_value = [
                (0.85, "classification_random_forest", {"n_estimators": 100}),
                (0.82, "classification_logistic_regression", {"C": 1.0}),
                (0.80, "classification_knn", {"n_neighbors": 5})
            ]
            mock_optimizer.return_value = mock_opt_instance
            
            # Mock knowledge base and recommender
            mock_kb_instance = MagicMock()
            mock_kb.return_value = mock_kb_instance
            
            mock_rec_instance = MagicMock()
            mock_rec_instance.recommend.return_value = []
            mock_rec_instance.get_preprocessing_hints.return_value = {}
            mock_recommender.return_value = mock_rec_instance
            
            # Mock experiment logger
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            
            # Mock pipeline
            mock_pipeline = MagicMock()
            mock_build_pipeline.return_value = mock_pipeline
            
            automl = AutoML(n_trials=5)
            automl.fit(self.X_classification, self.y_classification)
            
            assert automl.task_type == "classification"
            assert automl.best_pipeline is not None
            assert automl.dataset_profile is not None
    
    def test_automl_fit_regression(self):
        """Test AutoML fitting for regression"""
        # Mock all dependencies to speed up test and avoid external dependencies
        with patch('api.automl.OptunaOptimizer') as mock_optimizer, \
             patch('api.automl.profile_dataset') as mock_profile, \
             patch('api.automl.KnowledgeBase') as mock_kb, \
             patch('api.automl.MetaRecommender') as mock_recommender, \
             patch('api.automl.ExperimentLogger') as mock_logger, \
             patch('api.automl.build_pipeline') as mock_build_pipeline:
            
            # Mock dataset profiling
            mock_profile.return_value = {
                "num_rows": 100,
                "num_cols": 5,
                "numeric_cols": 5,
                "categorical_cols": 0,
                "missing_ratio": 0.0,
                "target_classes": 100  # Many unique values for regression
            }
            
            # Mock optimization
            mock_study = MagicMock()
            mock_study.best_value = -0.1  # Negative for regression (MSE)
            mock_study.best_params = {
                "model": "regression_random_forest_regressor",
                "n_estimators": 100
            }
            mock_study.trials = [MagicMock() for _ in range(5)]
            # Make best_trial.value work with formatting
            mock_study.best_trial = mock_study
            mock_study.value = -0.1
            # Mock params.copy() and pop() properly
            mock_params_dict = MagicMock()
            mock_params_dict.copy.return_value = {
                "model": "regression_random_forest_regressor",
                "n_estimators": 100
            }
            mock_params_dict.copy.return_value.pop = MagicMock(return_value="regression_random_forest_regressor")
            mock_study.params = mock_params_dict
            
            mock_opt_instance = MagicMock()
            mock_opt_instance.study = mock_study
            mock_opt_instance.optimize.return_value = [
                (-0.1, "regression_random_forest_regressor", {"n_estimators": 100}),
                (-0.12, "regression_linear_regression", {}),
                (-0.15, "regression_ridge", {"alpha": 1.0})
            ]
            mock_optimizer.return_value = mock_opt_instance
            
            # Mock knowledge base and recommender
            mock_kb_instance = MagicMock()
            mock_kb.return_value = mock_kb_instance
            
            mock_rec_instance = MagicMock()
            mock_rec_instance.recommend.return_value = []
            mock_rec_instance.get_preprocessing_hints.return_value = {}
            mock_recommender.return_value = mock_rec_instance
            
            # Mock experiment logger
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance
            
            # Mock pipeline
            mock_pipeline = MagicMock()
            mock_build_pipeline.return_value = mock_pipeline
            
            automl = AutoML(n_trials=5)
            automl.fit(self.X_regression, self.y_regression)
            
            assert automl.task_type == "regression"
            assert automl.best_pipeline is not None
    
    def test_automl_predict_before_fit(self):
        """Test prediction before fitting raises error"""
        automl = AutoML()
        
        with pytest.raises(ValueError, match="Model not fitted"):
            automl.predict(self.X_classification)
    
    def test_automl_predict_proba_regression(self):
        """Test predict_proba on regression task raises error"""
        automl = AutoML()
        automl.task_type = "regression"
        
        with pytest.raises(ValueError, match="only available for classification"):
            automl.predict_proba(self.X_classification)
    
    def test_optimizer_with_task_type(self):
        """Test OptunaOptimizer with task type support"""
        optimizer = OptunaOptimizer(n_trials=3, cv=3, task_type="classification")
        
        assert optimizer.task_type == "classification"
        assert optimizer.n_trials == 3
        assert optimizer.cv == 3
    
    def test_knowledge_base_error_handling(self):
        """Test KnowledgeBase error handling"""
        # Test with non-existent file
        kb = KnowledgeBase(self.log_path)
        
        # Should create empty file and return empty list
        records = kb.load()
        assert records == []
        
        # Test file creation
        assert os.path.exists(self.log_path)
        
        # Test saving record
        test_record = {
            "run_id": "test_1",
            "timestamp": "2023-01-01T00:00:00",
            "model": "test_model",
            "metrics": {"cv_score": 0.8}
        }
        
        success = kb.save_record(test_record)
        assert success
        
        # Test loading records
        records = kb.load()
        assert len(records) == 1
        assert records[0]["run_id"] == "test_1"
    
    def test_knowledge_base_corrupted_file(self):
        """Test KnowledgeBase handling of corrupted file"""
        # Create corrupted JSON file
        with open(self.log_path, 'w') as f:
            f.write("invalid json content")
        
        kb = KnowledgeBase(self.log_path)
        records = kb.load()
        
        # Should handle gracefully and return empty list
        assert records == []
        
        # Should create backup and new empty file
        assert os.path.exists(f"{self.log_path}.backup")
    
    def test_self_improver_error_handling(self):
        """Test SelfImprover error handling"""
        improver = SelfImprover(self.log_path)
        
        # Test with no logs
        success = improver.analyze()
        assert success is False
        
        # Test get_best with no data
        best = improver.get_best()
        assert best["model_priority"] == []
        assert best["scaler_priority"] == []
        assert best["imputer_priority"] == []
    
    def test_self_improver_with_data(self):
        """Test SelfImprover with actual data"""
        # Create test logs
        test_logs = [
            {
                "run_id": "test_1",
                "timestamp": "2023-01-01T00:00:00",
                "model": "classification_random_forest",
                "params": {"scaler": "standard", "imputer": "mean"},
                "metrics": {"cv_score": 0.85}
            },
            {
                "run_id": "test_2",
                "timestamp": "2023-01-01T01:00:00",
                "model": "classification_logistic_regression",
                "params": {"scaler": "minmax", "imputer": "median"},
                "metrics": {"cv_score": 0.78}
            }
        ]
        
        with open(self.log_path, 'w') as f:
            json.dump(test_logs, f)
        
        improver = SelfImprover(self.log_path)
        success = improver.analyze()
        
        assert success is True
        
        best = improver.get_best()
        assert len(best["model_priority"]) > 0
        assert "classification_random_forest" in best["model_priority"]
        assert len(best["scaler_priority"]) > 0
        assert "standard" in best["scaler_priority"]
    
    def test_model_registry_completeness(self):
        """Test model registry has all required models"""
        # Check that all models have task types
        for model_name in MODEL_REGISTRY:
            assert model_name in TASK_TYPES
            assert TASK_TYPES[model_name] in ["classification", "regression"]
        
        # Check classification models
        classification_models = [name for name, task in TASK_TYPES.items() if task == "classification"]
        assert len(classification_models) >= 8  # We added 8 classification models
        
        # Check regression models
        regression_models = [name for name, task in TASK_TYPES.items() if task == "regression"]
        assert len(regression_models) >= 7  # We added 7 regression models
    
    def test_model_instantiation(self):
        """Test all models can be instantiated"""
        X_small = np.random.rand(20, 3)
        y_class_small = np.random.randint(0, 2, 20)
        y_reg_small = np.random.rand(20)
        
        for model_name, model_class in MODEL_REGISTRY.items():
            try:
                model = model_class()
                
                if "classification" in model_name:
                    model.fit(X_small, y_class_small)
                    preds = model.predict(X_small)
                    assert len(preds) == len(y_class_small)
                else:
                    model.fit(X_small, y_reg_small)
                    preds = model.predict(X_small)
                    assert len(preds) == len(y_reg_small)
                    
            except Exception as e:
                pytest.fail(f"Model {model_name} failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
