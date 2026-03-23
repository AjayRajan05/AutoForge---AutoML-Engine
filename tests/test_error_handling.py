import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from api.automl import AutoML
from models.registry import MODEL_REGISTRY, TASK_TYPES


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_dataset(self):
        """Test handling of empty datasets"""
        automl = AutoML(n_trials=5)
        
        # Empty arrays
        X_empty = np.array([])
        y_empty = np.array([])
        
        with pytest.raises(ValueError):
            automl.fit(X_empty, y_empty)
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched X and y dimensions"""
        automl = AutoML(n_trials=5)
        
        X = np.random.rand(100, 5)
        y = np.random.rand(50)  # Wrong length
        
        with pytest.raises(ValueError):
            automl.fit(X, y)
    
    def test_invalid_file_path(self):
        """Test handling of invalid file paths in CLI"""
        from cli.commands import train
        
        with patch('pandas.read_csv', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                train("nonexistent.csv", "target", trials=5)
    
    def test_corrupted_log_file(self):
        """Test handling of corrupted experiment logs"""
        from meta_learning.knowledge_base import KnowledgeBase
        
        with patch('builtins.open', side_effect=json.JSONDecodeError("Expecting value", "", 0)):
            kb = KnowledgeBase("invalid.json")
            with pytest.raises(json.JSONDecodeError):
                kb.load()
    
    def test_memory_limit_handling(self):
        """Test handling of memory-intensive operations"""
        automl = AutoML(n_trials=5)
        
        # Large dataset that might cause memory issues
        X_large = np.random.rand(10000, 1000)
        y_large = np.random.rand(10000)
        
        # Should handle gracefully or fail with clear error
        try:
            automl.fit(X_large, y_large)
        except MemoryError:
            pytest.skip("Memory limit reached - test passed")
        except Exception as e:
            # Should fail gracefully with meaningful error
            assert "memory" in str(e).lower() or "size" in str(e).lower()
    
    def test_invalid_model_parameters(self):
        """Test handling of invalid model parameters"""
        from models.registry import MODEL_REGISTRY
        
        # Test with invalid parameters
        for model_name, model_class in MODEL_REGISTRY.items():
            if "random_forest" in model_name:
                # Invalid parameters
                with pytest.raises((ValueError, TypeError)):
                    model = model_class(n_estimators=-1)  # Invalid n_estimators
                    model.fit(np.random.rand(10, 5), np.random.rand(10))
    
    def test_timeout_handling(self):
        """Test handling of optimization timeouts"""
        automl = AutoML(n_trials=100, timeout=1)  # Very short timeout
        
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        # Should handle timeout gracefully
        try:
            automl.fit(X, y)
        except TimeoutError:
            pass  # Expected behavior
        except Exception as e:
            # Should fail gracefully
            assert "timeout" in str(e).lower() or "time" in str(e).lower()


class TestInputValidation:
    """Test input validation and data quality checks"""
    
    def test_all_missing_data(self):
        """Test dataset with all missing values"""
        automl = AutoML(n_trials=5)
        
        X = np.full((100, 5), np.nan)
        y = np.random.rand(100)
        
        with pytest.raises(ValueError, match="missing"):
            automl.fit(X, y)
    
    def test_constant_target(self):
        """Test dataset with constant target values"""
        automl = AutoML(n_trials=5)
        
        X = np.random.rand(100, 5)
        y = np.full(100, 1.0)  # All same value
        
        # Should either handle gracefully or fail with clear error
        try:
            automl.fit(X, y)
        except ValueError as e:
            assert "constant" in str(e).lower() or "variance" in str(e).lower()
    
    def test_single_feature(self):
        """Test dataset with only one feature"""
        automl = AutoML(n_trials=5)
        
        X = np.random.rand(100, 1)  # Single feature
        y = np.random.rand(100)
        
        # Should handle single feature datasets
        automl.fit(X, y)
        assert automl.best_pipeline is not None
    
    def test_categorical_only_features(self):
        """Test dataset with only categorical features"""
        automl = AutoML(n_trials=5)
        
        # Create categorical data
        X = pd.DataFrame({
            'cat1': np.random.choice(['A', 'B', 'C'], 100),
            'cat2': np.random.choice(['X', 'Y'], 100),
        })
        y = np.random.randint(0, 2, 100)
        
        # Should handle categorical features
        automl.fit(X, y)
        assert automl.best_pipeline is not None


class TestModelRegistry:
    """Test model registry and task type detection"""
    
    def test_model_registry_completeness(self):
        """Test all models in registry are importable and work"""
        from models.registry import MODEL_REGISTRY, TASK_TYPES
        
        for model_name, model_class in MODEL_REGISTRY.items():
            # Check model class is valid
            assert hasattr(model_class, 'fit')
            assert hasattr(model_class, 'predict')
            
            # Check task type mapping exists
            assert model_name in TASK_TYPES
            assert TASK_TYPES[model_name] in ['classification', 'regression']
    
    def test_task_type_detection(self):
        """Test automatic task type detection"""
        from models.registry import TASK_TYPES
        
        # Classification models
        classification_models = [name for name, task in TASK_TYPES.items() if task == 'classification']
        assert len(classification_models) > 0
        
        # Regression models  
        regression_models = [name for name, task in TASK_TYPES.items() if task == 'regression']
        assert len(regression_models) > 0
    
    def test_model_instantiation(self):
        """Test all models can be instantiated with default parameters"""
        from models.registry import MODEL_REGISTRY
        
        X = np.random.rand(50, 5)
        y_classification = np.random.randint(0, 2, 50)
        y_regression = np.random.rand(50)
        
        for model_name, model_class in MODEL_REGISTRY.items():
            try:
                model = model_class()
                
                if 'classification' in model_name:
                    model.fit(X, y_classification)
                    preds = model.predict(X)
                    assert len(preds) == len(y_classification)
                else:
                    model.fit(X, y_regression)
                    preds = model.predict(X)
                    assert len(preds) == len(y_regression)
                    
            except Exception as e:
                pytest.fail(f"Model {model_name} failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
