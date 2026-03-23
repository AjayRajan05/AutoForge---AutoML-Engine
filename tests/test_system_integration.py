import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import json
import subprocess
import sys
from pathlib import Path


class TestSystemIntegration:
    """Test complete AutoML system flow with real data"""
    
    def setup_method(self):
        """Setup test environment with real datasets"""
        np.random.seed(42)
        
        # Create realistic classification dataset
        self.X_class = np.random.rand(100, 5)
        # Create meaningful classification target
        self.y_class = (self.X_class[:, 0] + self.X_class[:, 1] > 1.0).astype(int)
        
        # Create realistic regression dataset
        self.X_reg = np.random.rand(100, 5)
        # Create meaningful regression target (linear combination + noise)
        self.y_reg = (2 * self.X_reg[:, 0] + 3 * self.X_reg[:, 1] - 
                     1.5 * self.X_reg[:, 2] + np.random.normal(0, 0.1, 100))
        
        # Create temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV files
        self.class_csv = os.path.join(self.temp_dir, "classification_data.csv")
        self.reg_csv = os.path.join(self.temp_dir, "regression_data.csv")
        
        # Save datasets to CSV
        class_df = pd.DataFrame(self.X_class, columns=[f"feature_{i}" for i in range(5)])
        class_df['target'] = self.y_class
        class_df.to_csv(self.class_csv, index=False)
        
        reg_df = pd.DataFrame(self.X_reg, columns=[f"feature_{i}" for i in range(5)])
        reg_df['target'] = self.y_reg
        reg_df.to_csv(self.reg_csv, index=False)
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_classification_flow(self):
        """Test complete AutoML flow for classification"""
        print("\n🧪 Testing Complete Classification Flow...")
        
        # Import the real AutoML system
        from api.automl import AutoML
        
        # Initialize AutoML with small number of trials for testing
        automl = AutoML(n_trials=5, cv=3)
        
        # Test the complete flow
        print("1. Input validation...")
        assert automl.best_pipeline is None
        assert automl.task_type is None
        
        print("2. Training AutoML...")
        automl.fit(self.X_class, self.y_class)
        
        print("3. Verifying results...")
        assert automl.task_type == "classification"
        assert automl.best_pipeline is not None
        assert automl.dataset_profile is not None
        
        print("4. Making predictions...")
        predictions = automl.predict(self.X_class[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
        
        print("5. Testing probability predictions...")
        if hasattr(automl.best_pipeline, 'predict_proba'):
            probabilities = automl.predict_proba(self.X_class[:10])
            assert probabilities.shape[0] == 10
            assert probabilities.shape[1] == 2  # Binary classification
        
        print("✅ Classification flow test passed!")
    
    def test_complete_regression_flow(self):
        """Test complete AutoML flow for regression"""
        print("\n🧪 Testing Complete Regression Flow...")
        
        from api.automl import AutoML
        
        # Initialize AutoML
        automl = AutoML(n_trials=5, cv=3)
        
        print("1. Training AutoML...")
        automl.fit(self.X_reg, self.y_reg)
        
        print("2. Verifying results...")
        assert automl.task_type == "regression"
        assert automl.best_pipeline is not None
        assert automl.dataset_profile is not None
        
        print("3. Making predictions...")
        predictions = automl.predict(self.X_reg[:10])
        assert len(predictions) == 10
        assert all(isinstance(pred, (int, float)) for pred in predictions)
        
        print("4. Testing prediction quality...")
        # Check that predictions are reasonable (not all same value)
        assert len(set(predictions)) > 1
        
        print("✅ Regression flow test passed!")
    
    def test_cli_classification_flow(self):
        """Test complete CLI flow for classification"""
        print("\n🧪 Testing CLI Classification Flow...")
        
        model_path = os.path.join(self.temp_dir, "classification_model.pkl")
        pred_path = os.path.join(self.temp_dir, "classification_predictions.csv")
        
        # Test training command
        print("1. Testing train command...")
        train_cmd = [
            sys.executable, "-m", "cli.main", "train",
            self.class_csv,
            "--target", "target",
            "--trials", "3",
            "--output", model_path,
            "--verbose"
        ]
        
        result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=".")
        
        print(f"Train command output: {result.stdout}")
        if result.stderr:
            print(f"Train command errors: {result.stderr}")
        
        assert result.returncode == 0
        assert os.path.exists(model_path)
        
        # Test prediction command
        print("2. Testing predict command...")
        pred_cmd = [
            sys.executable, "-m", "cli.main", "predict",
            model_path,
            self.class_csv,
            "--output", pred_path,
            "--probabilities"
        ]
        
        result = subprocess.run(pred_cmd, capture_output=True, text=True, cwd=".")
        
        print(f"Predict command output: {result.stdout}")
        if result.stderr:
            print(f"Predict command errors: {result.stderr}")
        
        assert result.returncode == 0
        assert os.path.exists(pred_path)
        
        # Verify predictions
        pred_df = pd.read_csv(pred_path)
        assert len(pred_df) == 100
        assert 'predictions' in pred_df.columns
        assert all(pred in [0, 1] for pred in pred_df['predictions'])
        
        print("✅ CLI classification flow test passed!")
    
    def test_cli_regression_flow(self):
        """Test complete CLI flow for regression"""
        print("\n🧪 Testing CLI Regression Flow...")
        
        model_path = os.path.join(self.temp_dir, "regression_model.pkl")
        pred_path = os.path.join(self.temp_dir, "regression_predictions.csv")
        
        # Test training command
        print("1. Testing train command...")
        train_cmd = [
            sys.executable, "-m", "cli.main", "train",
            self.reg_csv,
            "--target", "target",
            "--trials", "3",
            "--output", model_path
        ]
        
        result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=".")
        
        print(f"Train command output: {result.stdout}")
        if result.stderr:
            print(f"Train command errors: {result.stderr}")
        
        assert result.returncode == 0
        assert os.path.exists(model_path)
        
        # Test prediction command
        print("2. Testing predict command...")
        pred_cmd = [
            sys.executable, "-m", "cli.main", "predict",
            model_path,
            self.reg_csv,
            "--output", pred_path
        ]
        
        result = subprocess.run(pred_cmd, capture_output=True, text=True, cwd=".")
        
        assert result.returncode == 0
        assert os.path.exists(pred_path)
        
        # Verify predictions
        pred_df = pd.read_csv(pred_path)
        assert len(pred_df) == 100
        assert 'predictions' in pred_df.columns
        assert all(isinstance(pred, (int, float)) for pred in pred_df['predictions'])
        
        print("✅ CLI regression flow test passed!")
    
    def test_validate_command(self):
        """Test CLI validate command"""
        print("\n🧪 Testing Validate Command...")
        
        validate_cmd = [
            sys.executable, "-m", "cli.main", "validate",
            "--data-path", self.class_csv,
            "--target", "target",
            "--verbose"
        ]
        
        result = subprocess.run(validate_cmd, capture_output=True, text=True, cwd=".")
        
        print(f"Validate command output: {result.stdout}")
        if result.stderr:
            print(f"Validate command errors: {result.stderr}")
        
        assert result.returncode == 0
        assert "Dataset Validation" in result.stdout
        assert "Task type: classification" in result.stdout
        
        print("✅ Validate command test passed!")
    
    def test_logs_command(self):
        """Test CLI logs command"""
        print("\n🧪 Testing Logs Command...")
        
        # First run a quick training to generate logs
        from api.automl import AutoML
        automl = AutoML(n_trials=2, cv=2)
        automl.fit(self.X_class[:20], self.y_class[:20])  # Small dataset for speed
        
        # Test logs command
        logs_cmd = [
            sys.executable, "-m", "cli.main", "logs",
            "--limit", "5",
            "--verbose"
        ]
        
        result = subprocess.run(logs_cmd, capture_output=True, text=True, cwd=".")
        
        print(f"Logs command output: {result.stdout}")
        if result.stderr:
            print(f"Logs command errors: {result.stderr}")
        
        assert result.returncode == 0
        
        print("✅ Logs command test passed!")
    
    def test_knowledge_base_integration(self):
        """Test knowledge base integration with real data"""
        print("\n🧪 Testing Knowledge Base Integration...")
        
        from meta_learning.knowledge_base import KnowledgeBase
        from meta_learning.self_improver import SelfImprover
        
        # Initialize knowledge base
        kb_path = os.path.join(self.temp_dir, "test_kb.json")
        kb = KnowledgeBase(kb_path)
        
        # Test initial state
        records = kb.load()
        assert isinstance(records, list)
        
        # Test saving a record
        test_record = {
            "run_id": "test_run_1",
            "timestamp": "2023-01-01T00:00:00",
            "model": "classification_random_forest",
            "params": {"n_estimators": 100},
            "metrics": {"cv_score": 0.85}
        }
        
        success = kb.save_record(test_record)
        assert success
        
        # Test loading records
        records = kb.load()
        assert len(records) == 1
        assert records[0]["run_id"] == "test_run_1"
        
        # Test self-improver
        improver = SelfImprover(kb_path)
        success = improver.analyze()
        assert success
        
        best = improver.get_best()
        assert "model_priority" in best
        assert len(best["model_priority"]) > 0
        
        print("✅ Knowledge base integration test passed!")
    
    def test_model_registry_integration(self):
        """Test model registry with real models"""
        print("\n🧪 Testing Model Registry Integration...")
        
        from models.registry import MODEL_REGISTRY, TASK_TYPES
        
        # Test that all models can be imported
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from xgboost import XGBClassifier, XGBRegressor
        
        # Test that all models are properly registered
        assert len(MODEL_REGISTRY) >= 14
        assert len(TASK_TYPES) >= 14
        
        # Test that all models can be instantiated
        for model_name, model_class in MODEL_REGISTRY.items():
            try:
                model = model_class()
                print(f"  ✅ {model_name}: {model_class}")
            except Exception as e:
                pytest.fail(f"Failed to instantiate {model_name}: {e}")
        
        # Test task type mapping
        for model_name, task_type in TASK_TYPES.items():
            assert task_type in ["classification", "regression"]
            assert model_name.startswith("classification_") or model_name.startswith("regression_")
        
        print("✅ Model registry integration test passed!")
    
    def test_error_handling_integration(self):
        """Test error handling with real scenarios"""
        print("\n🧪 Testing Error Handling Integration...")
        
        from api.automl import AutoML, validate_input, detect_task_type
        
        # Test input validation errors
        with pytest.raises(ValueError, match="mismatched dimensions"):
            validate_input(np.random.rand(50, 5), np.random.rand(100))
        
        with pytest.raises(ValueError, match="too small"):
            validate_input(np.random.rand(5, 5), np.random.rand(5))
        
        # Test task type detection
        y_binary = np.array([0, 1, 0, 1, 0])
        assert detect_task_type(y_binary) == "classification"
        
        y_continuous = np.array([1.1, 2.3, 1.8, 2.1, 1.9])
        assert detect_task_type(y_continuous) == "regression"
        
        # Test AutoML error handling
        automl = AutoML()
        
        # Test prediction before fitting
        with pytest.raises(ValueError, match="Model not fitted"):
            automl.predict(np.random.rand(10, 5))
        
        # Test predict_proba on regression
        automl.task_type = "regression"
        with pytest.raises(ValueError, match="only available for classification"):
            automl.predict_proba(np.random.rand(10, 5))
        
        print("✅ Error handling integration test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
