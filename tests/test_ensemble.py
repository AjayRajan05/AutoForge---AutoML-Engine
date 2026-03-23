import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from ensemble.stacker import AdvancedStacker, AdaptiveStacker, Stacker
from ensemble.blender import Blender


class TestEnsembleMethods:
    """Test ensemble learning methods"""
    
    def setup_method(self):
        """Setup test data"""
        self.X, self.y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
        self.base_models = [
            RandomForestClassifier(n_estimators=10, random_state=42),
            LogisticRegression(random_state=42),
        ]
    
    def test_basic_stacker(self):
        """Test basic stacking functionality"""
        stacker = Stacker(self.base_models)
        stacker.fit(self.X, self.y)
        
        preds = stacker.predict(self.X)
        assert len(preds) == len(self.y)
        
        # Check accuracy is reasonable
        accuracy = np.mean(preds == self.y)
        assert accuracy > 0.5  # Should be better than random
    
    def test_advanced_stacker_classification(self):
        """Test advanced stacker for classification"""
        stacker = AdvancedStacker(self.base_models, task_type="classification")
        stacker.fit(self.X, self.y)
        
        # Test predictions
        preds = stacker.predict(self.X)
        assert len(preds) == len(self.y)
        
        # Test probabilities
        probs = stacker.predict_proba(self.X)
        assert probs.shape[0] == len(self.y)
        assert probs.shape[1] == 2  # Binary classification
        
        # Test feature importance
        importance = stacker.get_feature_importance()
        assert len(importance) == len(self.base_models)
    
    def test_advanced_stacker_regression(self):
        """Test advanced stacker for regression"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.datasets import make_regression
        
        X_reg, y_reg = make_regression(n_samples=200, n_features=10, random_state=42)
        reg_models = [
            RandomForestRegressor(n_estimators=10, random_state=42),
            Ridge(random_state=42),
        ]
        
        stacker = AdvancedStacker(reg_models, task_type="regression")
        stacker.fit(X_reg, y_reg)
        
        preds = stacker.predict(X_reg)
        assert len(preds) == len(y_reg)
        
        # Should not have predict_proba for regression
        with pytest.raises(ValueError):
            stacker.predict_proba(X_reg)
    
    def test_blender_ensemble(self):
        """Test blender ensemble"""
        blender = Blender(self.base_models)
        blender.fit(self.X, self.y)
        
        preds = blender.predict(self.X)
        assert len(preds) == len(self.y)
        
        # Check ensemble performance
        accuracy = np.mean(preds == self.y)
        assert accuracy > 0.5
    
    def test_single_model_ensemble(self):
        """Test ensemble with single model"""
        single_model = [RandomForestClassifier(n_estimators=10, random_state=42)]
        
        stacker = AdvancedStacker(single_model, task_type="classification")
        stacker.fit(self.X, self.y)
        
        preds = stacker.predict(self.X)
        assert len(preds) == len(self.y)
    
    def test_adaptive_stacker(self):
        """Test adaptive stacker with model selection"""
        # Create models with varying performance
        good_model = RandomForestClassifier(n_estimators=50, random_state=42)
        bad_model = LogisticRegression(max_iter=10)  # Poorly configured
        
        all_models = [good_model, bad_model]
        
        stacker = AdaptiveStacker(all_models, task_type="classification", 
                                 max_models=1, performance_threshold=0.1)
        stacker.fit(self.X, self.y)
        
        preds = stacker.predict(self.X)
        assert len(preds) == len(self.y)
    
    def test_ensemble_with_different_model_types(self):
        """Test ensemble with heterogeneous models"""
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        
        mixed_models = [
            RandomForestClassifier(n_estimators=10, random_state=42),
            SVC(probability=True, random_state=42),
            GaussianNB(),
        ]
        
        stacker = AdvancedStacker(mixed_models, task_type="classification")
        stacker.fit(self.X, self.y)
        
        preds = stacker.predict(self.X)
        probs = stacker.predict_proba(self.X)
        
        assert len(preds) == len(self.y)
        assert probs.shape == (len(self.y), 2)
    
    def test_ensemble_cross_validation(self):
        """Test ensemble cross-validation stability"""
        scores = []
        
        for seed in range(3):  # Multiple runs with different seeds
            X, y = make_classification(n_samples=100, n_features=10, random_state=seed)
            
            stacker = AdvancedStacker(self.base_models, task_type="classification")
            stacker.fit(X, y)
            preds = stacker.predict(X)
            
            accuracy = np.mean(preds == y)
            scores.append(accuracy)
        
        # Performance should be relatively stable
        score_std = np.std(scores)
        assert score_std < 0.1  # Less than 10% variation


class TestEnsembleEdgeCases:
    """Test ensemble edge cases and error handling"""
    
    def test_empty_base_models(self):
        """Test ensemble with no base models"""
        with pytest.raises(ValueError):
            AdvancedStacker([], task_type="classification")
    
    def test_very_small_dataset(self):
        """Test ensemble with very small dataset"""
        X_small = np.random.rand(10, 2)
        y_small = np.random.randint(0, 2, 10)
        
        models = [RandomForestClassifier(n_estimators=5, random_state=42)]
        stacker = AdvancedStacker(models, task_type="classification", n_folds=2)
        
        # Should handle small datasets gracefully
        stacker.fit(X_small, y_small)
        preds = stacker.predict(X_small)
        assert len(preds) == len(y_small)
    
    def test_imbalanced_dataset(self):
        """Test ensemble with imbalanced dataset"""
        from sklearn.datasets import make_classification
        
        X_imb, y_imb = make_classification(
            n_samples=200, n_features=10, weights=[0.9, 0.1], random_state=42
        )
        
        stacker = AdvancedStacker(self.base_models, task_type="classification")
        stacker.fit(X_imb, y_imb)
        
        preds = stacker.predict(X_imb)
        assert len(preds) == len(y_imb)
    
    def test_noisy_features(self):
        """Test ensemble with noisy features"""
        X_noisy = np.random.rand(100, 50)  # Many noisy features
        y_clean = np.random.randint(0, 2, 100)
        
        stacker = AdvancedStacker(self.base_models, task_type="classification")
        stacker.fit(X_noisy, y_clean)
        
        preds = stacker.predict(X_noisy)
        assert len(preds) == len(y_clean)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
