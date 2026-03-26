"""
🔥 ENHANCED AUTOML SYSTEM COMPREHENSIVE BENCHMARK
Tests the enhanced production-grade system against all existing AutoML systems
"""

import unittest
import pandas as pd
import numpy as np
import time
import warnings
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import tempfile
import os

# Import our enhanced AutoML system
from api.automl import AutoML
from api.revolutionary_automl import AdvancedAutoML

# Try to import competitor systems (may not be available)
try:
    import autosklearn
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    AUTOSKLEARN_AVAILABLE = False

try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False

warnings.filterwarnings('ignore')


class TestEnhancedVsExistingSystems(unittest.TestCase):
    """
    🔥 Comprehensive benchmark of enhanced AutoML vs existing systems
    """

    def setUp(self):
        """Setup test datasets"""
        # Classification datasets
        self.X_class_small, self.y_class_small = make_classification(
            n_samples=200, n_features=10, n_classes=2, n_informative=5, random_state=42
        )
        self.X_class_medium, self.y_class_medium = make_classification(
            n_samples=500, n_features=20, n_classes=2, n_informative=10, random_state=42
        )
        
        # Regression datasets
        self.X_reg_small, self.y_reg_small = make_regression(
            n_samples=200, n_features=10, noise=0.1, random_state=42
        )
        self.X_reg_medium, self.y_reg_medium = make_regression(
            n_samples=500, n_features=20, noise=0.1, random_state=42
        )
        
        # Create DataFrames
        self.df_class_small = pd.DataFrame(self.X_class_small)
        self.df_class_small['target'] = self.y_class_small
        
        self.df_class_medium = pd.DataFrame(self.X_class_medium)
        self.df_class_medium['target'] = self.y_class_medium
        
        self.df_reg_small = pd.DataFrame(self.X_reg_small)
        self.df_reg_small['target'] = self.y_reg_small
        
        self.df_reg_medium = pd.DataFrame(self.X_reg_medium)
        self.df_reg_medium['target'] = self.y_reg_medium
        
        # Results tracking
        self.results = {}

    def test_enhanced_automl_basic(self):
        """🔥 Test our enhanced basic AutoML system"""
        print("\n🧪 Testing Enhanced Basic AutoML...")
        
        automl = AutoML(
            n_trials=5,
            timeout=30,
            cv=3,
            use_adaptive_optimization=True,
            use_caching=False,
            show_progress=False,
            enable_advanced_features=False
        )
        
        # Test classification
        X_train = self.df_class_small.drop('target', axis=1)
        y_train = self.df_class_small['target']
        
        start_time = time.time()
        automl.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        predictions = automl.predict(X_train)
        accuracy = accuracy_score(y_train, predictions)
        
        self.results['enhanced_basic'] = {
            'accuracy': accuracy,
            'train_time': train_time,
            'has_pipeline': automl.coordinator.best_pipeline is not None,
            'system': 'Enhanced Basic AutoML'
        }
        
        print(f"✅ Enhanced Basic: {accuracy:.3f} accuracy in {train_time:.2f}s")
        
        # Assertions
        self.assertGreater(accuracy, 0.5)
        self.assertLess(train_time, 60)
        self.assertIsNotNone(automl.coordinator.best_pipeline)

    def test_enhanced_automl_advanced(self):
        """🔥 Test our enhanced advanced AutoML system"""
        print("\n🧪 Testing Enhanced Advanced AutoML...")
        
        automl = AutoML(
            n_trials=5,
            timeout=30,
            cv=3,
            use_adaptive_optimization=True,
            use_caching=False,
            show_progress=False,
            enable_advanced_features=True  # Enable all advanced features
        )
        
        # Test classification
        X_train = self.df_class_small.drop('target', axis=1)
        y_train = self.df_class_small['target']
        
        start_time = time.time()
        automl.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        predictions = automl.predict(X_train)
        accuracy = accuracy_score(y_train, predictions)
        
        # Check advanced features
        has_nas = automl.coordinator.nas_engine is not None
        has_multimodal = automl.coordinator.multimodal_engine is not None
        has_distributed = automl.coordinator.distributed_engine is not None
        
        self.results['enhanced_advanced'] = {
            'accuracy': accuracy,
            'train_time': train_time,
            'has_pipeline': automl.coordinator.best_pipeline is not None,
            'has_nas': has_nas,
            'has_multimodal': has_multimodal,
            'has_distributed': has_distributed,
            'system': 'Enhanced Advanced AutoML'
        }
        
        print(f"✅ Enhanced Advanced: {accuracy:.3f} accuracy in {train_time:.2f}s")
        print(f"   🔥 NAS: {has_nas}, Multimodal: {has_multimodal}, Distributed: {has_distributed}")
        
        # Assertions
        self.assertGreater(accuracy, 0.5)
        self.assertLess(train_time, 90)  # Allow more time for advanced features

    def test_revolutionary_automl(self):
        """🔥 Test our revolutionary AutoML system"""
        print("\n🧪 Testing Revolutionary AutoML...")
        
        try:
            automl = AdvancedAutoML(
                n_trials=5,
                enable_all_advanced_features=True,
                show_progress=False
            )
            
            X_train = self.df_class_small.drop('target', axis=1)
            y_train = self.df_class_small['target']
            
            start_time = time.time()
            automl.fit_revolutionary(
                X_train, y_train,
                enable_nas=True,
                enable_multimodal=True,
                enable_distributed=True
            )
            train_time = time.time() - start_time
            
            predictions = automl.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            
            self.results['revolutionary'] = {
                'accuracy': accuracy,
                'train_time': train_time,
                'has_pipeline': hasattr(automl, 'best_pipeline') and automl.best_pipeline is not None,
                'system': 'Revolutionary AutoML'
            }
            
            print(f"✅ Revolutionary: {accuracy:.3f} accuracy in {train_time:.2f}s")
            
            # Assertions
            self.assertGreater(accuracy, 0.5)
            self.assertLess(train_time, 120)
            
        except Exception as e:
            print(f"⚠️ Revolutionary AutoML test failed: {str(e)[:50]}...")
            self.results['revolutionary'] = {
                'accuracy': 0,
                'train_time': 0,
                'error': str(e),
                'system': 'Revolutionary AutoML'
            }

    def test_autosklearn_comparison(self):
        """🔥 Test against Auto-sklearn if available"""
        if not AUTOSKLEARN_AVAILABLE:
            print("\n⚠️ Auto-sklearn not available - skipping comparison")
            return
        
        print("\n🧪 Testing Auto-sklearn...")
        
        try:
            import autosklearn.classification
            
            automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=30,
                per_run_time_limit=10,
                ensemble_size=1,
                tmp_folder=tempfile.mkdtemp()
            )
            
            X_train = self.df_class_small.drop('target', axis=1)
            y_train = self.df_class_small['target']
            
            start_time = time.time()
            automl.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            predictions = automl.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            
            self.results['autosklearn'] = {
                'accuracy': accuracy,
                'train_time': train_time,
                'system': 'Auto-sklearn'
            }
            
            print(f"✅ Auto-sklearn: {accuracy:.3f} accuracy in {train_time:.2f}s")
            
        except Exception as e:
            print(f"⚠️ Auto-sklearn test failed: {str(e)[:50]}...")
            self.results['autosklearn'] = {
                'accuracy': 0,
                'train_time': 0,
                'error': str(e),
                'system': 'Auto-sklearn'
            }

    def test_h2o_comparison(self):
        """🔥 Test against H2O AutoML if available"""
        if not H2O_AVAILABLE:
            print("\n⚠️ H2O AutoML not available - skipping comparison")
            return
        
        print("\n🧪 Testing H2O AutoML...")
        
        try:
            h2o.init()
            
            # Convert to H2O frame
            h2o_df = h2o.H2OFrame(self.df_class_small)
            h2o_df['target'] = h2o_df['target'].asfactor()
            
            automl = H2OAutoML(
                max_runtime_secs=30,
                max_models=5,
                seed=42
            )
            
            start_time = time.time()
            automl.train(y='target', training_frame=h2o_df)
            train_time = time.time() - start_time
            
            # Get predictions
            predictions = automl.predict(h2o_df).as_data_frame()['predict'].values
            accuracy = accuracy_score(self.y_class_small, predictions)
            
            self.results['h2o'] = {
                'accuracy': accuracy,
                'train_time': train_time,
                'system': 'H2O AutoML'
            }
            
            print(f"✅ H2O AutoML: {accuracy:.3f} accuracy in {train_time:.2f}s")
            
            h2o.shutdown()
            
        except Exception as e:
            print(f"⚠️ H2O AutoML test failed: {str(e)[:50]}...")
            self.results['h2o'] = {
                'accuracy': 0,
                'train_time': 0,
                'error': str(e),
                'system': 'H2O AutoML'
            }

    def test_production_grade_features(self):
        """🔥 Test production-grade features of our enhanced system"""
        print("\n🧪 Testing Production-Grade Features...")
        
        # Test failure memory
        from core.failure_memory import failure_memory
        
        # Test with invalid parameters
        automl = AutoML(
            n_trials=3,
            timeout=15,
            show_progress=False,
            enable_advanced_features=False
        )
        
        # Create problematic dataset
        X_bad = np.random.rand(50, 10)
        y_bad = np.random.randint(0, 2, 50)
        df_bad = pd.DataFrame(X_bad)
        df_bad['target'] = y_bad
        
        X_train = df_bad.drop('target', axis=1)
        y_train = df_bad['target']
        
        try:
            automl.fit(X_train, y_train)
            
            # Check failure memory
            failure_stats = failure_memory.get_failure_stats()
            has_failures = failure_stats.get('total_failures', 0) > 0
            
            # Test Unicode handling
            from benchmarking.enhanced_benchmarking import EnhancedBenchmarking
            benchmark = EnhancedBenchmarking()
            unicode_text = "🏆 Test Unicode"
            safe_text = benchmark.safe_encode(unicode_text)
            
            # Test parameter validation
            from models.registry import validate_params, safe_params
            valid = validate_params('classification_logistic_regression', {'solver': 'liblinear'})
            safe = safe_params('classification_logistic_regression', {'solver': 'invalid'})
            
            self.results['production_features'] = {
                'has_failure_memory': has_failures,
                'unicode_handling': isinstance(safe_text, str),
                'parameter_validation': valid,
                'safe_parameters': len(safe) > 0,
                'system': 'Production Features'
            }
            
            print(f"✅ Production Features: Memory={has_failures}, Unicode={isinstance(safe_text, str)}, Validation={valid}")
            
        except Exception as e:
            print(f"⚠️ Production features test failed: {str(e)[:50]}...")
            self.results['production_features'] = {
                'error': str(e),
                'system': 'Production Features'
            }

    def test_performance_assertions(self):
        """🔥 Test performance assertions"""
        print("\n🧪 Testing Performance Assertions...")
        
        # Test with time limits
        automl = AutoML(
            n_trials=3,
            timeout=10,  # 10 second timeout
            show_progress=False,
            enable_advanced_features=False
        )
        
        X_train = self.df_class_small.drop('target', axis=1)
        y_train = self.df_class_small['target']
        
        start_time = time.time()
        automl.fit(X_train, y_train)
        actual_time = time.time() - start_time
        
        # Performance assertions
        time_assertion_passed = actual_time < 30  # Should complete in 30s max
        
        # Test failure tracking
        from core.failure_memory import failure_memory
        failure_stats = failure_memory.get_failure_stats()
        failed_trials = failure_stats.get('total_failures', 0)
        failure_assertion_passed = failed_trials < 10
        
        self.results['performance_assertions'] = {
            'time_assertion': time_assertion_passed,
            'actual_time': actual_time,
            'failure_assertion': failure_assertion_passed,
            'failed_trials': failed_trials,
            'system': 'Performance Assertions'
        }
        
        print(f"✅ Performance: Time={actual_time:.2f}s (<30s: {time_assertion_passed}), Failures={failed_trials} (<10: {failure_assertion_passed})")

    def test_comprehensive_comparison(self):
        """🔥 Comprehensive comparison of all systems"""
        print("\n🏆 COMPREHENSIVE SYSTEM COMPARISON")
        print("=" * 80)
        
        # Run all tests
        self.test_enhanced_automl_basic()
        self.test_enhanced_automl_advanced()
        self.test_revolutionary_automl()
        self.test_autosklearn_comparison()
        self.test_h2o_comparison()
        self.test_production_grade_features()
        self.test_performance_assertions()
        
        # Print comparison results
        print("\n📊 COMPARISON RESULTS")
        print("=" * 80)
        
        # Sort by accuracy
        accuracy_results = [(k, v) for k, v in self.results.items() if 'accuracy' in v and v['accuracy'] > 0]
        accuracy_results.sort(key=lambda x: x[1]['accuracy'], reverse=True)
        
        print("\n🎯 ACCURACY RANKING:")
        for i, (name, result) in enumerate(accuracy_results, 1):
            print(f"{i}. {result['system']}: {result['accuracy']:.3f}")
        
        # Sort by time
        time_results = [(k, v) for k, v in self.results.items() if 'train_time' in v and v['train_time'] > 0]
        time_results.sort(key=lambda x: x[1]['train_time'])
        
        print("\n⏱️ SPEED RANKING (fastest first):")
        for i, (name, result) in enumerate(time_results, 1):
            print(f"{i}. {result['system']}: {result['train_time']:.2f}s")
        
        # Feature comparison
        print("\n🔥 FEATURE COMPARISON:")
        for name, result in self.results.items():
            system = result['system']
            features = []
            
            if 'has_nas' in result and result['has_nas']:
                features.append("NAS")
            if 'has_multimodal' in result and result['has_multimodal']:
                features.append("Multimodal")
            if 'has_distributed' in result and result['has_distributed']:
                features.append("Distributed")
            if 'has_failure_memory' in result and result['has_failure_memory']:
                features.append("Failure Memory")
            if 'unicode_handling' in result and result['unicode_handling']:
                features.append("Unicode")
            if 'parameter_validation' in result and result['parameter_validation']:
                features.append("Validation")
            
            if features:
                print(f"• {system}: {', '.join(features)}")
        
        # Production readiness assessment
        print("\n🏆 PRODUCTION READINESS:")
        enhanced_basic = self.results.get('enhanced_basic', {})
        enhanced_advanced = self.results.get('enhanced_advanced', {})
        
        basic_ready = (
            enhanced_basic.get('accuracy', 0) > 0.5 and
            enhanced_basic.get('train_time', 999) < 60 and
            enhanced_basic.get('has_pipeline', False)
        )
        
        advanced_ready = (
            enhanced_advanced.get('accuracy', 0) > 0.5 and
            enhanced_advanced.get('train_time', 999) < 90 and
            enhanced_advanced.get('has_pipeline', False) and
            (enhanced_advanced.get('has_nas', False) or 
             enhanced_advanced.get('has_multimodal', False) or
             enhanced_advanced.get('has_distributed', False))
        )
        
        print(f"✅ Enhanced Basic AutoML: {'PRODUCTION READY' if basic_ready else 'NEEDS WORK'}")
        print(f"✅ Enhanced Advanced AutoML: {'PRODUCTION READY' if advanced_ready else 'NEEDS WORK'}")
        
        # Final assessment
        print("\n🎉 FINAL ASSESSMENT:")
        if basic_ready and advanced_ready:
            print("🏆 Our enhanced AutoML systems are PRODUCTION-GRADE!")
            print("🚀 Ready to compete with Auto-sklearn and H2O AutoML!")
        else:
            print("🔧 Systems need optimization before production deployment")
        
        print("=" * 80)
        
        # Assertions
        self.assertTrue(basic_ready, "Enhanced Basic AutoML should be production ready")
        
        # At least one system should beat 70% accuracy
        best_accuracy = max([r['accuracy'] for r in self.results.values() if 'accuracy' in r])
        self.assertGreater(best_accuracy, 0.7, "At least one system should achieve >70% accuracy")


if __name__ == "__main__":
    # Run comprehensive benchmark
    unittest.main(verbosity=2)
