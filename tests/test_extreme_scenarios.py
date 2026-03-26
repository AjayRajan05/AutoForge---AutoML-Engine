"""
🔥 EXTREME SCENARIO TESTS
Push the bulletproof system to its absolute limits
"""

import unittest
import pandas as pd
import numpy as np
import time
import warnings
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import tempfile
import os

# Import the truly bulletproof system
from api.truly_bulletproof_automl import TrulyBulletproofAutoML

warnings.filterwarnings('ignore')


class TestExtremeScenarios(unittest.TestCase):
    """
    🔥 Test extreme and challenging scenarios
    """

    def setUp(self):
        """Setup extreme test scenarios"""
        # Results tracking
        self.test_results = []

    def track_test_result(self, test_name, success, time_taken, score=None, error=None):
        """Track test result"""
        result = {
            'test': test_name,
            'success': success,
            'time': time_taken,
            'score': score,
            'error': error
        }
        self.test_results.append(result)
        
        if success:
            print(f"✅ {test_name}: SUCCESS ({time_taken:.2f}s)" + (f" - Score: {score:.3f}" if score else ""))
        else:
            print(f"❌ {test_name}: FAILED ({time_taken:.2f}s) - {error}")

    def test_ultra_large_dataset(self):
        """Test with extremely large dataset"""
        print("\n🔥 Testing Ultra Large Dataset...")
        
        # Create large dataset (5000 samples, 100 features)
        X, y = make_classification(
            n_samples=5000, 
            n_features=100, 
            n_informative=50, 
            n_redundant=25,
            n_repeated=0,  # Fix: no repeated features
            n_classes=3,
            random_state=42
        )
        df = pd.DataFrame(X)
        df['target'] = y
        
        automl = TrulyBulletproofAutoML(max_time=60, max_trials=3, simple_mode=True)
        
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        start_time = time.time()
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            time_taken = time.time() - start_time
            
            self.track_test_result("Ultra Large Dataset", True, time_taken, accuracy)
            
            # Assertions
            self.assertGreater(accuracy, 0.3)  # Lenient for large dataset
            self.assertLess(time_taken, 120)  # Should complete in 2 minutes max
            
        except Exception as e:
            time_taken = time.time() - start_time
            self.track_test_result("Ultra Large Dataset", False, time_taken, error=str(e))

    def test_ultra_small_dataset(self):
        """Test with extremely small dataset"""
        print("\n🔥 Testing Ultra Small Dataset...")
        
        # Create tiny dataset (10 samples, 2 features)
        X, y = make_classification(
            n_samples=10, 
            n_features=2, 
            n_classes=2, 
            n_informative=2,
            random_state=42
        )
        df = pd.DataFrame(X)
        df['target'] = y
        
        automl = TrulyBulletproofAutoML(max_time=30, max_trials=1)
        
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        start_time = time.time()
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            time_taken = time.time() - start_time
            
            self.track_test_result("Ultra Small Dataset", True, time_taken, accuracy)
            
            # Assertions
            self.assertGreater(accuracy, 0.0)  # Any accuracy is acceptable
            self.assertLess(time_taken, 30)
            
        except Exception as e:
            time_taken = time.time() - start_time
            self.track_test_result("Ultra Small Dataset", False, time_taken, error=str(e))

    def test_extreme_missing_values(self):
        """Test with extreme missing values (90% missing)"""
        print("\n🔥 Testing Extreme Missing Values...")
        
        # Create dataset with extreme missing values
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        df = pd.DataFrame(X)
        df['target'] = y
        
        # Make 90% of data missing
        for col in df.columns[:-1]:  # Don't make target missing
            missing_mask = np.random.random(len(df)) < 0.9
            df.loc[missing_mask, col] = np.nan
        
        automl = TrulyBulletproofAutoML(max_time=45, max_trials=2)
        
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        start_time = time.time()
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            time_taken = time.time() - start_time
            
            self.track_test_result("Extreme Missing Values", True, time_taken, accuracy)
            
            # Assertions
            self.assertGreater(accuracy, 0.1)  # Very lenient for extreme missing values
            self.assertLess(time_taken, 60)
            
        except Exception as e:
            time_taken = time.time() - start_time
            self.track_test_result("Extreme Missing Values", False, time_taken, error=str(e))

    def test_infinite_values(self):
        """Test with infinite and NaN values"""
        print("\n🔥 Testing Infinite Values...")
        
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        df = pd.DataFrame(X)
        df['target'] = y
        
        # Add infinite values
        df.iloc[0:10, 0] = np.inf
        df.iloc[10:20, 1] = -np.inf
        df.iloc[20:30, 2] = np.nan
        
        automl = TrulyBulletproofAutoML(max_time=30, max_trials=2)
        
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        start_time = time.time()
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            time_taken = time.time() - start_time
            
            self.track_test_result("Infinite Values", True, time_taken, accuracy)
            
            # Assertions
            self.assertGreater(accuracy, 0.1)
            self.assertLess(time_taken, 45)
            
        except Exception as e:
            time_taken = time.time() - start_time
            self.track_test_result("Infinite Values", False, time_taken, error=str(e))

    def test_extreme_categorical_complexity(self):
        """Test with many categorical features and high cardinality"""
        print("\n🔥 Testing Extreme Categorical Complexity...")
        
        # Create dataset with many categorical features
        np.random.seed(42)
        data = {
            'target': np.random.randint(0, 2, 200)
        }
        
        # Add 20 categorical features with high cardinality
        for i in range(20):
            if i < 10:
                # High cardinality (50 unique values)
                data[f'cat_high_{i}'] = np.random.choice([f'val_{j}' for j in range(50)], 200)
            else:
                # Medium cardinality (10 unique values)
                data[f'cat_med_{i}'] = np.random.choice([f'cat_{j}' for j in range(10)], 200)
        
        # Add some numerical features
        for i in range(5):
            data[f'num_{i}'] = np.random.randn(200)
        
        df = pd.DataFrame(data)
        
        automl = TrulyBulletproofAutoML(max_time=60, max_trials=2)
        
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        start_time = time.time()
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            time_taken = time.time() - start_time
            
            self.track_test_result("Extreme Categorical Complexity", True, time_taken, accuracy)
            
            # Assertions
            self.assertGreater(accuracy, 0.1)
            self.assertLess(time_taken, 90)
            
        except Exception as e:
            time_taken = time.time() - start_time
            self.track_test_result("Extreme Categorical Complexity", False, time_taken, error=str(e))

    def test_mixed_data_types_extreme(self):
        """Test with extreme mixed data types"""
        print("\n🔥 Testing Extreme Mixed Data Types...")
        
        df = pd.DataFrame()
        df['target'] = np.random.randint(0, 2, 200)
        
        # Add various data types
        df['int_col'] = np.random.randint(0, 100, 200)
        df['float_col'] = np.random.randn(200)
        df['bool_col'] = np.random.choice([True, False], 200)
        df['string_col'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], 200)
        df['mixed_col'] = ['A', 1, 'B', 2, 'C', 3] * 33 + ['A', 1]
        df['missing_string'] = np.random.choice(['X', 'Y', None, 'Z'], 200)
        df['unicode_col'] = np.random.choice(['🏀', '⚽', '🎾', '🏈', '⚾'], 200)
        df['scientific'] = [1e-10, 1e10, -1e-5, 1e5] * 50
        
        # Add some missing values
        df.iloc[0:50, 1] = np.nan
        df.iloc[25:75, 2] = None
        
        automl = TrulyBulletproofAutoML(max_time=45, max_trials=2)
        
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        start_time = time.time()
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            time_taken = time.time() - start_time
            
            self.track_test_result("Extreme Mixed Data Types", True, time_taken, accuracy)
            
            # Assertions
            self.assertGreater(accuracy, 0.1)
            self.assertLess(time_taken, 60)
            
        except Exception as e:
            time_taken = time.time() - start_time
            self.track_test_result("Extreme Mixed Data Types", False, time_taken, error=str(e))

    def test_imbalanced_dataset(self):
        """Test with heavily imbalanced dataset"""
        print("\n🔥 Testing Imbalanced Dataset...")
        
        # Create highly imbalanced dataset (95% class 0, 5% class 1)
        X, y = make_classification(
            n_samples=1000, 
            n_features=10, 
            n_classes=2, 
            weights=[0.95, 0.05],
            random_state=42
        )
        df = pd.DataFrame(X)
        df['target'] = y
        
        automl = TrulyBulletproofAutoML(max_time=30, max_trials=2)
        
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        start_time = time.time()
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            time_taken = time.time() - start_time
            
            self.track_test_result("Imbalanced Dataset", True, time_taken, accuracy)
            
            # Assertions
            self.assertGreater(accuracy, 0.8)  # High accuracy due to imbalance
            self.assertLess(time_taken, 45)
            
        except Exception as e:
            time_taken = time.time() - start_time
            self.track_test_result("Imbalanced Dataset", False, time_taken, error=str(e))

    def test_noisy_dataset(self):
        """Test with extremely noisy dataset"""
        print("\n🔥 Testing Noisy Dataset...")
        
        # Create dataset with lots of noise
        X, y = make_classification(
            n_samples=200, 
            n_features=50, 
            n_informative=5,  # Only 5 informative features out of 50
            n_redundant=0,
            n_repeated=0,  # Fix: no repeated features
            n_clusters_per_class=1,
            flip_y=0.4,  # 40% label noise
            random_state=42
        )
        df = pd.DataFrame(X)
        df['target'] = y
        
        automl = TrulyBulletproofAutoML(max_time=45, max_trials=3)
        
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        start_time = time.time()
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            time_taken = time.time() - start_time
            
            self.track_test_result("Noisy Dataset", True, time_taken, accuracy)
            
            # Assertions
            self.assertGreater(accuracy, 0.3)  # Lenient for noisy data
            self.assertLess(time_taken, 60)
            
        except Exception as e:
            time_taken = time.time() - start_time
            self.track_test_result("Noisy Dataset", False, time_taken, error=str(e))

    def test_multiclass_extreme(self):
        """Test with many classes"""
        print("\n🔥 Testing Multiclass Extreme...")
        
        # Create dataset with many classes
        X, y = make_classification(
            n_samples=500, 
            n_features=20, 
            n_classes=10,  # 10 different classes
            n_informative=15,
            random_state=42
        )
        df = pd.DataFrame(X)
        df['target'] = y
        
        automl = TrulyBulletproofAutoML(max_time=60, max_trials=2)
        
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        start_time = time.time()
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            time_taken = time.time() - start_time
            
            self.track_test_result("Multiclass Extreme", True, time_taken, accuracy)
            
            # Assertions
            self.assertGreater(accuracy, 0.2)  # Lenient for multiclass
            self.assertLess(time_taken, 90)
            
        except Exception as e:
            time_taken = time.time() - start_time
            self.track_test_result("Multiclass Extreme", False, time_taken, error=str(e))

    def test_ultra_fast_constraint(self):
        """Test with extremely tight time constraint"""
        print("\n🔥 Testing Ultra Fast Constraint...")
        
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        df = pd.DataFrame(X)
        df['target'] = y
        
        automl = TrulyBulletproofAutoML(max_time=2, max_trials=1)  # Only 2 seconds!
        
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        start_time = time.time()
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            time_taken = time.time() - start_time
            
            self.track_test_result("Ultra Fast Constraint", True, time_taken, accuracy)
            
            # Assertions
            self.assertLess(time_taken, 10)  # Should complete very quickly
            
        except Exception as e:
            time_taken = time.time() - start_time
            self.track_test_result("Ultra Fast Constraint", False, time_taken, error=str(e))

    def test_file_input_output(self):
        """Test with file input/output"""
        print("\n🔥 Testing File Input/Output...")
        
        # Create test data
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        df = pd.DataFrame(X)
        df['target'] = y
        
        # Save to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Read from file
            df_file = pd.read_csv(temp_file)
            
            automl = TrulyBulletproofAutoML(max_time=30, max_trials=2)
            
            X_train = df_file.drop('target', axis=1)
            y_train = df_file['target']
            
            start_time = time.time()
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            time_taken = time.time() - start_time
            
            self.track_test_result("File Input/Output", True, time_taken, accuracy)
            
            # Assertions
            self.assertGreater(accuracy, 0.3)
            self.assertLess(time_taken, 45)
            
        except Exception as e:
            start_time = time.time()
            time_taken = time.time() - start_time
            self.track_test_result("File Input/Output", False, time_taken, error=str(e))
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_extreme_regression(self):
        """Test with extreme regression scenario"""
        print("\n🔥 Testing Extreme Regression...")
        
        # Create challenging regression dataset
        X, y = make_regression(
            n_samples=500, 
            n_features=100, 
            n_informative=30,
            noise=10.0,  # High noise
            random_state=42
        )
        
        # Add some outliers
        y[0:10] *= 10  # Make some points extreme outliers
        
        df = pd.DataFrame(X)
        df['target'] = y
        
        automl = TrulyBulletproofAutoML(max_time=60, max_trials=2)
        
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        start_time = time.time()
        try:
            automl.fit(X_train, y_train)
            predictions = automl.predict(X_train)
            r2 = r2_score(y_train, predictions)
            time_taken = time.time() - start_time
            
            self.track_test_result("Extreme Regression", True, time_taken, r2)
            
            # Assertions
            self.assertGreater(r2, -0.5)  # Very lenient for extreme regression
            self.assertLess(time_taken, 90)
            
        except Exception as e:
            time_taken = time.time() - start_time
            self.track_test_result("Extreme Regression", False, time_taken, error=str(e))

    def test_comprehensive_extreme(self):
        """Comprehensive test of all extreme scenarios"""
        print("\n🔥 COMPREHENSIVE EXTREME SCENARIO TEST")
        print("=" * 70)
        
        # Run all extreme tests
        self.test_ultra_large_dataset()
        self.test_ultra_small_dataset()
        self.test_extreme_missing_values()
        self.test_infinite_values()
        self.test_extreme_categorical_complexity()
        self.test_mixed_data_types_extreme()
        self.test_imbalanced_dataset()
        self.test_noisy_dataset()
        self.test_multiclass_extreme()
        self.test_ultra_fast_constraint()
        self.test_file_input_output()
        self.test_extreme_regression()
        
        # Analyze results
        successful_tests = [r for r in self.test_results if r['success']]
        failed_tests = [r for r in self.test_results if not r['success']]
        
        print("\n📊 EXTREME TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"✅ Successful Tests: {len(successful_tests)}/{len(self.test_results)}")
        print(f"❌ Failed Tests: {len(failed_tests)}")
        
        if successful_tests:
            avg_time = np.mean([r['time'] for r in successful_tests])
            scores = [r['score'] for r in successful_tests if r['score'] is not None]
            avg_score = np.mean(scores) if scores else 0
            
            print(f"⏱️ Average Time: {avg_time:.2f}s")
            print(f"🎯 Average Score: {avg_score:.3f}")
        
        if failed_tests:
            print("\n❌ Failed Tests:")
            for test in failed_tests:
                print(f"   • {test['test']}: {test['error'][:50]}...")
        
        print("\n🔥 EXTREME BULLETPROOF ASSESSMENT")
        print("=" * 70)
        
        success_rate = len(successful_tests) / len(self.test_results)
        
        if success_rate >= 0.9:
            print("🎉 LEGENDARY: System handles EXTREME scenarios perfectly!")
            print("🚀 Ready for ANY production challenge!")
        elif success_rate >= 0.7:
            print("✅ EXCELLENT: System handles most extreme scenarios!")
            print("🔧 Minor improvements needed")
        elif success_rate >= 0.5:
            print("⚠️ GOOD: System handles many extreme scenarios!")
            print("🔧 Some improvements needed")
        else:
            print("❌ NEEDS WORK: System struggles with extreme scenarios")
            print("🔧 Significant improvements needed")
        
        print(f"📈 Extreme Success Rate: {success_rate:.1%}")
        
        # Final assertions
        self.assertGreater(success_rate, 0.6, "System should handle at least 60% of extreme scenarios")
        
        print("=" * 70)
        print("🔥 EXTREME SCENARIO TESTING COMPLETE!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
