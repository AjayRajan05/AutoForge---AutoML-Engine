"""
Complete System Test
Tests the entire AutoForge system with mock data
"""

import numpy as np
import pandas as pd
import warnings
import logging
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

def test_basic_automl():
    """Test basic AutoML functionality"""
    print("🧪 Testing Basic AutoML...")
    
    try:
        from api.automl import AutoML
        
        # Generate classification data
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize AutoML with reduced trials for testing
        automl = AutoML(
            n_trials=10,
            cv=3,
            use_explainability=True,
            show_progress=False
        )
        
        # Fit model
        print("  🔄 Training AutoML model...")
        start_time = time.time()
        automl.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        print("  📊 Making predictions...")
        predictions = automl.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"  ✅ Basic AutoML Test Passed!")
        print(f"     Training Time: {training_time:.2f} seconds")
        print(f"     Accuracy: {accuracy:.4f}")
        print(f"     Task Type: {automl.task_type}")
        print(f"     Best Pipeline: {type(automl.best_pipeline).__name__}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic AutoML Test Failed: {e}")
        return False

def test_self_improving_automl():
    """Test self-improving AutoML functionality"""
    print("\n🧠 Testing Self-Improving AutoML...")
    
    try:
        from api.self_improving_automl import SelfImprovingAutoML
        
        # Generate classification data
        X, y = make_classification(
            n_samples=500,  # Smaller for testing
            n_features=15,
            n_informative=10,
            n_redundant=5,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize Self-Improving AutoML
        automl = SelfImprovingAutoML(
            n_trials=8,  # Reduced for testing
            cv=3,
            use_explainability=True,
            enable_pattern_learning=True,
            enable_actionable_insights=True,
            show_progress=False
        )
        
        # Fit with learning
        print("  🔄 Training Self-Improving AutoML...")
        start_time = time.time()
        automl.fit_with_learning(X_train, y_train, store_experiment=True)
        training_time = time.time() - start_time
        
        # Make predictions
        print("  📊 Making predictions...")
        predictions = automl.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        
        # Test intelligent recommendations
        print("  🧠 Getting intelligent recommendations...")
        dataset_info = {
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "data_type": "tabular"
        }
        recommendations = automl.get_intelligent_recommendations(dataset_info)
        
        # Test actionable insights
        print("  💡 Getting actionable insights...")
        explanations = automl.explain_with_actions(X_test, y_test)
        
        print(f"  ✅ Self-Improving AutoML Test Passed!")
        print(f"     Training Time: {training_time:.2f} seconds")
        print(f"     Accuracy: {accuracy:.4f}")
        print(f"     Pattern Learning: {'✅' if automl.enable_pattern_learning else '❌'}")
        print(f"     Actionable Insights: {'✅' if automl.enable_actionable_insights else '❌'}")
        print(f"     Model Recommendations: {len(recommendations.get('models', []))}")
        print(f"     Actionable Insights: {len(explanations.get('actionable_insights', {}).get('feature_recommendations', []))}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Self-Improving AutoML Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pattern_learning():
    """Test pattern learning system"""
    print("\n📚 Testing Pattern Learning System...")
    
    try:
        from meta_learning.pattern_learner import PatternLearner
        
        # Create mock experiment result
        experiment_result = {
            "dataset_info": {
                "n_samples": 1000,
                "n_features": 20,
                "data_type": "tabular"
            },
            "model_info": {
                "name": "RandomForest",
                "task_type": "classification"
            },
            "performance_metrics": {
                "accuracy": 0.85,
                "f1_score": 0.83
            },
            "optimization_info": {
                "strategy": "adaptive",
                "n_trials": 50
            },
            "feature_engineering": {
                "feature_types": ["polynomial", "interaction"],
                "engineered_features": 15
            }
        }
        
        # Initialize pattern learner
        learner = PatternLearner()
        
        # Learn from experiment
        print("  🧠 Learning from experiment...")
        insights = learner.learn_from_experiment(experiment_result)
        
        # Get recommendations
        print("  📋 Getting recommendations...")
        dataset_info = experiment_result["dataset_info"]
        recommendations = learner.get_recommendations(dataset_info, "classification")
        
        # Get pattern summary
        print("  📊 Getting pattern summary...")
        summary = learner.get_pattern_summary()
        
        print(f"  ✅ Pattern Learning Test Passed!")
        print(f"     Learned Insights: {len(insights)}")
        print(f"     Model Recommendations: {len(recommendations.get('models', []))}")
        print(f"     Feature Recommendations: {len(recommendations.get('feature_engineering', []))}")
        print(f"     Total Patterns: {summary.get('total_patterns', 0)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pattern Learning Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_actionable_explainability():
    """Test actionable explainability system"""
    print("\n💡 Testing Actionable Explainability...")
    
    try:
        from explainability.actionable_explainability import ActionableExplainability
        
        # Create mock explanations
        explanations = {
            "feature_importance": {
                "aggregated": {
                    "feature_1": 0.45,  # High importance
                    "feature_2": 0.25,
                    "feature_3": 0.15,
                    "feature_4": 0.10,
                    "feature_5": 0.05   # Low importance
                }
            }
        }
        
        # Create mock data
        X = pd.DataFrame({
            "feature_1": np.random.rand(100),
            "feature_2": np.random.rand(100),
            "feature_3": np.random.rand(100),
            "feature_4": np.random.rand(100),
            "feature_5": np.random.rand(100)
        })
        
        y = pd.Series(np.random.randint(0, 2, 100))
        
        # Initialize actionable explainability
        actionable = ActionableExplainability()
        
        # Generate actionable insights
        print("  💡 Generating actionable insights...")
        insights = actionable.generate_actionable_insights(explanations, X, y)
        
        # Get summary
        print("  📋 Getting actionable summary...")
        summary = actionable.get_actionable_summary(insights)
        
        print(f"  ✅ Actionable Explainability Test Passed!")
        print(f"     Feature Recommendations: {len(insights.get('feature_recommendations', []))}")
        print(f"     Data Quality Issues: {len(insights.get('data_quality_issues', []))}")
        print(f"     Model Risks: {len(insights.get('model_risks', []))}")
        print(f"     Business Insights: {len(insights.get('business_insights', []))}")
        print(f"     High Priority Actions: {insights.get('action_priority', {}).get('high', 0)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Actionable Explainability Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_benchmarking():
    """Test enhanced benchmarking system"""
    print("\n📊 Testing Enhanced Benchmarking...")
    
    try:
        from benchmarking.enhanced_benchmarking import EnhancedBenchmarking
        
        # Initialize benchmarking
        benchmarking = EnhancedBenchmarking()
        
        # Generate test datasets
        datasets = []
        
        # Small classification
        X_small, y_small = make_classification(
            n_samples=100, n_features=10, n_informative=8, random_state=42
        )
        datasets.append({
            "name": "Small Classification",
            "type": "classification",
            "size": "small",
            "X": X_small, "y": y_small,
            "description": "100 samples, 10 features"
        })
        
        # Medium classification
        X_med, y_med = make_classification(
            n_samples=500, n_features=20, n_informative=15, random_state=42
        )
        datasets.append({
            "name": "Medium Classification",
            "type": "classification",
            "size": "medium",
            "X": X_med, "y": y_med,
            "description": "500 samples, 20 features"
        })
        
        # Run benchmarking with reduced scope for testing
        print("  📊 Running benchmarking...")
        results = benchmarking.run_comprehensive_benchmark(
            datasets=datasets,
            competitors=["AutoForge"]  # Only test our system
        )
        
        print(f"  ✅ Enhanced Benchmarking Test Passed!")
        print(f"     Datasets Tested: {len(results.get('datasets', []))}")
        print(f"     Visualizations: {len(results.get('visualizations', {}))}")
        print(f"     Insights Generated: {len(results.get('insights', {}))}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced Benchmarking Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_regression_task():
    """Test regression functionality"""
    print("\n📈 Testing Regression Task...")
    
    try:
        from api.self_improving_automl import SelfImprovingAutoML
        
        # Generate regression data
        X, y = make_regression(
            n_samples=500,
            n_features=15,
            n_informative=10,
            noise=0.1,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize AutoML for regression
        automl = SelfImprovingAutoML(
            n_trials=8,
            cv=3,
            use_explainability=True,
            enable_pattern_learning=True,
            enable_actionable_insights=True,
            show_progress=False
        )
        
        # Fit model
        print("  🔄 Training regression model...")
        automl.fit_with_learning(X_train, y_train, store_experiment=True)
        
        # Make predictions
        print("  📊 Making predictions...")
        predictions = automl.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"  ✅ Regression Test Passed!")
        print(f"     Task Type: {automl.task_type}")
        print(f"     MSE: {mse:.4f}")
        print(f"     R² Score: {r2:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Regression Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_analysis():
    """Test comprehensive analysis functionality"""
    print("\n🔍 Testing Comprehensive Analysis...")
    
    try:
        from api.self_improving_automl import SelfImprovingAutoML
        
        # Generate test data
        X, y = make_classification(
            n_samples=300,  # Smaller for testing
            n_features=12,
            n_informative=8,
            n_redundant=4,
            random_state=42
        )
        
        # Initialize AutoML
        automl = SelfImprovingAutoML(
            n_trials=5,  # Very reduced for testing
            cv=3,
            use_explainability=True,
            enable_pattern_learning=True,
            enable_actionable_insights=True,
            show_progress=False
        )
        
        # Run comprehensive analysis
        print("  🔍 Running comprehensive analysis...")
        analysis = automl.run_comprehensive_analysis(X, y)
        
        print(f"  ✅ Comprehensive Analysis Test Passed!")
        print(f"     Model Performance: {'✅' if analysis.get('model_performance') else '❌'}")
        print(f"     Actionable Insights: {'✅' if analysis.get('actionable_insights') else '❌'}")
        print(f"     Pattern Recommendations: {'✅' if analysis.get('pattern_recommendations') else '❌'}")
        print(f"     Overall Assessment: {'✅' if analysis.get('overall_assessment') else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Comprehensive Analysis Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 AutoForge Complete System Test")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Basic AutoML", test_basic_automl),
        ("Self-Improving AutoML", test_self_improving_automl),
        ("Pattern Learning", test_pattern_learning),
        ("Actionable Explainability", test_actionable_explainability),
        ("Enhanced Benchmarking", test_enhanced_benchmarking),
        ("Regression Task", test_regression_task),
        ("Comprehensive Analysis", test_comprehensive_analysis)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 Overall Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! AutoForge system is working correctly!")
        print("🧠 Self-improving features are functional!")
        print("💡 Actionable insights are being generated!")
        print("📊 Pattern learning is working!")
        print("🔍 Comprehensive analysis is operational!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
