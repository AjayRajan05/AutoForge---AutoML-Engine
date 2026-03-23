import pytest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from click.testing import CliRunner
from cli.main import cli
from cli.commands import train, predict, show_logs


class TestCLICommands:
    """Test CLI command functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.runner = CliRunner()
        
        # Create temporary CSV file
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.temp_dir, "test_data.csv")
        
        # Generate test data
        np.random.seed(42)
        data = {
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        }
        df = pd.DataFrame(data)
        df.to_csv(self.test_csv, index=False)
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_train_command_basic(self):
        """Test basic train command"""
        with patch('cli.commands.AutoML') as mock_automl:
            mock_instance = MagicMock()
            mock_automl.return_value = mock_instance
            
            result = self.runner.invoke(cli, [
                'train', self.test_csv, '--target', 'target', '--trials', '5'
            ])
            
            assert result.exit_code == 0
            mock_automl.assert_called_once_with(n_trials=5)
            mock_instance.fit.assert_called_once()
    
    def test_train_command_file_not_found(self):
        """Test train command with non-existent file"""
        result = self.runner.invoke(cli, [
            'train', 'nonexistent.csv', '--target', 'target'
        ])
        
        assert result.exit_code != 0
        assert 'not found' in result.output.lower() or 'no such file' in result.output.lower()
    
    def test_train_command_missing_target(self):
        """Test train command without target column"""
        result = self.runner.invoke(cli, [
            'train', self.test_csv
        ])
        
        assert result.exit_code != 0
        assert 'target' in result.output.lower()
    
    def test_train_command_invalid_target(self):
        """Test train command with non-existent target column"""
        result = self.runner.invoke(cli, [
            'train', self.test_csv, '--target', 'nonexistent_column'
        ])
        
        assert result.exit_code != 0
        assert 'column' in result.output.lower()
    
    def test_predict_command_basic(self):
        """Test basic predict command"""
        # Create a mock model file
        model_path = os.path.join(self.temp_dir, "model.pkl")
        
        with patch('joblib.load') as mock_load, \
             patch('pandas.read_csv') as mock_read_csv:
            
            # Mock model and data
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0, 1, 0, 1])
            mock_load.return_value = mock_model
            
            mock_df = pd.DataFrame({'feature1': [1, 2, 3, 4]})
            mock_read_csv.return_value = mock_df
            
            result = self.runner.invoke(cli, [
                'predict', model_path, self.test_csv
            ])
            
            assert result.exit_code == 0
            mock_load.assert_called_once_with(model_path)
            mock_model.predict.assert_called_once()
    
    def test_predict_command_model_not_found(self):
        """Test predict command with non-existent model"""
        result = self.runner.invoke(cli, [
            'predict', 'nonexistent_model.pkl', self.test_csv
        ])
        
        assert result.exit_code != 0
        assert 'not found' in result.output.lower() or 'no such file' in result.output.lower()
    
    def test_predict_command_data_not_found(self):
        """Test predict command with non-existent data file"""
        model_path = os.path.join(self.temp_dir, "model.pkl")
        
        with open(model_path, 'w') as f:
            f.write("mock model")
        
        result = self.runner.invoke(cli, [
            'predict', model_path, 'nonexistent.csv'
        ])
        
        assert result.exit_code != 0
        assert 'not found' in result.output.lower() or 'no such file' in result.output.lower()
    
    def test_logs_command_with_file(self):
        """Test logs command with existing log file"""
        log_path = os.path.join(self.temp_dir, "logs.json")
        
        # Create mock log file
        logs = [
            {"run_id": "1", "model": "random_forest", "score": 0.85},
            {"run_id": "2", "model": "logistic_regression", "score": 0.78},
        ]
        with open(log_path, 'w') as f:
            json.dump(logs, f)
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.__enter__.return_value.read.return_value = json.dumps(logs)
            mock_open.return_value.__enter__.return_value.__exit__.return_value = None
            
            result = self.runner.invoke(cli, ['logs'])
            
            # Should not crash
            assert result.exit_code == 0
    
    def test_logs_command_no_file(self):
        """Test logs command with no log file"""
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = self.runner.invoke(cli, ['logs'])
            
            assert result.exit_code != 0
            assert 'not found' in result.output.lower()
    
    def test_help_command(self):
        """Test help command"""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'train' in result.output
        assert 'predict' in result.output
        assert 'logs' in result.output
    
    def test_train_help(self):
        """Test train command help"""
        result = self.runner.invoke(cli, ['train', '--help'])
        
        assert result.exit_code == 0
        assert 'target' in result.output
        assert 'trials' in result.output
    
    def test_predict_help(self):
        """Test predict command help"""
        result = self.runner.invoke(cli, ['predict', '--help'])
        
        assert result.exit_code == 0
        assert 'model' in result.output
        assert 'data' in result.output


class TestCLIIntegration:
    """Test CLI integration scenarios"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create more realistic test data
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'numeric_feature1': np.random.normal(0, 1, n_samples),
            'numeric_feature2': np.random.exponential(1, n_samples),
            'categorical_feature': np.random.choice(['TypeA', 'TypeB', 'TypeC'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        }
        
        # Add some missing values
        data['numeric_feature1'][:10] = np.nan
        
        self.df = pd.DataFrame(data)
        self.train_csv = os.path.join(self.temp_dir, "train.csv")
        self.test_csv = os.path.join(self.temp_dir, "test.csv")
        
        # Split into train/test
        train_df = self.df.iloc[:150]
        test_df = self.df.iloc[150:].drop('target', axis=1)
        
        train_df.to_csv(self.train_csv, index=False)
        test_df.to_csv(self.test_csv, index=False)
    
    def teardown_method(self):
        """Cleanup integration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_workflow_simulation(self):
        """Test simulated full workflow (without actual training)"""
        with patch('cli.commands.AutoML') as mock_automl_class, \
             patch('joblib.dump') as mock_dump, \
             patch('joblib.load') as mock_load:
            
            # Mock AutoML instance
            mock_automl = MagicMock()
            mock_automl_class.return_value = mock_automl
            
            # Training phase
            train_result = self.runner.invoke(cli, [
                'train', self.train_csv, '--target', 'target', '--trials', '10'
            ])
            
            assert train_result.exit_code == 0
            mock_automl.fit.assert_called_once()
            mock_dump.assert_called_once()
            
            # Prediction phase
            mock_automl.predict.return_value = np.array([0, 1, 0, 1, 1])
            mock_load.return_value = mock_automl
            
            predict_result = self.runner.invoke(cli, [
                'predict', 'model.pkl', self.test_csv
            ])
            
            assert predict_result.exit_code == 0
            mock_automl.predict.assert_called_once()
    
    def test_error_recovery(self):
        """Test CLI error recovery scenarios"""
        # Test with invalid CSV format
        invalid_csv = os.path.join(self.temp_dir, "invalid.csv")
        with open(invalid_csv, 'w') as f:
            f.write("invalid,csv,format\nno,proper,columns")
        
        result = self.runner.invoke(cli, [
            'train', invalid_csv, '--target', 'target'
        ])
        
        assert result.exit_code != 0
        assert 'error' in result.output.lower() or 'invalid' in result.output.lower()
    
    def test_large_dataset_handling(self):
        """Test CLI with larger dataset"""
        # Create larger dataset
        large_data = {
            f'feature_{i}': np.random.rand(1000) 
            for i in range(20)
        }
        large_data['target'] = np.random.randint(0, 2, 1000)
        
        large_df = pd.DataFrame(large_data)
        large_csv = os.path.join(self.temp_dir, "large.csv")
        large_df.to_csv(large_csv, index=False)
        
        with patch('cli.commands.AutoML') as mock_automl:
            mock_instance = MagicMock()
            mock_automl.return_value = mock_instance
            
            result = self.runner.invoke(cli, [
                'train', large_csv, '--target', 'target', '--trials', '3'
            ])
            
            assert result.exit_code == 0
            mock_instance.fit.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
