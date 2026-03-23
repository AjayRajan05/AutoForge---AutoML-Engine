import click
import pandas as pd
import numpy as np
import logging
import os
import sys
import traceback
from datetime import datetime
import joblib
from api.automl import AutoML, validate_input, detect_task_type


# Setup CLI logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.command()
@click.argument("data_path")
@click.option("--target", required=True, help="Target column name")
@click.option("--trials", default=20, help="Number of optimization trials")
@click.option("--cv", default=3, help="Cross-validation folds")
@click.option("--timeout", default=None, help="Timeout in seconds")
@click.option("--output", default="model.pkl", help="Output model file")
@click.option("--verbose", is_flag=True, help="Verbose output")
def train(data_path, target, trials, cv, timeout, output, verbose):
    """
    Train AutoML model on the specified dataset
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info(f"Starting AutoML training on {data_path}")
        logger.info(f"Target column: {target}")
        logger.info(f"Trials: {trials}, CV folds: {cv}")
        
        # Validate input file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data with error handling
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded dataset: {df.shape}")
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
        
        # Validate target column exists
        if target not in df.columns:
            available_columns = list(df.columns)
            raise ValueError(f"Target column '{target}' not found. Available columns: {available_columns}")
        
        # Prepare features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Basic data validation
        if len(X) < 10:
            raise ValueError(f"Dataset too small: {len(X)} samples (minimum 10 required)")
        
        if X.shape[1] < 1:
            raise ValueError("Dataset has no feature columns")
        
        # Detect and log task type
        task_type = detect_task_type(y.values)
        logger.info(f"Detected task type: {task_type}")
        
        # Check for missing values
        missing_ratio = X.isnull().sum().sum() / X.size
        if missing_ratio > 0.8:
            logger.warning(f"High missing value ratio: {missing_ratio:.2%}")
        
        # Initialize AutoML
        automl = AutoML(n_trials=trials, cv=cv, timeout=timeout)
        
        logger.info("Starting AutoML training...")
        start_time = datetime.now()
        
        # Train model
        automl.fit(X, y)
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save model
        try:
            joblib.dump(automl, output)
            logger.info(f"Model saved to {output}")
            
            # Get model info
            model_info = {
                "task_type": automl.task_type,
                "dataset_shape": X.shape,
                "training_time": training_time,
                "target_column": target
            }
            
            # Save metadata
            metadata_path = output.replace('.pkl', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"Model metadata saved to {metadata_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
        
        click.echo(f"SUCCESS: Model trained successfully!")
        click.echo(f"   Task type: {automl.task_type}")
        click.echo(f"   Training time: {training_time:.2f} seconds")
        click.echo(f"   Model saved: {output}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        click.echo("Training interrupted")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if verbose:
            logger.error(traceback.format_exc())
        click.echo(f"ERROR: Training failed: {e}")
        sys.exit(1)


@click.command()
@click.argument("model_path")
@click.argument("data_path")
@click.option("--output", default="predictions.csv", help="Output predictions file")
@click.option("--probabilities", is_flag=True, help="Output class probabilities (classification only)")
@click.option("--verbose", is_flag=True, help="Verbose output")
def predict(model_path, data_path, output, probabilities, verbose):
    """
    Make predictions using trained AutoML model
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Validate model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        try:
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
        
        # Validate model has required attributes
        if not hasattr(model, 'predict'):
            raise ValueError("Invalid model: missing predict method")
        
        # Load data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded prediction data: {df.shape}")
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
        
        # If target column exists, drop it for prediction
        if 'target' in df.columns:
            df = df.drop(columns=['target'])
            logger.info(f"Dropped target column, new shape: {df.shape}")
        
        # Make predictions
        logger.info("Making predictions...")
        start_time = datetime.now()
        
        try:
            predictions = model.predict(df)
            prediction_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Predictions completed in {prediction_time:.2f} seconds")
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
        
        # Prepare output
        output_df = df.copy()
        output_df['predictions'] = predictions
        
        # Add probabilities if requested and available
        if probabilities and hasattr(model, 'predict_proba') and model.task_type == 'classification':
            try:
                proba = model.predict_proba(df)
                if len(proba[0]) == 2:  # Binary classification
                    output_df['probability_class_0'] = proba[:, 0]
                    output_df['probability_class_1'] = proba[:, 1]
                else:  # Multi-class
                    for i, prob in enumerate(proba.T):
                        output_df[f'probability_class_{i}'] = prob
                        
                logger.info("Probabilities calculated successfully")
                
            except Exception as e:
                logger.warning(f"Failed to calculate probabilities: {e}")
        
        # Save predictions
        try:
            output_df.to_csv(output, index=False)
            logger.info(f"Predictions saved to {output}")
        except Exception as e:
            raise RuntimeError(f"Failed to save predictions: {e}")
        
        click.echo(f"SUCCESS: Predictions completed!")
        click.echo(f"   Prediction time: {prediction_time:.2f} seconds")
        click.echo(f"   Predictions saved: {output}")
        click.echo(f"   Samples predicted: {len(predictions)}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if verbose:
            logger.error(traceback.format_exc())
        click.echo(f"ERROR: Prediction failed: {e}")
        sys.exit(1)


@click.command(name="logs")
@click.option("--limit", default=10, help="Number of recent logs to show")
@click.option("--model", help="Filter by model name")
@click.option("--score", type=float, help="Filter by minimum score")
@click.option("--verbose", is_flag=True, help="Verbose output")
def show_logs(limit, model, score, verbose):
    """
    Show experiment logs and statistics
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        log_path = "automl/experiments/logs.json"
        
        if not os.path.exists(log_path):
            click.echo("No experiment logs found")
            return
        
        # Load logs
        try:
            import json
            with open(log_path, "r") as f:
                logs = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load logs: {e}")
        
        if not logs:
            click.echo(" No experiment logs found")
            return
        
        # Filter logs
        filtered_logs = logs
        
        if model:
            filtered_logs = [log for log in filtered_logs if log.get("model") == model]
        
        if score is not None:
            filtered_logs = [log for log in filtered_logs 
                           if log.get("metrics", {}).get("cv_score", 0) >= score]
        
        # Show recent logs
        recent_logs = filtered_logs[-limit:]
        
        click.echo(f"Recent {len(recent_logs)} experiments:")
        click.echo("-" * 80)
        
        for i, log in enumerate(reversed(recent_logs), 1):
            model_name = log.get("model", "Unknown")
            metrics = log.get("metrics", {})
            cv_score = metrics.get("cv_score", "N/A")
            timestamp = log.get("timestamp", "Unknown")
            
            click.echo(f"{i}. {model_name}")
            click.echo(f"   Score: {cv_score}")
            click.echo(f"   Time: {timestamp}")
            
            if verbose:
                params = log.get("params", {})
                if params:
                    click.echo(f"   Params: {params}")
            
            click.echo()
        
        # Show statistics
        if logs:
            scores = [log.get("metrics", {}).get("cv_score") for log in logs 
                      if log.get("metrics", {}).get("cv_score") is not None]
            
            if scores:
                click.echo(f"Statistics:")
                click.echo(f"   Total experiments: {len(logs)}")
                click.echo(f"   Average score: {np.mean(scores):.4f}")
                click.echo(f"   Best score: {np.max(scores):.4f}")
                click.echo(f"   Score std: {np.std(scores):.4f}")
        
    except Exception as e:
        logger.error(f"Failed to show logs: {e}")
        if verbose:
            logger.error(traceback.format_exc())
        click.echo(f"Failed to show logs: {e}")
        sys.exit(1)


@click.command()
@click.option("--data-path", help="Dataset to validate")
@click.option("--target", help="Target column for validation")
@click.option("--verbose", is_flag=True, help="Verbose output")
def validate(data_path, target, verbose):
    """
    Validate dataset and show statistics
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if not data_path:
            click.echo("ERROR: Please provide --data-path")
            sys.exit(1)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        
        click.echo(f"Dataset Validation: {data_path}")
        click.echo(f"   Shape: {df.shape}")
        click.echo(f"   Columns: {list(df.columns)}")
        
        # Missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) > 0:
            click.echo(f"   Missing values:")
            for col, count in missing_cols.items():
                percentage = (count / len(df)) * 100
                click.echo(f"     {col}: {count} ({percentage:.1f}%)")
        else:
            click.echo(f"   Missing values: None")
        
        # Data types
        click.echo(f"   Data types:")
        for col, dtype in df.dtypes.items():
            click.echo(f"     {col}: {dtype}")
        
        # Target validation
        if target:
            if target not in df.columns:
                click.echo(f"ERROR: Target column '{target}' not found")
                sys.exit(1)
            
            y = df[target]
            task_type = detect_task_type(y.values)
            
            click.echo(f"   Target column: {target}")
            click.echo(f"   Task type: {task_type}")
            click.echo(f"   Unique values: {len(y.unique())}")
            
            if task_type == "classification":
                value_counts = y.value_counts()
                click.echo(f"   Class distribution:")
                for class_val, count in value_counts.items():
                    percentage = (count / len(y)) * 100
                    click.echo(f"     {class_val}: {count} ({percentage:.1f}%)")
            else:
                click.echo(f"   Target range: [{y.min():.3f}, {y.max():.3f}]")
                click.echo(f"   Target mean: {y.mean():.3f}")
                click.echo(f"   Target std: {y.std():.3f}")
        
        click.echo("Dataset validation completed")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        if verbose:
            logger.error(traceback.format_exc())
        click.echo(f"ERROR: Validation failed: {e}")
        sys.exit(1)

