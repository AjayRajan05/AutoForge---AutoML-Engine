from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, mean_absolute_error,
    r2_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
import numpy as np


def evaluate(y_true, y_pred, task="classification", y_pred_proba=None):
    """
    Comprehensive evaluation metrics for classification and regression tasks
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        task: "classification" or "regression"
        y_pred_proba: Predicted probabilities (for classification AUC)
        
    Returns:
        Dictionary of comprehensive metrics
    """
    if task == "classification":
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_micro": f1_score(y_true, y_pred, average="micro"),
        }
        
        # Add precision and recall
        try:
            metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted")
            metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro")
            metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted")
            metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro")
        except Exception:
            pass  # Handle edge cases
        
        # Add AUC if probabilities are provided
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Multi-class
                    metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="weighted")
            except Exception:
                pass
        
        return metrics
    
    elif task == "regression":
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        try:
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        except Exception:
            mape = np.inf
        
        # Calculate relative metrics
        y_mean = np.mean(y_true)
        y_std = np.std(y_true)
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2_score(y_true, y_pred),
            "mean_y_true": y_mean,
            "std_y_true": y_std,
            "rmse_relative": rmse / y_std if y_std > 0 else np.inf,  # RMSE relative to target std
            "mae_relative": mae / y_std if y_std > 0 else np.inf,   # MAE relative to target std
        }
        
        return metrics
    
    else:
        raise ValueError("Task must be 'classification' or 'regression'")


def print_evaluation_report(metrics, task="classification"):
    """
    Print a formatted evaluation report
    """
    print(f"\n📊 {task.title()} Performance Report")
    print("=" * 50)
    
    if task == "classification":
        print(f"🎯 Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"🎯 F1-Score (Weighted): {metrics.get('f1_weighted', 0):.4f}")
        print(f"🎯 F1-Score (Macro): {metrics.get('f1_macro', 0):.4f}")
        print(f"🎯 Precision (Weighted): {metrics.get('precision_weighted', 0):.4f}")
        print(f"🎯 Recall (Weighted): {metrics.get('recall_weighted', 0):.4f}")
        if 'auc_roc' in metrics:
            print(f"🎯 AUC-ROC: {metrics['auc_roc']:.4f}")
    
    elif task == "regression":
        print(f"📈 MSE: {metrics.get('mse', 0):.4f}")
        print(f"📈 RMSE: {metrics.get('rmse', 0):.4f}")
        print(f"📈 MAE: {metrics.get('mae', 0):.4f}")
        print(f"📈 MAPE: {metrics.get('mape', 0):.2f}%")
        print(f"📈 R²: {metrics.get('r2', 0):.4f}")
        print(f"📈 RMSE/Std: {metrics.get('rmse_relative', 0):.4f}")
        print(f"📈 MAE/Std: {metrics.get('mae_relative', 0):.4f}")
        
        # Performance interpretation
        r2 = metrics.get('r2', 0)
        if r2 >= 0.9:
            print("🏆 Excellent performance (R² ≥ 0.9)")
        elif r2 >= 0.7:
            print("🥈 Good performance (R² ≥ 0.7)")
        elif r2 >= 0.5:
            print("🥉 Fair performance (R² ≥ 0.5)")
        else:
            print("⚠️  Poor performance (R² < 0.5)")
    
    print("=" * 50)