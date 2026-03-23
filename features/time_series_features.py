"""
Time Series Feature Engineering
Lag features, rolling statistics, and temporal patterns
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


class TimeSeriesFeatureEngineer:
    """
    Advanced time series feature engineering
    """
    
    def __init__(self,
                 max_lags: int = 12,
                 rolling_windows: List[int] = [3, 7, 14, 30],
                 seasonal_periods: List[int] = [7, 30, 365],
                 min_samples_for_features: int = 50):
        """
        Initialize time series feature engineer
        
        Args:
            max_lags: Maximum number of lag features to create
            rolling_windows: Window sizes for rolling statistics
            seasonal_periods: Periods for seasonal features
            min_samples_for_features: Minimum samples required to create features
        """
        self.max_lags = max_lags
        self.rolling_windows = rolling_windows
        self.seasonal_periods = seasonal_periods
        self.min_samples_for_features = min_samples_for_features
        
        self.feature_metadata = {}
        self.created_features = []
        
    def engineer_time_series_features(self, 
                                    X: Union[np.ndarray, pd.DataFrame],
                                    y: Optional[Union[np.ndarray, pd.Series]] = None,
                                    datetime_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Engineer time series features
        
        Args:
            X: Input features
            y: Target variable (for supervised lag features)
            datetime_col: Name of datetime column
            Returns:
            Engineered features and metadata
        """
        logger.info("Engineering time series features...")
        
        # Convert to DataFrame
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            df = X.copy()
        
        # Add target as column if provided
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                df['target'] = y.values
            else:
                df['target'] = y
        
        # Process datetime column
        if datetime_col and datetime_col in df.columns:
            df = self._process_datetime_column(df, datetime_col)
        
        # Create lag features
        if len(df) >= self.min_samples_for_features:
            df = self._create_lag_features(df)
            df = self._create_rolling_features(df)
            df = self._create_seasonal_features(df)
            df = self._create_temporal_features(df)
        
        # Compile metadata
        metadata = {
            "original_features": X.shape[1],
            "engineered_features": df.shape[1] - X.shape[1],
            "total_features": df.shape[1],
            "lag_features_created": len([f for f in self.created_features if 'lag' in f]),
            "rolling_features_created": len([f for f in self.created_features if 'rolling' in f]),
            "seasonal_features_created": len([f for f in self.created_features if 'seasonal' in f]),
            "temporal_features_created": len([f for f in self.created_features if 'temporal' in f]),
            "created_features": self.created_features
        }
        
        logger.info(f"Time series engineering completed: {metadata['total_features']} features "
                   f"({metadata['engineered_features']} engineered)")
        
        # Remove target column if it was added
        if y is not None and 'target' in df.columns:
            df = df.drop(columns=['target'])
        
        return df, metadata
    
    def _process_datetime_column(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """
        Process datetime column to extract temporal features
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame with temporal features
        """
        try:
            # Convert to datetime
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            
            # Extract basic temporal features
            df[f'{datetime_col}_year'] = df[datetime_col].dt.year
            df[f'{datetime_col}_month'] = df[datetime_col].dt.month
            df[f'{datetime_col}_day'] = df[datetime_col].dt.day
            df[f'{datetime_col}_hour'] = df[datetime_col].dt.hour
            df[f'{datetime_col}_minute'] = df[datetime_col].dt.minute
            df[f'{datetime_col}_dayofweek'] = df[datetime_col].dt.dayofweek
            df[f'{datetime_col}_dayofyear'] = df[datetime_col].dt.dayofyear
            df[f'{datetime_col}_quarter'] = df[datetime_col].dt.quarter
            df[f'{datetime_col}_is_weekend'] = (df[datetime_col].dt.dayofweek >= 5).astype(int)
            
            # Cyclical encoding for temporal features
            df[f'{datetime_col}_month_sin'] = np.sin(2 * np.pi * df[datetime_col].dt.month / 12)
            df[f'{datetime_col}_month_cos'] = np.cos(2 * np.pi * df[datetime_col].dt.month / 12)
            df[f'{datetime_col}_day_sin'] = np.sin(2 * np.pi * df[datetime_col].dt.day / 31)
            df[f'{datetime_col}_day_cos'] = np.cos(2 * np.pi * df[datetime_col].dt.day / 31)
            df[f'{datetime_col}_hour_sin'] = np.sin(2 * np.pi * df[datetime_col].dt.hour / 24)
            df[f'{datetime_col}_hour_cos'] = np.cos(2 * np.pi * df[datetime_col].dt.hour / 24)
            
            # Add to created features
            temporal_features = [col for col in df.columns if datetime_col in col and col != datetime_col]
            self.created_features.extend(temporal_features)
            
            logger.info(f"Created {len(temporal_features)} temporal features from {datetime_col}")
            
        except Exception as e:
            logger.warning(f"Failed to process datetime column {datetime_col}: {e}")
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for time series
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with lag features
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'target':
                continue  # Skip target for now
                
            # Create lag features
            for lag in range(1, min(self.max_lags + 1, len(df) // 2)):
                lag_col = f"{col}_lag_{lag}"
                df[lag_col] = df[col].shift(lag)
                self.created_features.append(lag_col)
        
        logger.info(f"Created lag features for {len(numeric_cols)} columns")
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with rolling features
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'target':
                continue
                
            for window in self.rolling_windows:
                if window >= len(df):
                    continue
                
                # Rolling statistics
                df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window).mean()
                df[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window).std()
                df[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window).min()
                df[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window).max()
                df[f"{col}_rolling_median_{window}"] = df[col].rolling(window=window).median()
                
                # Rolling differences
                df[f"{col}_rolling_diff_{window}"] = df[col] - df[col].rolling(window=window).mean()
                
                # Add to created features
                rolling_features = [f"{col}_rolling_mean_{window}", f"{col}_rolling_std_{window}",
                                  f"{col}_rolling_min_{window}", f"{col}_rolling_max_{window}",
                                  f"{col}_rolling_median_{window}", f"{col}_rolling_diff_{window}"]
                self.created_features.extend(rolling_features)
        
        logger.info(f"Created rolling features for windows: {self.rolling_windows}")
        
        return df
    
    def _create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create seasonal features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with seasonal features
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'target':
                continue
                
            for period in self.seasonal_periods:
                if period >= len(df):
                    continue
                
                # Seasonal lag features
                df[f"{col}_seasonal_lag_{period}"] = df[col].shift(period)
                
                # Seasonal rolling statistics
                df[f"{col}_seasonal_mean_{period}"] = df[col].rolling(period).mean()
                df[f"{col}_seasonal_std_{period}"] = df[col].rolling(period).std()
                
                # Seasonal differences
                df[f"{col}_seasonal_diff_{period}"] = df[col] - df[col].shift(period)
                
                # Add to created features
                seasonal_features = [f"{col}_seasonal_lag_{period}", f"{col}_seasonal_mean_{period}",
                                   f"{col}_seasonal_std_{period}", f"{col}_seasonal_diff_{period}"]
                self.created_features.extend(seasonal_features)
        
        logger.info(f"Created seasonal features for periods: {self.seasonal_periods}")
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced temporal features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temporal features
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Trend features
        for col in numeric_cols:
            if col == 'target':
                continue
                
            # Simple trend (difference from previous value)
            df[f"{col}_trend"] = df[col].diff()
            
            # Acceleration (difference of differences)
            df[f"{col}_acceleration"] = df[col].diff().diff()
            
            # Rate of change (percentage change)
            df[f"{col}_rate_of_change"] = df[col].pct_change()
            
            # Add to created features
            trend_features = [f"{col}_trend", f"{col}_acceleration", f"{col}_rate_of_change"]
            self.created_features.extend(trend_features)
        
        # Interaction features between temporal components
        datetime_cols = [col for col in df.columns if any(x in col for x in ['year', 'month', 'day', 'hour'])]
        
        if len(datetime_cols) > 1 and len(numeric_cols) > 0:
            for i, dt_col1 in enumerate(datetime_cols[:3]):  # Limit interactions
                for j, dt_col2 in enumerate(datetime_cols[i+1:i+3], i+1):
                    for col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                        interaction_col = f"{col}_{dt_col1}_{dt_col2}_interaction"
                        df[interaction_col] = df[col] * df[dt_col1] * df[dt_col2]
                        self.created_features.append(interaction_col)
        
        logger.info("Created advanced temporal features")
        
        return df
    
    def get_feature_importance_ranking(self, df: pd.DataFrame, target_col: str = 'target') -> Dict[str, float]:
        """
        Get feature importance ranking for time series features
        
        Args:
            df: DataFrame with engineered features
            target_col: Target column name
            
        Returns:
            Feature importance scores
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
            
            if len(feature_cols) == 0:
                return {}
            
            X = df[feature_cols].fillna(0)
            y = df[target_col].fillna(df[target_col].mean())
            
            # Remove rows with NaN values
            valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) < 10:
                return {}
            
            # Train Random Forest for feature importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_clean, y_clean)
            
            # Get feature importances
            importances = dict(zip(feature_cols, rf.feature_importances_))
            
            # Sort by importance
            sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_importances
            
        except Exception as e:
            logger.warning(f"Failed to calculate feature importance: {e}")
            return {}


def engineer_time_series_features(X: Union[np.ndarray, pd.DataFrame],
                                 y: Optional[Union[np.ndarray, pd.Series]] = None,
                                 datetime_col: Optional[str] = None,
                                 **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function for time series feature engineering
    
    Args:
        X: Input features
        y: Target variable
        datetime_col: Name of datetime column
        **kwargs: Additional arguments for TimeSeriesFeatureEngineer
        
    Returns:
        Engineered features and metadata
    """
    engineer = TimeSeriesFeatureEngineer(**kwargs)
    return engineer.engineer_time_series_features(X, y, datetime_col)
