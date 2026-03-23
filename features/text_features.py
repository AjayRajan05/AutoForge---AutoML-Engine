"""
Text Feature Engineering
TF-IDF, embeddings, and natural language processing
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import re
from collections import Counter
import warnings

logger = logging.getLogger(__name__)


class TextFeatureEngineer:
    """
    Advanced text feature engineering
    """
    
    def __init__(self,
                 max_features: int = 10000,
                 min_df: int = 2,
                 max_df: float = 0.95,
                 ngram_range: Tuple[int, int] = (1, 2),
                 use_tfidf: bool = True,
                 reduce_dimensions: bool = True,
                 n_components: int = 100):
        """
        Initialize text feature engineer
        
        Args:
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: Range of n-grams to extract
            use_tfidf: Whether to use TF-IDF (vs. count vectorizer)
            reduce_dimensions: Whether to reduce dimensions with SVD
            n_components: Number of components for dimensionality reduction
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_tfidf = use_tfidf
        self.reduce_dimensions = reduce_dimensions
        self.n_components = n_components
        
        self.vectorizers = {}
        self.svd_reducers = {}
        self.feature_metadata = {}
        self.created_features = []
        
    def engineer_text_features(self, 
                             X: Union[np.ndarray, pd.DataFrame],
                             text_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Engineer text features
        
        Args:
            X: Input features
            text_columns: List of text column names (if None, auto-detect)
            
        Returns:
            Engineered features and metadata
        """
        logger.info("Engineering text features...")
        
        # Convert to DataFrame
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            df = X.copy()
        
        # Auto-detect text columns if not provided
        if text_columns is None:
            text_columns = self._detect_text_columns(df)
        
        if not text_columns:
            logger.warning("No text columns detected")
            return df, {"text_columns_found": 0, "engineered_features": 0}
        
        logger.info(f"Processing {len(text_columns)} text columns: {text_columns}")
        
        # Process each text column
        engineered_dfs = []
        
        for col in text_columns:
            if col not in df.columns:
                continue
                
            # Preprocess text
            processed_text = self._preprocess_text(df[col])
            
            # Create vectorized features
            vectorized_df = self._create_vectorized_features(processed_text, col)
            engineered_dfs.append(vectorized_df)
            
            # Create linguistic features
            linguistic_df = self._create_linguistic_features(processed_text, col)
            engineered_dfs.append(linguistic_df)
        
        # Combine engineered features
        if engineered_dfs:
            engineered_features = pd.concat(engineered_dfs, axis=1)
            
            # Combine with original features (excluding original text columns)
            non_text_cols = [col for col in df.columns if col not in text_columns]
            if non_text_cols:
                original_features = df[non_text_cols]
                result_df = pd.concat([original_features, engineered_features], axis=1)
            else:
                result_df = engineered_features
        else:
            result_df = df
        
        # Compile metadata
        metadata = {
            "text_columns_found": len(text_columns),
            "text_columns": text_columns,
            "original_features": X.shape[1],
            "engineered_features": result_df.shape[1] - X.shape[1],
            "total_features": result_df.shape[1],
            "vectorizer_type": "tfidf" if self.use_tfidf else "count",
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
            "dimensionality_reduction": self.reduce_dimensions,
            "created_features": self.created_features
        }
        
        logger.info(f"Text engineering completed: {metadata['total_features']} features "
                   f"({metadata['engineered_features']} engineered)")
        
        return result_df, metadata
    
    def _detect_text_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Auto-detect text columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of text column names
        """
        text_columns = []
        
        for col in df.columns:
            # Skip if too few samples
            if len(df[col].dropna()) < 10:
                continue
            
            # Check if it's object type (likely text)
            if df[col].dtype == 'object':
                # Sample some values
                sample_values = df[col].dropna().head(100).astype(str)
                
                # Check for text characteristics
                avg_length = sample_values.str.len().mean()
                unique_ratio = sample_values.nunique() / len(sample_values)
                
                # Check if it contains words
                has_words = sample_values.str.contains(r'\b\w+\b').any()
                
                # Determine if it's text data
                is_text = (
                    avg_length > 10 and  # Reasonable length
                    unique_ratio > 0.1 and  # Some variety
                    has_words  # Contains words
                )
                
                if is_text:
                    text_columns.append(col)
        
        return text_columns
    
    def _preprocess_text(self, text_series: pd.Series) -> pd.Series:
        """
        Preprocess text data
        
        Args:
            text_series: Text data
            
        Returns:
            Preprocessed text series
        """
        try:
            # Convert to string and handle missing values
            text = text_series.fillna('').astype(str)
            
            # Basic preprocessing
            text = text.str.lower()  # Lowercase
            text = text.str.replace(r'\d+', 'num', regex=True)  # Replace numbers
            text = text.str.replace(r'[^\w\s]', ' ', regex=True)  # Remove punctuation
            text = text.str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
            text = text.str.strip()  # Remove leading/trailing whitespace
            
            # Remove very short documents
            text = text[text.str.len() > 3]
            
            return text
            
        except Exception as e:
            logger.warning(f"Text preprocessing failed: {e}")
            return text_series.fillna('').astype(str)
    
    def _create_vectorized_features(self, text_series: pd.Series, column_name: str) -> pd.DataFrame:
        """
        Create vectorized text features
        
        Args:
            text_series: Preprocessed text data
            column_name: Original column name
            
        Returns:
            DataFrame with vectorized features
        """
        try:
            # Initialize vectorizer
            if self.use_tfidf:
                vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    min_df=self.min_df,
                    max_df=self.max_df,
                    ngram_range=self.ngram_range,
                    stop_words='english',
                    lowercase=False  # Already preprocessed
                )
            else:
                vectorizer = CountVectorizer(
                    max_features=self.max_features,
                    min_df=self.min_df,
                    max_df=self.max_df,
                    ngram_range=self.ngram_range,
                    stop_words='english',
                    lowercase=False
                )
            
            # Fit and transform
            text_matrix = vectorizer.fit_transform(text_series)
            
            # Convert to DataFrame
            feature_names = [f"{column_name}_{feat}" for feat in vectorizer.get_feature_names_out()]
            vectorized_df = pd.DataFrame(
                text_matrix.toarray(),
                columns=feature_names,
                index=text_series.index
            )
            
            # Dimensionality reduction if enabled
            if self.reduce_dimensions and text_matrix.shape[1] > self.n_components:
                svd = TruncatedSVD(n_components=self.n_components, random_state=42)
                reduced_matrix = svd.fit_transform(text_matrix)
                
                reduced_feature_names = [f"{column_name}_svd_{i}" for i in range(self.n_components)]
                vectorized_df = pd.DataFrame(
                    reduced_matrix,
                    columns=reduced_feature_names,
                    index=text_series.index
                )
                
                # Store reducer for later use
                self.svd_reducers[column_name] = svd
            
            # Store vectorizer for later use
            self.vectorizers[column_name] = vectorizer
            
            # Add to created features
            self.created_features.extend(vectorized_df.columns.tolist())
            
            logger.info(f"Created {vectorized_df.shape[1]} vectorized features for {column_name}")
            
            return vectorized_df
            
        except Exception as e:
            logger.warning(f"Vectorization failed for {column_name}: {e}")
            return pd.DataFrame(index=text_series.index)
    
    def _create_linguistic_features(self, text_series: pd.Series, column_name: str) -> pd.DataFrame:
        """
        Create linguistic features
        
        Args:
            text_series: Preprocessed text data
            column_name: Original column name
            
        Returns:
            DataFrame with linguistic features
        """
        try:
            features = {}
            
            # Basic text statistics
            features[f"{column_name}_char_count"] = text_series.str.len()
            features[f"{column_name}_word_count"] = text_series.str.split().str.len()
            features[f"{column_name}_sentence_count"] = text_series.str.count(r'[.!?]+')
            features[f"{column_name}_avg_word_length"] = (
                text_series.str.len() / text_series.str.split().str.len()
            )
            
            # Punctuation features
            features[f"{column_name}_exclamation_count"] = text_series.str.count('!')
            features[f"{column_name}_question_count"] = text_series.str.count(r'\?')
            features[f"{column_name}_period_count"] = text_series.str.count(r'\.')
            
            # Capitalization features
            features[f"{column_name}_uppercase_count"] = text_series.str.count(r'[A-Z]')
            features[f"{column_name}_uppercase_ratio"] = (
                text_series.str.count(r'[A-Z]') / text_series.str.len()
            )
            
            # Digit features
            features[f"{column_name}_digit_count"] = text_series.str.count(r'\d')
            features[f"{column_name}_digit_ratio"] = (
                text_series.str.count(r'\d') / text_series.str.len()
            )
            
            # Vocabulary richness
            def vocabulary_richness(text):
                words = text.split()
                if len(words) == 0:
                    return 0
                return len(set(words)) / len(words)
            
            features[f"{column_name}_vocab_richness"] = text_series.apply(vocabulary_richness)
            
            # Create DataFrame
            linguistic_df = pd.DataFrame(features, index=text_series.index)
            
            # Handle infinite values
            linguistic_df = linguistic_df.replace([np.inf, -np.inf], 0)
            linguistic_df = linguistic_df.fillna(0)
            
            # Add to created features
            self.created_features.extend(linguistic_df.columns.tolist())
            
            logger.info(f"Created {linguistic_df.shape[1]} linguistic features for {column_name}")
            
            return linguistic_df
            
        except Exception as e:
            logger.warning(f"Linguistic feature creation failed for {column_name}: {e}")
            return pd.DataFrame(index=text_series.index)
    
    def transform_new_data(self, 
                          X: Union[np.ndarray, pd.DataFrame],
                          text_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform new data using fitted vectorizers
        
        Args:
            X: New data to transform
            text_columns: List of text column names
            
        Returns:
            Transformed features
        """
        if not self.vectorizers:
            raise ValueError("No fitted vectorizers available. Call engineer_text_features first.")
        
        # Convert to DataFrame
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            df = X.copy()
        
        # Auto-detect text columns if not provided
        if text_columns is None:
            text_columns = self._detect_text_columns(df)
        
        if not text_columns:
            return df
        
        # Process each text column
        engineered_dfs = []
        
        for col in text_columns:
            if col not in df.columns or col not in self.vectorizers:
                continue
            
            # Preprocess text
            processed_text = self._preprocess_text(df[col])
            
            # Transform using fitted vectorizer
            vectorizer = self.vectorizers[col]
            text_matrix = vectorizer.transform(processed_text)
            
            # Convert to DataFrame
            feature_names = [f"{col}_{feat}" for feat in vectorizer.get_feature_names_out()]
            vectorized_df = pd.DataFrame(
                text_matrix.toarray(),
                columns=feature_names,
                index=processed_text.index
            )
            
            # Apply dimensionality reduction if used during training
            if col in self.svd_reducers:
                svd = self.svd_reducers[col]
                reduced_matrix = svd.transform(text_matrix)
                reduced_feature_names = [f"{col}_svd_{i}" for i in range(svd.n_components_)]
                vectorized_df = pd.DataFrame(
                    reduced_matrix,
                    columns=reduced_feature_names,
                    index=processed_text.index
                )
            
            engineered_dfs.append(vectorized_df)
            
            # Create linguistic features
            linguistic_df = self._create_linguistic_features(processed_text, col)
            engineered_dfs.append(linguistic_df)
        
        # Combine engineered features
        if engineered_dfs:
            engineered_features = pd.concat(engineered_dfs, axis=1)
            
            # Combine with original features (excluding original text columns)
            non_text_cols = [col for col in df.columns if col not in text_columns]
            if non_text_cols:
                original_features = df[non_text_cols]
                result_df = pd.concat([original_features, engineered_features], axis=1)
            else:
                result_df = engineered_features
        else:
            result_df = df
        
        return result_df
    
    def get_feature_importance_by_type(self) -> Dict[str, List[str]]:
        """
        Get feature importance by type
        
        Returns:
            Dictionary of feature types and their importance
        """
        feature_types = {
            "vectorized_features": [],
            "linguistic_features": [],
            "svd_features": []
        }
        
        for feature in self.created_features:
            if "svd_" in feature:
                feature_types["svd_features"].append(feature)
            elif any(col in feature for col in ["char_count", "word_count", "sentence_count", 
                                              "exclamation", "question", "period", 
                                              "uppercase", "digit", "vocab"]):
                feature_types["linguistic_features"].append(feature)
            else:
                feature_types["vectorized_features"].append(feature)
        
        return feature_types


def engineer_text_features(X: Union[np.ndarray, pd.DataFrame],
                         text_columns: Optional[List[str]] = None,
                         **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function for text feature engineering
    
    Args:
        X: Input features
        text_columns: List of text column names
        **kwargs: Additional arguments for TextFeatureEngineer
        
    Returns:
        Engineered features and metadata
    """
    engineer = TextFeatureEngineer(**kwargs)
    return engineer.engineer_text_features(X, text_columns)
