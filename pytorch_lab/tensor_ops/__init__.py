"""
Tensor Operations Module for PyTorch Lab

This module provides practical, reusable tensor operation utilities
for common deep learning tasks including data preprocessing, masking,
feature engineering, and batch operations.
"""

from .masking import (
    BooleanMasking,
    create_attention_mask,
    create_padding_mask,
    apply_dropout_mask,
)

from .preprocessing import (
    DataPreprocessor,
    normalize_features,
    standardize_features,
    handle_missing_values,
)

from .feature_engineering import (
    FeatureEngineer,
    create_polynomial_features,
    create_interaction_features,
    create_binned_features,
)

from .batch_ops import (
    BatchOperations,
    collate_variable_length,
    create_mini_batches,
    shuffle_batch,
)

__all__ = [
    # Masking
    'BooleanMasking',
    'create_attention_mask',
    'create_padding_mask',
    'apply_dropout_mask',
    
    # Preprocessing
    'DataPreprocessor',
    'normalize_features',
    'standardize_features',
    'handle_missing_values',
    
    # Feature Engineering
    'FeatureEngineer',
    'create_polynomial_features',
    'create_interaction_features',
    'create_binned_features',
    
    # Batch Operations
    'BatchOperations',
    'collate_variable_length',
    'create_mini_batches',
    'shuffle_batch',
]
