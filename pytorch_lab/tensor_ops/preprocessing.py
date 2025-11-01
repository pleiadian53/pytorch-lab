"""
Data Preprocessing Operations for Deep Learning

This module provides tensor-based preprocessing utilities commonly used
in deep learning pipelines, including normalization, standardization,
and handling missing values.
"""

import torch
from typing import Optional, Tuple


class DataPreprocessor:
    """
    Comprehensive data preprocessing utilities for deep learning.
    
    These operations are essential for:
    - Improving model convergence speed
    - Preventing numerical instabilities
    - Handling real-world messy data
    - Ensuring features are on similar scales
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
    
    def fit_normalize(self, data: torch.Tensor, dim: int = 0) -> 'DataPreprocessor':
        """
        Fit normalization parameters (min, max) from training data.
        
        Use Case: Learn normalization parameters from training set,
        then apply to validation/test sets to prevent data leakage.
        
        Args:
            data: Training data tensor
            dim: Dimension along which to compute statistics
            
        Returns:
            self for method chaining
        """
        self.min = data.min(dim=dim, keepdim=True)[0]
        self.max = data.max(dim=dim, keepdim=True)[0]
        return self
    
    def fit_standardize(self, data: torch.Tensor, dim: int = 0) -> 'DataPreprocessor':
        """
        Fit standardization parameters (mean, std) from training data.
        
        Use Case: Learn standardization parameters from training set,
        ensuring consistent preprocessing across train/val/test splits.
        
        Args:
            data: Training data tensor
            dim: Dimension along which to compute statistics
            
        Returns:
            self for method chaining
        """
        self.mean = data.mean(dim=dim, keepdim=True)
        self.std = data.std(dim=dim, keepdim=True)
        return self
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply min-max normalization to scale data to [0, 1].
        
        Use Case: When features have different ranges and you want
        to scale them to a common range, especially useful for
        neural networks with sigmoid/tanh activations.
        
        Args:
            data: Input tensor to normalize
            
        Returns:
            Normalized tensor with values in [0, 1]
        """
        if self.min is None or self.max is None:
            raise ValueError("Must call fit_normalize() before normalize()")
        
        return (data - self.min) / (self.max - self.min + 1e-8)
    
    def standardize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply z-score standardization (mean=0, std=1).
        
        Use Case: When features have different scales and you want
        them to have zero mean and unit variance, which helps with
        gradient descent convergence.
        
        Args:
            data: Input tensor to standardize
            
        Returns:
            Standardized tensor with mean≈0, std≈1
        """
        if self.mean is None or self.std is None:
            raise ValueError("Must call fit_standardize() before standardize()")
        
        return (data - self.mean) / (self.std + 1e-8)
    
    def inverse_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization to get original scale.
        
        Use Case: After making predictions with a normalized model,
        convert predictions back to original scale for interpretation.
        """
        if self.min is None or self.max is None:
            raise ValueError("Must call fit_normalize() before inverse_normalize()")
        
        return data * (self.max - self.min) + self.min
    
    def inverse_standardize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverse standardization to get original scale.
        
        Use Case: Convert model predictions back to original scale
        for interpretation and evaluation.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Must call fit_standardize() before inverse_standardize()")
        
        return data * self.std + self.mean


def normalize_features(data: torch.Tensor, 
                      dim: int = 0,
                      eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize features to [0, 1] range using min-max scaling.
    
    Use Case: Image preprocessing, where pixel values are scaled to [0, 1]
    for better neural network training.
    
    Args:
        data: Input tensor of shape (batch_size, features)
        dim: Dimension along which to compute min/max
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor
        
    Example:
        >>> images = torch.randint(0, 256, (32, 3, 224, 224)).float()  # RGB images
        >>> normalized = normalize_features(images, dim=0)
        >>> assert normalized.min() >= 0 and normalized.max() <= 1
    """
    min_val = data.min(dim=dim, keepdim=True)[0]
    max_val = data.max(dim=dim, keepdim=True)[0]
    
    normalized = (data - min_val) / (max_val - min_val + eps)
    return normalized


def standardize_features(data: torch.Tensor,
                        dim: int = 0,
                        eps: float = 1e-8) -> torch.Tensor:
    """
    Standardize features to have mean=0 and std=1.
    
    Use Case: Preprocessing tabular data before feeding to neural networks,
    ensuring all features contribute equally to the learning process.
    
    Args:
        data: Input tensor of shape (batch_size, features)
        dim: Dimension along which to compute mean/std
        eps: Small constant for numerical stability
        
    Returns:
        Standardized tensor
        
    Example:
        >>> features = torch.randn(1000, 20) * 10 + 5  # Random features
        >>> standardized = standardize_features(features, dim=0)
        >>> assert torch.allclose(standardized.mean(0), torch.zeros(20), atol=1e-6)
    """
    mean = data.mean(dim=dim, keepdim=True)
    std = data.std(dim=dim, keepdim=True)
    
    standardized = (data - mean) / (std + eps)
    return standardized


def handle_missing_values(data: torch.Tensor,
                         strategy: str = 'mean',
                         fill_value: Optional[float] = None) -> torch.Tensor:
    """
    Handle missing values (NaN) in data using various strategies.
    
    Use Case: Real-world datasets often have missing values. This function
    provides common strategies to handle them before training.
    
    Args:
        data: Input tensor that may contain NaN values
        strategy: Strategy to handle missing values:
            - 'mean': Replace with column mean
            - 'median': Replace with column median
            - 'zero': Replace with 0
            - 'forward_fill': Replace with previous valid value
            - 'constant': Replace with fill_value
        fill_value: Value to use when strategy='constant'
        
    Returns:
        Tensor with missing values handled
        
    Example:
        >>> data = torch.tensor([[1.0, float('nan'), 3.0],
        ...                      [4.0, 5.0, float('nan')]])
        >>> clean = handle_missing_values(data, strategy='mean')
    """
    if not torch.isnan(data).any():
        return data
    
    result = data.clone()
    
    if strategy == 'mean':
        # Replace NaN with column mean
        for col in range(data.shape[1]):
            col_data = data[:, col]
            mask = ~torch.isnan(col_data)
            if mask.any():
                mean_val = col_data[mask].mean()
                result[torch.isnan(result[:, col]), col] = mean_val
    
    elif strategy == 'median':
        # Replace NaN with column median
        for col in range(data.shape[1]):
            col_data = data[:, col]
            mask = ~torch.isnan(col_data)
            if mask.any():
                median_val = col_data[mask].median()
                result[torch.isnan(result[:, col]), col] = median_val
    
    elif strategy == 'zero':
        result[torch.isnan(result)] = 0.0
    
    elif strategy == 'forward_fill':
        # Replace NaN with previous valid value (row-wise)
        for row in range(data.shape[0]):
            row_data = result[row]
            mask = torch.isnan(row_data)
            if mask.any():
                # Forward fill
                last_valid = None
                for i in range(len(row_data)):
                    if not torch.isnan(row_data[i]):
                        last_valid = row_data[i].item()
                    elif last_valid is not None:
                        result[row, i] = last_valid
                # If still NaN at start, use 0
                result[row][torch.isnan(result[row])] = 0.0
    
    elif strategy == 'constant':
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy='constant'")
        result[torch.isnan(result)] = fill_value
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return result


def clip_gradients(gradients: torch.Tensor,
                  max_norm: float = 1.0) -> torch.Tensor:
    """
    Clip gradients to prevent exploding gradients problem.
    
    Use Case: In RNNs and deep networks, gradients can explode during
    backpropagation. Clipping prevents this and stabilizes training.
    
    Args:
        gradients: Gradient tensor
        max_norm: Maximum allowed norm
        
    Returns:
        Clipped gradients
        
    Example:
        >>> # During training loop:
        >>> loss.backward()
        >>> for param in model.parameters():
        ...     if param.grad is not None:
        ...         param.grad = clip_gradients(param.grad, max_norm=1.0)
        >>> optimizer.step()
    """
    norm = gradients.norm()
    if norm > max_norm:
        gradients = gradients * (max_norm / norm)
    return gradients


def apply_mixup(x1: torch.Tensor,
               x2: torch.Tensor,
               y1: torch.Tensor,
               y2: torch.Tensor,
               alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mixup data augmentation for better generalization.
    
    Use Case: Mixup is a data augmentation technique that creates virtual
    training examples by mixing pairs of examples and their labels,
    improving model robustness and generalization.
    
    Args:
        x1, x2: Input tensors to mix
        y1, y2: Corresponding labels
        alpha: Beta distribution parameter (higher = more mixing)
        
    Returns:
        Mixed inputs and labels
        
    Example:
        >>> x1 = torch.randn(32, 3, 224, 224)  # Batch of images
        >>> x2 = torch.randn(32, 3, 224, 224)
        >>> y1 = torch.randint(0, 10, (32,))
        >>> y2 = torch.randint(0, 10, (32,))
        >>> x_mixed, y_mixed = apply_mixup(x1, x2, y1, y2)
    """
    # Sample mixing coefficient from Beta distribution
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = 1.0
    
    # Mix inputs
    x_mixed = lam * x1 + (1 - lam) * x2
    
    # Mix labels (for one-hot encoded labels)
    y_mixed = lam * y1 + (1 - lam) * y2
    
    return x_mixed, y_mixed
