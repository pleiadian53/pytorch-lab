"""
Feature Engineering Operations for Deep Learning

This module provides tensor-based feature engineering utilities
for creating new features from existing ones, which can improve
model performance on tabular and structured data.
"""

import torch
from typing import List, Optional, Tuple


class FeatureEngineer:
    """
    Feature engineering utilities for creating derived features.
    
    Feature engineering is crucial for:
    - Improving model performance on tabular data
    - Capturing non-linear relationships
    - Creating domain-specific features
    - Reducing feature dimensionality
    """
    
    @staticmethod
    def create_time_features(timestamps: torch.Tensor) -> torch.Tensor:
        """
        Extract cyclical time features from timestamps.
        
        Use Case: In time series forecasting, cyclical encoding of time
        features (hour, day, month) helps models learn periodic patterns.
        
        Args:
            timestamps: Unix timestamps or datetime values
            
        Returns:
            Tensor with cyclical time features (sin/cos encoding)
            
        Example:
            >>> # For a time series prediction model
            >>> timestamps = torch.arange(0, 24)  # Hours 0-23
            >>> time_features = FeatureEngineer.create_time_features(timestamps)
        """
        # Normalize to [0, 1] range (assuming 24-hour cycle)
        normalized = (timestamps % 24) / 24.0
        
        # Create cyclical features using sin/cos
        sin_features = torch.sin(2 * torch.pi * normalized)
        cos_features = torch.cos(2 * torch.pi * normalized)
        
        return torch.stack([sin_features, cos_features], dim=-1)
    
    @staticmethod
    def create_rolling_features(data: torch.Tensor,
                               window_sizes: List[int]) -> torch.Tensor:
        """
        Create rolling window statistics (mean, std, min, max).
        
        Use Case: In time series and sequential data, rolling statistics
        capture local trends and patterns that help with prediction.
        
        Args:
            data: Time series data of shape (seq_length, features)
            window_sizes: List of window sizes for rolling statistics
            
        Returns:
            Tensor with rolling features concatenated
            
        Example:
            >>> prices = torch.randn(100, 1)  # Stock prices
            >>> rolling_features = FeatureEngineer.create_rolling_features(
            ...     prices, window_sizes=[5, 10, 20]
            ... )
        """
        features = []
        
        for window in window_sizes:
            # Pad the beginning to maintain sequence length
            padded = torch.nn.functional.pad(data, (0, 0, window-1, 0), mode='replicate')
            
            # Compute rolling statistics
            rolling_mean = torch.nn.functional.avg_pool1d(
                padded.transpose(0, 1).unsqueeze(0),
                kernel_size=window,
                stride=1
            ).squeeze(0).transpose(0, 1)
            
            features.append(rolling_mean)
        
        return torch.cat(features, dim=-1)
    
    @staticmethod
    def create_lag_features(data: torch.Tensor,
                           lags: List[int]) -> torch.Tensor:
        """
        Create lagged versions of features for time series.
        
        Use Case: In autoregressive models, past values are important
        predictors of future values. Lag features explicitly provide
        this historical context.
        
        Args:
            data: Time series data of shape (seq_length, features)
            lags: List of lag values to create
            
        Returns:
            Tensor with lagged features
            
        Example:
            >>> sales = torch.randn(365, 1)  # Daily sales
            >>> lag_features = FeatureEngineer.create_lag_features(
            ...     sales, lags=[1, 7, 30]  # Yesterday, last week, last month
            ... )
        """
        lagged_features = []
        
        for lag in lags:
            # Shift data by lag positions
            lagged = torch.roll(data, shifts=lag, dims=0)
            # Set first 'lag' positions to 0 (no valid history)
            lagged[:lag] = 0
            lagged_features.append(lagged)
        
        return torch.cat(lagged_features, dim=-1)
    
    @staticmethod
    def create_embedding_features(categorical_data: torch.Tensor,
                                 num_categories: int,
                                 embedding_dim: int) -> torch.Tensor:
        """
        Create learned embeddings for categorical features.
        
        Use Case: For high-cardinality categorical features (e.g., user IDs,
        product IDs), embeddings learn dense representations that capture
        semantic relationships.
        
        Args:
            categorical_data: Integer tensor with category indices
            num_categories: Total number of categories
            embedding_dim: Dimension of embedding vectors
            
        Returns:
            Embedded features
            
        Example:
            >>> user_ids = torch.randint(0, 1000, (100,))  # 100 users
            >>> embedding = torch.nn.Embedding(1000, 32)
            >>> user_embeddings = embedding(user_ids)
        """
        embedding = torch.nn.Embedding(num_categories, embedding_dim)
        return embedding(categorical_data)


def create_polynomial_features(data: torch.Tensor,
                               degree: int = 2) -> torch.Tensor:
    """
    Create polynomial features up to specified degree.
    
    Use Case: Polynomial features help models capture non-linear
    relationships without requiring deep architectures. Useful for
    shallow models on tabular data.
    
    Args:
        data: Input features of shape (batch_size, num_features)
        degree: Maximum polynomial degree
        
    Returns:
        Tensor with original and polynomial features
        
    Example:
        >>> features = torch.randn(100, 3)  # 3 features
        >>> poly_features = create_polynomial_features(features, degree=2)
        >>> # Now includes x, x^2 for each feature
    """
    features = [data]
    
    for d in range(2, degree + 1):
        features.append(torch.pow(data, d))
    
    return torch.cat(features, dim=-1)


def create_interaction_features(data: torch.Tensor,
                                max_interactions: int = 2) -> torch.Tensor:
    """
    Create interaction features between pairs of features.
    
    Use Case: Interaction features capture relationships between features
    (e.g., age * income) that may be important for prediction but not
    captured by individual features.
    
    Args:
        data: Input features of shape (batch_size, num_features)
        max_interactions: Maximum order of interactions (2 = pairwise)
        
    Returns:
        Tensor with original and interaction features
        
    Example:
        >>> features = torch.randn(100, 4)
        >>> with_interactions = create_interaction_features(features)
        >>> # Includes original features plus all pairwise products
    """
    batch_size, num_features = data.shape
    features = [data]
    
    if max_interactions >= 2:
        # Create pairwise interactions
        for i in range(num_features):
            for j in range(i + 1, num_features):
                interaction = data[:, i:i+1] * data[:, j:j+1]
                features.append(interaction)
    
    return torch.cat(features, dim=-1)


def create_binned_features(data: torch.Tensor,
                          num_bins: int = 10,
                          strategy: str = 'uniform') -> torch.Tensor:
    """
    Create binned (discretized) versions of continuous features.
    
    Use Case: Binning can help models learn non-linear patterns and
    reduce the impact of outliers. Useful for decision tree ensembles
    and when features have non-monotonic relationships with target.
    
    Args:
        data: Continuous features of shape (batch_size, num_features)
        num_bins: Number of bins to create
        strategy: Binning strategy ('uniform' or 'quantile')
        
    Returns:
        One-hot encoded binned features
        
    Example:
        >>> ages = torch.randn(100, 1) * 20 + 40  # Ages around 40
        >>> age_bins = create_binned_features(ages, num_bins=5)
        >>> # Creates 5 age groups as one-hot encoded features
    """
    batch_size, num_features = data.shape
    binned_features = []
    
    for feat_idx in range(num_features):
        feature = data[:, feat_idx]
        
        if strategy == 'uniform':
            # Equal-width bins
            min_val, max_val = feature.min(), feature.max()
            bins = torch.linspace(min_val, max_val, num_bins + 1)
        elif strategy == 'quantile':
            # Equal-frequency bins
            sorted_vals, _ = torch.sort(feature)
            quantiles = torch.linspace(0, len(sorted_vals) - 1, num_bins + 1).long()
            bins = sorted_vals[quantiles]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Assign to bins
        bin_indices = torch.searchsorted(bins[:-1], feature, right=False)
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
        
        # One-hot encode
        one_hot = torch.nn.functional.one_hot(bin_indices, num_classes=num_bins)
        binned_features.append(one_hot.float())
    
    return torch.cat(binned_features, dim=-1)


def create_statistical_features(data: torch.Tensor,
                                window_size: int = 5) -> torch.Tensor:
    """
    Create statistical features over a sliding window.
    
    Use Case: In time series or sequential data, local statistics
    (mean, std, min, max) provide useful context for predictions.
    
    Args:
        data: Sequential data of shape (seq_length, features)
        window_size: Size of sliding window
        
    Returns:
        Tensor with statistical features
        
    Example:
        >>> sensor_data = torch.randn(1000, 3)  # 1000 timesteps, 3 sensors
        >>> stats = create_statistical_features(sensor_data, window_size=10)
    """
    seq_length, num_features = data.shape
    
    # Pad to maintain sequence length
    padded = torch.nn.functional.pad(
        data.unsqueeze(0),
        (0, 0, window_size-1, 0),
        mode='replicate'
    ).squeeze(0)
    
    # Compute rolling statistics
    features = []
    for i in range(seq_length):
        window = padded[i:i+window_size]
        
        mean = window.mean(dim=0)
        std = window.std(dim=0)
        min_val = window.min(dim=0)[0]
        max_val = window.max(dim=0)[0]
        
        stats = torch.cat([mean, std, min_val, max_val])
        features.append(stats)
    
    return torch.stack(features)


def create_ratio_features(data: torch.Tensor,
                         feature_pairs: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Create ratio features between pairs of features.
    
    Use Case: Ratios can capture important relationships (e.g., debt-to-income
    ratio in finance, price-to-earnings in stocks) that are more predictive
    than individual features.
    
    Args:
        data: Input features of shape (batch_size, num_features)
        feature_pairs: List of (numerator_idx, denominator_idx) tuples
        
    Returns:
        Tensor with ratio features
        
    Example:
        >>> financial_data = torch.randn(100, 10)
        >>> # Create debt/income and assets/liabilities ratios
        >>> ratios = create_ratio_features(financial_data, [(0, 1), (2, 3)])
    """
    ratios = []
    
    for num_idx, denom_idx in feature_pairs:
        numerator = data[:, num_idx:num_idx+1]
        denominator = data[:, denom_idx:denom_idx+1]
        
        # Add small epsilon to avoid division by zero
        ratio = numerator / (denominator + 1e-8)
        ratios.append(ratio)
    
    return torch.cat(ratios, dim=-1)


def create_difference_features(data: torch.Tensor,
                               feature_pairs: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Create difference features between pairs of features.
    
    Use Case: Differences can highlight changes or gaps that are important
    for prediction (e.g., price changes, temperature differences).
    
    Args:
        data: Input features of shape (batch_size, num_features)
        feature_pairs: List of (feature1_idx, feature2_idx) tuples
        
    Returns:
        Tensor with difference features
        
    Example:
        >>> time_series = torch.randn(100, 5)
        >>> # Create differences between consecutive features
        >>> diffs = create_difference_features(time_series, [(0, 1), (1, 2)])
    """
    differences = []
    
    for idx1, idx2 in feature_pairs:
        diff = data[:, idx1:idx1+1] - data[:, idx2:idx2+1]
        differences.append(diff)
    
    return torch.cat(differences, dim=-1)
