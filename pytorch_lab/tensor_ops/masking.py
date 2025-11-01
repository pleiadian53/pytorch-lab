"""
Boolean Masking Operations for Deep Learning

This module demonstrates practical use cases of boolean masking in deep learning:
1. Attention Masks - For transformer models to ignore padding tokens
2. Padding Masks - For variable-length sequences in RNNs
3. Dropout Masks - For regularization during training
4. Data Filtering - For outlier detection and data cleaning
"""

import torch
from typing import Optional, Tuple


class BooleanMasking:
    """
    Comprehensive boolean masking utilities for deep learning applications.
    
    Boolean masking is crucial in deep learning for:
    - Handling variable-length sequences (NLP, time series)
    - Implementing attention mechanisms (Transformers)
    - Data filtering and outlier removal
    - Selective gradient computation
    """
    
    @staticmethod
    def filter_outliers(data: torch.Tensor, 
                       n_std: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter outliers using boolean masking (z-score method).
        
        Use Case: Data cleaning before training to remove anomalous samples
        that could negatively impact model convergence.
        
        Args:
            data: Input tensor of shape (batch_size, features)
            n_std: Number of standard deviations for outlier threshold
            
        Returns:
            filtered_data: Data with outliers removed
            mask: Boolean mask indicating which samples were kept
            
        Example:
            >>> data = torch.randn(1000, 10)
            >>> clean_data, mask = BooleanMasking.filter_outliers(data)
            >>> print(f"Kept {mask.sum()}/{len(mask)} samples")
        """
        mean = data.mean(dim=0)
        std = data.std(dim=0)
        
        # Create boolean mask: True for values within n_std standard deviations
        z_scores = torch.abs((data - mean) / (std + 1e-8))
        mask = (z_scores < n_std).all(dim=1)
        
        filtered_data = data[mask]
        return filtered_data, mask
    
    @staticmethod
    def select_top_k_features(features: torch.Tensor, 
                             scores: torch.Tensor, 
                             k: int) -> torch.Tensor:
        """
        Select top-k features based on importance scores using masking.
        
        Use Case: Feature selection in high-dimensional data to reduce
        computational cost and prevent overfitting.
        
        Args:
            features: Feature tensor of shape (batch_size, num_features)
            scores: Feature importance scores of shape (num_features,)
            k: Number of top features to select
            
        Returns:
            Selected features of shape (batch_size, k)
            
        Example:
            >>> features = torch.randn(100, 50)  # 100 samples, 50 features
            >>> scores = torch.rand(50)  # Feature importance scores
            >>> top_features = BooleanMasking.select_top_k_features(features, scores, k=10)
        """
        # Get indices of top-k scores
        _, top_indices = torch.topk(scores, k)
        
        # Create boolean mask
        mask = torch.zeros(features.shape[1], dtype=torch.bool)
        mask[top_indices] = True
        
        # Select features
        selected_features = features[:, mask]
        return selected_features
    
    @staticmethod
    def mask_invalid_values(data: torch.Tensor, 
                           fill_value: float = 0.0) -> torch.Tensor:
        """
        Mask and replace invalid values (NaN, Inf) in data.
        
        Use Case: Handling numerical instabilities during training,
        especially in loss computation or gradient calculations.
        
        Args:
            data: Input tensor that may contain invalid values
            fill_value: Value to replace invalid entries with
            
        Returns:
            Cleaned tensor with invalid values replaced
            
        Example:
            >>> data = torch.tensor([1.0, float('nan'), 3.0, float('inf')])
            >>> clean = BooleanMasking.mask_invalid_values(data)
        """
        # Create mask for valid values
        mask = torch.isfinite(data)
        
        # Replace invalid values
        cleaned_data = data.clone()
        cleaned_data[~mask] = fill_value
        
        return cleaned_data


def create_attention_mask(seq_lengths: torch.Tensor, 
                         max_length: int) -> torch.Tensor:
    """
    Create attention mask for transformer models.
    
    Use Case: In transformer models (BERT, GPT), attention masks prevent
    the model from attending to padding tokens, which improves training
    efficiency and model quality.
    
    Args:
        seq_lengths: Actual lengths of sequences in batch (batch_size,)
        max_length: Maximum sequence length (padded length)
        
    Returns:
        Attention mask of shape (batch_size, max_length)
        True for real tokens, False for padding
        
    Example:
        >>> seq_lengths = torch.tensor([5, 3, 7])  # 3 sequences
        >>> mask = create_attention_mask(seq_lengths, max_length=10)
        >>> # Use in transformer: attention_output = model(input_ids, attention_mask=mask)
    """
    batch_size = seq_lengths.size(0)
    
    # Create position indices
    positions = torch.arange(max_length).unsqueeze(0).expand(batch_size, -1)
    
    # Create mask: True where position < sequence length
    mask = positions < seq_lengths.unsqueeze(1)
    
    return mask


def create_padding_mask(sequences: torch.Tensor, 
                       pad_value: float = 0.0) -> torch.Tensor:
    """
    Create padding mask by detecting padding values.
    
    Use Case: For RNNs and LSTMs processing variable-length sequences,
    padding masks ensure that padding tokens don't affect the hidden states.
    
    Args:
        sequences: Padded sequences of shape (batch_size, seq_length, features)
        pad_value: Value used for padding
        
    Returns:
        Mask of shape (batch_size, seq_length)
        True for real tokens, False for padding
        
    Example:
        >>> sequences = torch.tensor([
        ...     [[1, 2], [3, 4], [0, 0]],  # Length 2, padded to 3
        ...     [[5, 6], [7, 8], [9, 10]]  # Length 3
        ... ])
        >>> mask = create_padding_mask(sequences, pad_value=0.0)
    """
    # Check if all features in a position are equal to pad_value
    mask = (sequences != pad_value).any(dim=-1)
    return mask


def apply_dropout_mask(x: torch.Tensor, 
                      dropout_rate: float = 0.5,
                      training: bool = True) -> torch.Tensor:
    """
    Apply dropout using boolean masking for regularization.
    
    Use Case: Dropout is a regularization technique that randomly sets
    a fraction of input units to 0 during training, preventing overfitting.
    
    Args:
        x: Input tensor of any shape
        dropout_rate: Probability of dropping a unit (0 to 1)
        training: If False, returns input unchanged (inference mode)
        
    Returns:
        Tensor with dropout applied
        
    Example:
        >>> x = torch.randn(32, 128)  # Batch of 32, 128 features
        >>> x_dropped = apply_dropout_mask(x, dropout_rate=0.3, training=True)
    """
    if not training or dropout_rate == 0.0:
        return x
    
    # Create dropout mask: True for units to keep
    keep_prob = 1.0 - dropout_rate
    mask = torch.rand_like(x) < keep_prob
    
    # Apply mask and scale to maintain expected value
    output = x * mask.float() / keep_prob
    
    return output


def create_causal_mask(seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create causal (autoregressive) mask for decoder-only transformers.
    
    Use Case: In GPT-style models, causal masking ensures that predictions
    for position i can only depend on positions < i, enabling autoregressive
    generation.
    
    Args:
        seq_length: Length of the sequence
        device: Device to create the mask on
        
    Returns:
        Causal mask of shape (seq_length, seq_length)
        True where attention is allowed, False where it should be masked
        
    Example:
        >>> mask = create_causal_mask(5)
        >>> # mask[i, j] is True only if j <= i
        >>> # This prevents "looking ahead" in the sequence
    """
    # Create lower triangular matrix
    mask = torch.tril(torch.ones(seq_length, seq_length, device=device)).bool()
    return mask


def apply_label_smoothing_mask(targets: torch.Tensor,
                               num_classes: int,
                               smoothing: float = 0.1) -> torch.Tensor:
    """
    Apply label smoothing using masking for better generalization.
    
    Use Case: Label smoothing prevents the model from becoming overconfident
    by distributing some probability mass to incorrect classes, improving
    generalization and calibration.
    
    Args:
        targets: Target class indices of shape (batch_size,)
        num_classes: Total number of classes
        smoothing: Smoothing factor (0 to 1)
        
    Returns:
        Smoothed target distribution of shape (batch_size, num_classes)
        
    Example:
        >>> targets = torch.tensor([0, 2, 1])  # 3 samples
        >>> smoothed = apply_label_smoothing_mask(targets, num_classes=5, smoothing=0.1)
        >>> # Instead of [1, 0, 0, 0, 0], target becomes [0.925, 0.025, 0.025, 0.025, 0.025]
    """
    batch_size = targets.size(0)
    
    # Create one-hot encoding
    one_hot = torch.zeros(batch_size, num_classes, device=targets.device)
    one_hot.scatter_(1, targets.unsqueeze(1), 1)
    
    # Apply smoothing
    smoothed = one_hot * (1 - smoothing) + smoothing / num_classes
    
    return smoothed
