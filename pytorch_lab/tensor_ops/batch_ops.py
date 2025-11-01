"""
Batch Operations for Deep Learning

This module provides utilities for handling batches of data,
including collation, shuffling, and managing variable-length sequences.
"""

import torch
from typing import List, Tuple, Optional


class BatchOperations:
    """
    Utilities for batch processing in deep learning.
    
    Batch operations are essential for:
    - Efficient GPU utilization
    - Stable gradient estimates
    - Handling variable-length sequences
    - Data shuffling and augmentation
    """
    
    @staticmethod
    def pad_sequences(sequences: List[torch.Tensor],
                     padding_value: float = 0.0,
                     batch_first: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad variable-length sequences to same length for batching.
        
        Use Case: In NLP and time series, sequences have different lengths.
        Padding allows batching while preserving original lengths.
        
        Args:
            sequences: List of tensors with shape (seq_length, features)
            padding_value: Value to use for padding
            batch_first: If True, output shape is (batch, seq, features)
            
        Returns:
            padded_sequences: Padded tensor
            lengths: Original sequence lengths
            
        Example:
            >>> seqs = [torch.randn(5, 10), torch.randn(3, 10), torch.randn(7, 10)]
            >>> padded, lengths = BatchOperations.pad_sequences(seqs)
            >>> # padded.shape = (3, 7, 10), lengths = [5, 3, 7]
        """
        lengths = torch.tensor([len(seq) for seq in sequences])
        max_length = lengths.max().item()
        
        # Get feature dimension
        feature_dim = sequences[0].shape[1] if sequences[0].dim() > 1 else 1
        
        # Create padded tensor
        if batch_first:
            padded = torch.full(
                (len(sequences), max_length, feature_dim),
                padding_value,
                dtype=sequences[0].dtype
            )
            for i, seq in enumerate(sequences):
                padded[i, :len(seq)] = seq
        else:
            padded = torch.full(
                (max_length, len(sequences), feature_dim),
                padding_value,
                dtype=sequences[0].dtype
            )
            for i, seq in enumerate(sequences):
                padded[:len(seq), i] = seq
        
        return padded, lengths
    
    @staticmethod
    def pack_padded_sequence(padded: torch.Tensor,
                            lengths: torch.Tensor,
                            batch_first: bool = True) -> torch.nn.utils.rnn.PackedSequence:
        """
        Pack padded sequences for efficient RNN processing.
        
        Use Case: PackedSequence allows RNNs to skip padding tokens,
        improving computational efficiency and preventing padding
        from affecting hidden states.
        
        Args:
            padded: Padded sequences
            lengths: Original sequence lengths
            batch_first: If True, input shape is (batch, seq, features)
            
        Returns:
            PackedSequence object for RNN input
            
        Example:
            >>> padded, lengths = BatchOperations.pad_sequences(sequences)
            >>> packed = BatchOperations.pack_padded_sequence(padded, lengths)
            >>> output, hidden = rnn(packed)
        """
        return torch.nn.utils.rnn.pack_padded_sequence(
            padded, lengths, batch_first=batch_first, enforce_sorted=False
        )
    
    @staticmethod
    def create_attention_batches(queries: torch.Tensor,
                                keys: torch.Tensor,
                                values: torch.Tensor,
                                batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Create batches for attention computation.
        
        Use Case: When computing attention over large sequences,
        batching prevents memory overflow while maintaining efficiency.
        
        Args:
            queries: Query tensor (seq_len, d_model)
            keys: Key tensor (seq_len, d_model)
            values: Value tensor (seq_len, d_model)
            batch_size: Size of each batch
            
        Returns:
            List of (query_batch, key_batch, value_batch) tuples
        """
        num_batches = (len(queries) + batch_size - 1) // batch_size
        batches = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(queries))
            
            q_batch = queries[start_idx:end_idx]
            k_batch = keys[start_idx:end_idx]
            v_batch = values[start_idx:end_idx]
            
            batches.append((q_batch, k_batch, v_batch))
        
        return batches


def collate_variable_length(batch: List[Tuple[torch.Tensor, torch.Tensor]],
                            padding_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader with variable-length sequences.
    
    Use Case: Custom collate function for PyTorch DataLoader when
    working with variable-length sequences (e.g., text, time series).
    
    Args:
        batch: List of (sequence, label) tuples
        padding_value: Value to use for padding
        
    Returns:
        padded_sequences: Batched and padded sequences
        lengths: Original sequence lengths
        labels: Batched labels
        
    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = MyVariableLengthDataset()
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_variable_length)
    """
    sequences, labels = zip(*batch)
    
    # Pad sequences
    lengths = torch.tensor([len(seq) for seq in sequences])
    max_length = lengths.max().item()
    
    padded = torch.full(
        (len(sequences), max_length, sequences[0].shape[-1]),
        padding_value,
        dtype=sequences[0].dtype
    )
    
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    
    # Stack labels
    labels = torch.stack(labels)
    
    return padded, lengths, labels


def create_mini_batches(data: torch.Tensor,
                       labels: torch.Tensor,
                       batch_size: int,
                       shuffle: bool = True) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create mini-batches from data and labels.
    
    Use Case: Manual batching when not using DataLoader, useful for
    custom training loops or when working with multiple data sources.
    
    Args:
        data: Input data tensor (num_samples, ...)
        labels: Label tensor (num_samples, ...)
        batch_size: Size of each mini-batch
        shuffle: Whether to shuffle data before batching
        
    Returns:
        List of (data_batch, label_batch) tuples
        
    Example:
        >>> data = torch.randn(1000, 10)
        >>> labels = torch.randint(0, 2, (1000,))
        >>> batches = create_mini_batches(data, labels, batch_size=32)
        >>> for x_batch, y_batch in batches:
        ...     # Training step
        ...     pass
    """
    num_samples = len(data)
    indices = torch.randperm(num_samples) if shuffle else torch.arange(num_samples)
    
    batches = []
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        data_batch = data[batch_indices]
        label_batch = labels[batch_indices]
        batches.append((data_batch, label_batch))
    
    return batches


def shuffle_batch(data: torch.Tensor,
                 labels: Optional[torch.Tensor] = None,
                 seed: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Shuffle a batch of data (and optionally labels) together.
    
    Use Case: Shuffling batches during training helps prevent the model
    from learning order-dependent patterns and improves generalization.
    
    Args:
        data: Data tensor to shuffle
        labels: Optional label tensor to shuffle in sync
        seed: Random seed for reproducibility
        
    Returns:
        Shuffled data and labels
        
    Example:
        >>> data = torch.randn(100, 10)
        >>> labels = torch.randint(0, 2, (100,))
        >>> shuffled_data, shuffled_labels = shuffle_batch(data, labels)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    indices = torch.randperm(len(data))
    shuffled_data = data[indices]
    shuffled_labels = labels[indices] if labels is not None else None
    
    return shuffled_data, shuffled_labels


def stratified_split(data: torch.Tensor,
                    labels: torch.Tensor,
                    train_ratio: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform stratified train-test split maintaining class distribution.
    
    Use Case: For imbalanced datasets, stratified splitting ensures
    that train and test sets have similar class distributions.
    
    Args:
        data: Input data tensor
        labels: Label tensor
        train_ratio: Ratio of data to use for training
        
    Returns:
        train_data, train_labels, test_data, test_labels
        
    Example:
        >>> data = torch.randn(1000, 10)
        >>> labels = torch.randint(0, 3, (1000,))  # 3 classes
        >>> X_train, y_train, X_test, y_test = stratified_split(data, labels)
    """
    unique_labels = labels.unique()
    train_indices = []
    test_indices = []
    
    for label in unique_labels:
        # Get indices for this class
        class_indices = (labels == label).nonzero(as_tuple=True)[0]
        num_train = int(len(class_indices) * train_ratio)
        
        # Shuffle and split
        perm = torch.randperm(len(class_indices))
        shuffled_indices = class_indices[perm]
        
        train_indices.append(shuffled_indices[:num_train])
        test_indices.append(shuffled_indices[num_train:])
    
    # Concatenate all indices
    train_indices = torch.cat(train_indices)
    test_indices = torch.cat(test_indices)
    
    # Shuffle the final indices
    train_indices = train_indices[torch.randperm(len(train_indices))]
    test_indices = test_indices[torch.randperm(len(test_indices))]
    
    return data[train_indices], labels[train_indices], data[test_indices], labels[test_indices]


def create_k_folds(data: torch.Tensor,
                  labels: torch.Tensor,
                  k: int = 5) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create k-fold cross-validation splits.
    
    Use Case: K-fold cross-validation provides more robust model
    evaluation by training and testing on different data splits.
    
    Args:
        data: Input data tensor
        labels: Label tensor
        k: Number of folds
        
    Returns:
        List of (train_data, train_labels, val_data, val_labels) tuples
        
    Example:
        >>> data = torch.randn(1000, 10)
        >>> labels = torch.randint(0, 2, (1000,))
        >>> folds = create_k_folds(data, labels, k=5)
        >>> for train_x, train_y, val_x, val_y in folds:
        ...     # Train and validate model
        ...     pass
    """
    num_samples = len(data)
    indices = torch.randperm(num_samples)
    fold_size = num_samples // k
    
    folds = []
    for i in range(k):
        # Validation indices for this fold
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else num_samples
        val_indices = indices[val_start:val_end]
        
        # Training indices (everything else)
        train_indices = torch.cat([indices[:val_start], indices[val_end:]])
        
        train_data = data[train_indices]
        train_labels = labels[train_indices]
        val_data = data[val_indices]
        val_labels = labels[val_indices]
        
        folds.append((train_data, train_labels, val_data, val_labels))
    
    return folds


def balance_classes(data: torch.Tensor,
                   labels: torch.Tensor,
                   strategy: str = 'oversample') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Balance imbalanced datasets using oversampling or undersampling.
    
    Use Case: Imbalanced datasets can cause models to be biased toward
    majority classes. Balancing helps improve performance on minority classes.
    
    Args:
        data: Input data tensor
        labels: Label tensor
        strategy: 'oversample' or 'undersample'
        
    Returns:
        Balanced data and labels
        
    Example:
        >>> # Imbalanced dataset: 900 class 0, 100 class 1
        >>> data = torch.randn(1000, 10)
        >>> labels = torch.cat([torch.zeros(900), torch.ones(100)])
        >>> balanced_data, balanced_labels = balance_classes(data, labels)
    """
    unique_labels = labels.unique()
    class_counts = [(labels == label).sum().item() for label in unique_labels]
    
    if strategy == 'oversample':
        # Oversample minority classes to match majority
        target_count = max(class_counts)
        balanced_data = []
        balanced_labels = []
        
        for label in unique_labels:
            class_indices = (labels == label).nonzero(as_tuple=True)[0]
            current_count = len(class_indices)
            
            # Repeat samples to reach target count
            repeats = target_count // current_count
            remainder = target_count % current_count
            
            repeated_indices = class_indices.repeat(repeats)
            if remainder > 0:
                extra_indices = class_indices[torch.randperm(current_count)[:remainder]]
                repeated_indices = torch.cat([repeated_indices, extra_indices])
            
            balanced_data.append(data[repeated_indices])
            balanced_labels.append(labels[repeated_indices])
        
        balanced_data = torch.cat(balanced_data)
        balanced_labels = torch.cat(balanced_labels)
    
    elif strategy == 'undersample':
        # Undersample majority classes to match minority
        target_count = min(class_counts)
        balanced_data = []
        balanced_labels = []
        
        for label in unique_labels:
            class_indices = (labels == label).nonzero(as_tuple=True)[0]
            
            # Randomly select target_count samples
            selected_indices = class_indices[torch.randperm(len(class_indices))[:target_count]]
            
            balanced_data.append(data[selected_indices])
            balanced_labels.append(labels[selected_indices])
        
        balanced_data = torch.cat(balanced_data)
        balanced_labels = torch.cat(balanced_labels)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Shuffle the balanced dataset
    indices = torch.randperm(len(balanced_data))
    return balanced_data[indices], balanced_labels[indices]
