# Tensor Operations for Deep Learning

A comprehensive collection of practical tensor operation utilities for PyTorch, focusing on real-world deep learning applications.

## Overview

This package provides reusable, production-ready tensor operations organized into four main categories:

1. **Masking** - Boolean masking for attention, padding, and data filtering
2. **Preprocessing** - Data normalization, standardization, and cleaning
3. **Feature Engineering** - Creating derived features for better model performance
4. **Batch Operations** - Efficient batching and handling variable-length sequences

## Installation

```python
# Add to your project
from pytorch_lab.tensor_ops import (
    BooleanMasking,
    DataPreprocessor,
    FeatureEngineer,
    BatchOperations,
)
```

## Quick Start Examples

### 1. Boolean Masking

#### Attention Masks for Transformers

```python
from pytorch_lab.tensor_ops import create_attention_mask

# Create attention mask for variable-length sequences
seq_lengths = torch.tensor([5, 3, 7])  # Actual lengths
max_length = 10  # Padded length

mask = create_attention_mask(seq_lengths, max_length)
# Use in transformer: outputs = model(input_ids, attention_mask=mask)
```

**Use Case**: BERT, GPT, and other transformer models need to ignore padding tokens during attention computation.

#### Filter Outliers

```python
from pytorch_lab.tensor_ops import BooleanMasking

# Remove outliers from sensor data
data = torch.randn(1000, 10)  # 1000 samples, 10 features
clean_data, mask = BooleanMasking.filter_outliers(data, n_std=3.0)
```

**Use Case**: IoT sensor networks, medical data, financial data - any domain where outliers can corrupt model training.

#### Feature Selection

```python
# Select top-k most important features
features = torch.randn(100, 50)  # 100 samples, 50 features
importance_scores = torch.rand(50)  # From correlation/model analysis

selected = BooleanMasking.select_top_k_features(features, importance_scores, k=10)
```

**Use Case**: Genomics (select relevant genes), financial modeling (choose predictive indicators), dimensionality reduction.

### 2. Data Preprocessing

#### Normalization vs Standardization

```python
from pytorch_lab.tensor_ops import normalize_features, standardize_features

# Normalize to [0, 1] - good for bounded outputs
normalized = normalize_features(data, dim=0)

# Standardize to mean=0, std=1 - good for gradient descent
standardized = standardize_features(data, dim=0)
```

**When to use**:
- **Normalization**: Neural networks with sigmoid/tanh, image data, bounded outputs
- **Standardization**: ReLU networks, features with different units, algorithms assuming normal distribution

#### Proper Train-Test Preprocessing

```python
from pytorch_lab.tensor_ops import DataPreprocessor

# CORRECT way to prevent data leakage
preprocessor = DataPreprocessor()

# Fit on training data only
preprocessor.fit_standardize(train_data, dim=0)

# Transform both train and test
train_processed = preprocessor.standardize(train_data)
test_processed = preprocessor.standardize(test_data)

# Save for production
# preprocessor.save('preprocessor.pkl')
```

**Critical**: Always fit preprocessing on training data only to prevent data leakage!

### 3. Feature Engineering

#### Polynomial Features

```python
from pytorch_lab.tensor_ops import create_polynomial_features

# Add polynomial features for non-linear patterns
features = torch.randn(100, 3)
poly_features = create_polynomial_features(features, degree=2)
# Now includes x, x² for each feature
```

**Use Case**: Physics simulations, economics (diminishing returns), shallow models needing non-linearity.

#### Time-Based Features

```python
from pytorch_lab.tensor_ops import FeatureEngineer

# Cyclical encoding for time features
hours = torch.arange(0, 24)
time_features = FeatureEngineer.create_time_features(hours)
# Returns sin/cos encoding - hour 23 and 0 are now close!
```

**Use Case**: Time series forecasting, retail sales prediction, energy consumption modeling.

#### Lag Features for Time Series

```python
# Create lagged features for autoregressive models
sales = torch.randn(365, 1)  # Daily sales
lag_features = FeatureEngineer.create_lag_features(
    sales, 
    lags=[1, 7, 30]  # Yesterday, last week, last month
)
```

**Use Case**: Stock price prediction, demand forecasting, weather prediction.

### 4. Batch Operations

#### Padding Variable-Length Sequences

```python
from pytorch_lab.tensor_ops import BatchOperations

# Pad sequences for batching
sequences = [
    torch.randn(5, 10),  # Length 5
    torch.randn(3, 10),  # Length 3
    torch.randn(7, 10),  # Length 7
]

padded, lengths = BatchOperations.pad_sequences(sequences)
# padded.shape = (3, 7, 10) - all same length now
```

**Use Case**: NLP (variable-length sentences), time series (different recording lengths), audio processing.

#### Stratified Train-Test Split

```python
from pytorch_lab.tensor_ops import stratified_split

# Maintain class distribution in splits
X_train, y_train, X_test, y_test = stratified_split(
    data, labels, train_ratio=0.8
)
```

**Use Case**: Imbalanced datasets (fraud detection, medical diagnosis, rare event prediction).

#### Class Balancing

```python
from pytorch_lab.tensor_ops import balance_classes

# Balance imbalanced dataset
balanced_data, balanced_labels = balance_classes(
    data, labels, strategy='oversample'
)
```

**Use Case**: Fraud detection (99% normal, 1% fraud), disease diagnosis, anomaly detection.

## Real-World Examples

### Example 1: Building a Sentiment Classifier

```python
from pytorch_lab.tensor_ops import (
    create_attention_mask,
    DataPreprocessor,
    BatchOperations,
)

# 1. Pad variable-length sentences
padded_texts, lengths = BatchOperations.pad_sequences(tokenized_texts)

# 2. Create attention mask
attention_mask = create_attention_mask(lengths, max_length)

# 3. Preprocess features (if using additional features)
preprocessor = DataPreprocessor()
preprocessor.fit_standardize(train_features)
features = preprocessor.standardize(all_features)

# 4. Feed to model
outputs = model(padded_texts, attention_mask=attention_mask, features=features)
```

### Example 2: Time Series Forecasting

```python
from pytorch_lab.tensor_ops import FeatureEngineer, standardize_features

# 1. Create time-based features
time_features = FeatureEngineer.create_time_features(timestamps)

# 2. Create lag features
lag_features = FeatureEngineer.create_lag_features(
    sales_data, lags=[1, 7, 30, 365]
)

# 3. Create rolling statistics
rolling_features = FeatureEngineer.create_rolling_features(
    sales_data, window_sizes=[7, 30, 90]
)

# 4. Combine and standardize
all_features = torch.cat([time_features, lag_features, rolling_features], dim=-1)
standardized = standardize_features(all_features)
```

### Example 3: Handling Imbalanced Medical Data

```python
from pytorch_lab.tensor_ops import (
    BooleanMasking,
    balance_classes,
    stratified_split,
)

# 1. Filter outliers (invalid measurements)
clean_data, mask = BooleanMasking.filter_outliers(medical_data)

# 2. Balance classes (rare disease)
balanced_data, balanced_labels = balance_classes(
    clean_data, labels, strategy='oversample'
)

# 3. Stratified split to maintain class distribution
X_train, y_train, X_test, y_test = stratified_split(
    balanced_data, balanced_labels, train_ratio=0.8
)
```

## API Reference

### Masking Module

- `BooleanMasking.filter_outliers()` - Remove outliers using z-score
- `BooleanMasking.select_top_k_features()` - Feature selection
- `BooleanMasking.mask_invalid_values()` - Handle NaN/Inf
- `create_attention_mask()` - For transformer models
- `create_padding_mask()` - For RNNs/LSTMs
- `create_causal_mask()` - For autoregressive models
- `apply_dropout_mask()` - Dropout regularization

### Preprocessing Module

- `DataPreprocessor` - Stateful preprocessing with fit/transform
- `normalize_features()` - Min-max scaling to [0, 1]
- `standardize_features()` - Z-score normalization
- `handle_missing_values()` - Multiple strategies for NaN
- `clip_gradients()` - Prevent exploding gradients
- `apply_mixup()` - Mixup data augmentation

### Feature Engineering Module

- `FeatureEngineer.create_time_features()` - Cyclical time encoding
- `FeatureEngineer.create_rolling_features()` - Rolling statistics
- `FeatureEngineer.create_lag_features()` - Lagged features
- `create_polynomial_features()` - Polynomial expansion
- `create_interaction_features()` - Feature interactions
- `create_binned_features()` - Discretization
- `create_ratio_features()` - Ratio between features
- `create_difference_features()` - Feature differences

### Batch Operations Module

- `BatchOperations.pad_sequences()` - Pad variable-length sequences
- `BatchOperations.pack_padded_sequence()` - For efficient RNN processing
- `collate_variable_length()` - Custom collate for DataLoader
- `create_mini_batches()` - Manual batching
- `shuffle_batch()` - Shuffle data and labels together
- `stratified_split()` - Maintain class distribution
- `create_k_folds()` - K-fold cross-validation
- `balance_classes()` - Handle imbalanced data

## Best Practices

### 1. Preprocessing Pipeline

```python
# Always follow this order:
# 1. Split data first
# 2. Fit preprocessing on train only
# 3. Transform all sets
# 4. Save preprocessing for production

train, test = split_data(data)
preprocessor.fit(train)
train = preprocessor.transform(train)
test = preprocessor.transform(test)
preprocessor.save('preprocessor.pkl')
```

### 2. Feature Engineering

```python
# Create features before splitting to avoid leakage
features = engineer_features(raw_data)
train, test = split_data(features)
```

### 3. Masking in Models

```python
# Always create masks for variable-length sequences
mask = create_attention_mask(lengths, max_length)
outputs = model(inputs, attention_mask=mask)
```

### 4. Batch Processing

```python
# Use DataLoader with custom collate for efficiency
from torch.utils.data import DataLoader
loader = DataLoader(
    dataset, 
    batch_size=32, 
    collate_fn=collate_variable_length
)
```

## Performance Tips

1. **Use in-place operations** when possible to save memory
2. **Batch operations** instead of loops for GPU efficiency
3. **Cache preprocessed data** to avoid recomputation
4. **Use appropriate dtypes** (float32 vs float64) for memory/speed tradeoff

## Contributing

When adding new operations:
1. Include docstring with use case
2. Add example in this README
3. Write unit tests
4. Document edge cases

## License

MIT License - see LICENSE file for details
