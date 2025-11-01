# Tensor Operations: From Tutorial to Production

This document explains the refactoring of the tensor operations notebook into a comprehensive, production-ready package with practical deep learning use cases.

## What Was Created

### 1. New Package Structure

```
pytorch_lab/tensor_ops/
├── __init__.py              # Package exports
├── masking.py               # Boolean masking operations
├── preprocessing.py         # Data preprocessing utilities
├── feature_engineering.py   # Feature creation tools
├── batch_ops.py            # Batch processing utilities
└── README.md               # Comprehensive documentation
```

### 2. Demo Notebook

`examples/tensor_operations_practical_guide.ipynb` - A practical guide showing real-world applications of each operation.

## Answering Your Questions

### Q: What are practical use cases for boolean masking (e.g., `x[x > 5]`)?

Boolean masking is **fundamental** in modern deep learning. Here are the key use cases:

#### 1. **Attention Masks in Transformers** (Most Important!)

**The Problem**: When processing text with BERT/GPT, sentences have different lengths:
- "Hello" → 1 token
- "Hello, how are you today?" → 6 tokens

To batch them, we pad shorter sequences with zeros. But we don't want the model to "attend" to padding!

**The Solution**:
```python
# Create mask: True for real tokens, False for padding
seq_lengths = torch.tensor([5, 3, 7])
mask = create_attention_mask(seq_lengths, max_length=10)

# Use in transformer
outputs = bert_model(input_ids, attention_mask=mask)
```

**Real Impact**: Without attention masks, transformers would:
- Waste computation on padding tokens
- Learn incorrect patterns from padding
- Produce worse predictions

#### 2. **Data Filtering and Outlier Removal**

**The Problem**: Real-world data has errors:
- Sensor malfunctions (temperature = 999°C)
- Data entry mistakes (age = 200)
- Measurement anomalies

**The Solution**:
```python
# Filter outliers using z-score
clean_data, mask = BooleanMasking.filter_outliers(sensor_data, n_std=3.0)
# Only keep samples where all features are within 3 standard deviations
```

**Real Impact**: Training on outliers can:
- Skew model parameters
- Slow convergence
- Reduce generalization

#### 3. **Feature Selection**

**The Problem**: High-dimensional data (1000s of features) causes:
- Slow training
- Overfitting
- Poor interpretability

**The Solution**:
```python
# Select top-k most important features
selected = BooleanMasking.select_top_k_features(
    features, importance_scores, k=10
)
```

**Real Impact**: Used in:
- Genomics: Select relevant genes from 20,000+
- Finance: Choose predictive indicators from 100s of metrics
- Computer Vision: Channel pruning in neural architecture search

#### 4. **Handling Missing/Invalid Values**

**The Problem**: NaN and Inf values crash training or cause numerical instabilities.

**The Solution**:
```python
# Mask and replace invalid values
clean = BooleanMasking.mask_invalid_values(data, fill_value=0.0)
```

**Real Impact**: Prevents:
- NaN propagation through network
- Gradient explosions
- Training crashes

#### 5. **Label Smoothing**

**The Problem**: Hard labels (one-hot) make models overconfident.

**The Solution**:
```python
# Instead of [1, 0, 0], use [0.9, 0.05, 0.05]
smoothed = apply_label_smoothing_mask(targets, num_classes=3, smoothing=0.1)
```

**Real Impact**: Improves:
- Generalization
- Calibration (confidence matches accuracy)
- Robustness to label noise

#### 6. **Causal Masking for Autoregressive Models**

**The Problem**: GPT-style models must not "see the future" during training.

**The Solution**:
```python
# Mask future tokens
mask = create_causal_mask(seq_length=10)
# mask[i, j] is True only if j <= i
```

**Real Impact**: Enables autoregressive generation (GPT, language models).

### Summary: Why Boolean Masking Matters

Boolean masking is **not just a convenience** - it's essential for:

1. **Efficiency**: Skip computation on irrelevant data (padding)
2. **Correctness**: Prevent models from learning wrong patterns
3. **Stability**: Handle invalid values without crashes
4. **Performance**: Remove noise and focus on signal

**Every modern NLP model** (BERT, GPT, T5) uses attention masks extensively!

## Package Organization: Why `pytorch_lab/tensor_ops/`?

### Rationale

1. **Logical Grouping**: Tensor operations are foundational utilities used across multiple projects
2. **Reusability**: Can be imported in any notebook or script
3. **Maintainability**: Centralized location for updates
4. **Discoverability**: Clear namespace (`pytorch_lab.tensor_ops`)

### Alternative Locations Considered

```
Option 1: pytorch_lab/tensor_ops/     ✅ CHOSEN
  + Clear purpose
  + Parallel to other modules (data/, models/, etc.)
  + Easy to import

Option 2: pytorch_lab/utils/tensors/  ❌
  - "utils" is too generic
  - Harder to discover

Option 3: examples/tensor_utils/      ❌
  - Not reusable across projects
  - Mixes examples with utilities
```

## Module Breakdown

### 1. `masking.py` - Boolean Masking Operations

**What it does**: Implements all boolean masking patterns used in deep learning.

**Key functions**:
- `create_attention_mask()` - For transformers
- `create_padding_mask()` - For RNNs
- `create_causal_mask()` - For GPT-style models
- `filter_outliers()` - Data cleaning
- `select_top_k_features()` - Feature selection

**When to use**: 
- Building transformer models
- Processing variable-length sequences
- Cleaning real-world data
- Feature selection pipelines

### 2. `preprocessing.py` - Data Preprocessing

**What it does**: Standardization, normalization, and data cleaning.

**Key functions**:
- `DataPreprocessor` - Stateful preprocessing (fit/transform pattern)
- `normalize_features()` - Min-max scaling to [0, 1]
- `standardize_features()` - Z-score normalization
- `handle_missing_values()` - Multiple strategies for NaN
- `apply_mixup()` - Data augmentation

**When to use**:
- Before training any model
- When features have different scales
- Handling messy real-world data

**Critical Pattern**:
```python
# ALWAYS fit on train, transform both
preprocessor.fit(train_data)
train_processed = preprocessor.transform(train_data)
test_processed = preprocessor.transform(test_data)
```

### 3. `feature_engineering.py` - Feature Creation

**What it does**: Creates derived features that improve model performance.

**Key functions**:
- `create_polynomial_features()` - For non-linear patterns
- `create_time_features()` - Cyclical encoding (hour, day, month)
- `create_lag_features()` - For time series
- `create_rolling_features()` - Moving averages, etc.
- `create_interaction_features()` - Feature combinations

**When to use**:
- Time series forecasting
- Tabular data with non-linear patterns
- When shallow models need more expressiveness
- Domain-specific feature engineering

### 4. `batch_ops.py` - Batch Processing

**What it does**: Handles batching, especially for variable-length sequences.

**Key functions**:
- `pad_sequences()` - Pad to same length
- `collate_variable_length()` - Custom DataLoader collate
- `stratified_split()` - Maintain class distribution
- `balance_classes()` - Handle imbalanced data
- `create_k_folds()` - Cross-validation

**When to use**:
- NLP (variable-length text)
- Time series (different recording lengths)
- Imbalanced datasets
- Custom training loops

## Practical Examples

### Example 1: Building a Sentiment Classifier

```python
from pytorch_lab.tensor_ops import (
    create_attention_mask,
    DataPreprocessor,
    BatchOperations,
)

# 1. Pad variable-length sentences
texts = [tokenize(s) for s in sentences]
padded, lengths = BatchOperations.pad_sequences(texts)

# 2. Create attention mask
mask = create_attention_mask(lengths, max_length)

# 3. Feed to BERT
outputs = bert_model(padded, attention_mask=mask)
```

### Example 2: Time Series Forecasting

```python
from pytorch_lab.tensor_ops import FeatureEngineer, standardize_features

# 1. Create time features (cyclical encoding)
time_features = FeatureEngineer.create_time_features(timestamps)

# 2. Create lag features (yesterday, last week, last month)
lag_features = FeatureEngineer.create_lag_features(
    sales, lags=[1, 7, 30]
)

# 3. Create rolling statistics
rolling = FeatureEngineer.create_rolling_features(
    sales, window_sizes=[7, 30]
)

# 4. Combine and standardize
features = torch.cat([time_features, lag_features, rolling], dim=-1)
features = standardize_features(features)
```

### Example 3: Handling Imbalanced Medical Data

```python
from pytorch_lab.tensor_ops import (
    BooleanMasking,
    balance_classes,
    stratified_split,
)

# 1. Filter outliers (invalid measurements)
clean_data, _ = BooleanMasking.filter_outliers(medical_data)

# 2. Balance classes (rare disease: 1% positive)
balanced_data, balanced_labels = balance_classes(
    clean_data, labels, strategy='oversample'
)

# 3. Stratified split (maintain class distribution)
X_train, y_train, X_test, y_test = stratified_split(
    balanced_data, balanced_labels
)
```

## How to Use This in Your Projects

### 1. Import What You Need

```python
from pytorch_lab.tensor_ops import (
    create_attention_mask,
    DataPreprocessor,
    FeatureEngineer,
    BatchOperations,
)
```

### 2. Follow the Patterns

**Preprocessing**:
```python
preprocessor = DataPreprocessor()
preprocessor.fit_standardize(train_data)
train = preprocessor.standardize(train_data)
test = preprocessor.standardize(test_data)
```

**Feature Engineering**:
```python
# Create features before splitting
features = FeatureEngineer.create_time_features(timestamps)
train, test = split(features)
```

**Batching**:
```python
from torch.utils.data import DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_variable_length
)
```

## Next Steps

### For Learning

1. **Run the demo notebook**: `examples/tensor_operations_practical_guide.ipynb`
2. **Read the module docstrings**: Each function has detailed use cases
3. **Try the examples**: Modify them for your own data

### For Your Projects

1. **Start with preprocessing**: Always normalize/standardize your data
2. **Add masking**: If using transformers or variable-length sequences
3. **Engineer features**: For tabular data or time series
4. **Handle imbalance**: If your dataset has rare classes

### For Contributing

1. **Add new operations**: Follow the existing patterns
2. **Include use cases**: Every function should explain when/why to use it
3. **Write tests**: Ensure correctness
4. **Update docs**: Keep README current

## Key Takeaways

1. **Boolean masking is essential** for modern deep learning, especially NLP
2. **Preprocessing prevents data leakage** when done correctly (fit on train only)
3. **Feature engineering** can dramatically improve model performance
4. **Proper batching** is crucial for variable-length sequences
5. **Reusable modules** save time and reduce errors

## Resources

- **Demo Notebook**: `examples/tensor_operations_practical_guide.ipynb`
- **API Docs**: `pytorch_lab/tensor_ops/README.md`
- **Original Tutorial**: `examples/C1_M1_Lab_3_tensors.ipynb`
- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html

## Questions?

If you have questions about:
- **When to use a specific operation**: Check the function docstring
- **How to apply to your data**: See the examples in the README
- **Why something works this way**: Read the "Use Case" sections
- **Performance optimization**: See the "Performance Tips" in README
