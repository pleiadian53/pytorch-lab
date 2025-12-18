# Gated CNN Architecture for DNA Sequence Encoding

This document explains the **GatedCNNEncoder** and **GatedResidualBlock** architectures used in the MetaSpliceAI project for splice site variant effect prediction. These architectures serve as excellent PyTorch learning examples for building sequence encoders with gating mechanisms.

> **Source**: `meta_spliceai/splice_engine/meta_layer/models/validated_delta_predictor.py`

---

## Table of Contents
1. [Overview](#overview)
2. [GatedResidualBlock](#gatedresidualblock)
3. [GatedCNNEncoder](#gatedcnnencoder)
4. [Data Flow & Dimensionality](#data-flow--dimensionality)
5. [Complete Architecture: ValidatedDeltaPredictor](#complete-architecture-validateddeltapredictor)
6. [PyTorch Implementation Details](#pytorch-implementation-details)

---

## Overview

The Gated CNN architecture is designed for **DNA sequence encoding** with these key features:

- **Dilated convolutions**: Capture long-range dependencies without excessive parameters
- **Gating mechanism**: Learn which features to pass through (inspired by LSTM/GRU gates)
- **Residual connections**: Enable training of deeper networks
- **Global pooling**: Convert variable-length sequences to fixed-size representations

### Why Gating?

Standard CNNs apply the same transformation to all inputs. Gating allows the network to **selectively filter information**:

```
output = content * sigmoid(gate)
```

- When `gate → +∞`: `sigmoid(gate) → 1`, content passes through
- When `gate → -∞`: `sigmoid(gate) → 0`, content is blocked

This is particularly useful for DNA sequences where certain motifs should be emphasized while others are suppressed.

---

## GatedResidualBlock

The `GatedResidualBlock` is the fundamental building block.

### Architecture Diagram

```
Input: x [B, C, L]
         │
         ├─────────────────────────────────┐ (residual connection)
         │                                 │
         ▼                                 │
┌─────────────────────────────────────┐    │
│  Dilated Conv1d(C → 2C, k, d)       │    │
│  kernel_size=k, dilation=d          │    │
│  padding=(k-1)*d//2 (same padding)  │    │
└─────────────────────────────────────┘    │
         │                                 │
         ▼                                 │
    [B, 2C, L]                             │
         │                                 │
         ▼                                 │
┌─────────────────────────────────────┐    │
│  chunk(2, dim=1) → split in half    │    │
│  content: [B, C, L]                 │    │
│  gate:    [B, C, L]                 │    │
└─────────────────────────────────────┘    │
         │                                 │
         ▼                                 │
┌─────────────────────────────────────┐    │
│  Gating: content * sigmoid(gate)    │    │
│  Output: [B, C, L]                  │    │
└─────────────────────────────────────┘    │
         │                                 │
         ▼                                 │
┌─────────────────────────────────────┐    │
│  Permute: [B, C, L] → [B, L, C]     │    │
│  LayerNorm(C)                       │    │
│  Dropout(p)                         │    │
│  Permute: [B, L, C] → [B, C, L]     │    │
└─────────────────────────────────────┘    │
         │                                 │
         ▼                                 │
         + ◄───────────────────────────────┘
         │
         ▼
Output: [B, C, L]
```

### PyTorch Code

```python
class GatedResidualBlock(nn.Module):
    """Gated residual block with dilated convolution."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Padding formula for "same" output length with dilation
        padding = (kernel_size - 1) * dilation // 2
        
        # Single conv outputs 2x channels (for content + gate)
        self.conv = nn.Conv1d(
            channels, channels * 2,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Gated convolution
        out = self.conv(x)              # [B, 2*C, L]
        out, gate = out.chunk(2, dim=1) # Split: [B, C, L] each
        out = out * torch.sigmoid(gate) # Gating
        
        # LayerNorm expects [B, L, C], so permute
        out = out.permute(0, 2, 1)      # [B, L, C]
        out = self.norm(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)      # [B, C, L]
        
        return out + residual           # Residual connection
```

### Dimensionality Trace (Example)

| Step | Operation | Input Shape | Output Shape |
|------|-----------|-------------|--------------|
| 1 | Input | - | `[32, 128, 501]` |
| 2 | Conv1d (k=15, d=2) | `[32, 128, 501]` | `[32, 256, 501]` |
| 3 | chunk(2, dim=1) | `[32, 256, 501]` | 2× `[32, 128, 501]` |
| 4 | Gating (content * σ(gate)) | `[32, 128, 501]` | `[32, 128, 501]` |
| 5 | Permute | `[32, 128, 501]` | `[32, 501, 128]` |
| 6 | LayerNorm | `[32, 501, 128]` | `[32, 501, 128]` |
| 7 | Permute back | `[32, 501, 128]` | `[32, 128, 501]` |
| 8 | Add residual | `[32, 128, 501]` | `[32, 128, 501]` |

**Key insight**: The block preserves dimensions `[B, C, L]` throughout, enabling stacking.

---

## GatedCNNEncoder

The `GatedCNNEncoder` stacks multiple `GatedResidualBlock`s with increasing dilation rates.

### Architecture Diagram

```
Input: One-hot DNA sequence [B, 4, L]
         │
         ▼
┌─────────────────────────────────────┐
│  Embedding Conv1d(4 → H, k=1)       │
│  Projects 4 nucleotides → H dims    │
└─────────────────────────────────────┘
         │
         ▼
    [B, H, L]
         │
         ▼
┌─────────────────────────────────────┐
│  GatedResidualBlock (dilation=1)    │  Layer 0
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  GatedResidualBlock (dilation=2)    │  Layer 1
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  GatedResidualBlock (dilation=4)    │  Layer 2
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  GatedResidualBlock (dilation=8)    │  Layer 3
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  GatedResidualBlock (dilation=1)    │  Layer 4 (cycle repeats)
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  GatedResidualBlock (dilation=2)    │  Layer 5
└─────────────────────────────────────┘
         │
         ▼
    [B, H, L]
         │
         ▼
┌─────────────────────────────────────┐
│  AdaptiveAvgPool1d(1)               │
│  Global average over sequence       │
└─────────────────────────────────────┘
         │
         ▼
    [B, H, 1]
         │
         ▼
┌─────────────────────────────────────┐
│  squeeze(-1)                        │
└─────────────────────────────────────┘
         │
         ▼
Output: [B, H]  (global sequence embedding)
```

### Dilation Pattern

The dilation follows a cyclic pattern: `1, 2, 4, 8, 1, 2, 4, 8, ...`

```python
dilation = 2 ** (i % 4)  # For layer i
```

This creates an **exponentially growing receptive field**:

| Layer | Dilation | Receptive Field Growth |
|-------|----------|------------------------|
| 0 | 1 | 15 positions |
| 1 | 2 | +28 positions |
| 2 | 4 | +56 positions |
| 3 | 8 | +112 positions |
| 4 | 1 | +14 positions |
| 5 | 2 | +28 positions |

With 6 layers and kernel_size=15, the effective receptive field covers **~250+ positions**, sufficient for capturing splice site motifs.

### PyTorch Code

```python
class GatedCNNEncoder(nn.Module):
    """Gated CNN encoder for DNA sequences."""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 6,
        kernel_size: int = 15,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Initial projection: one-hot [4] → hidden_dim
        self.embed = nn.Conv1d(4, hidden_dim, kernel_size=1)
        
        # Dilated residual blocks with cyclic dilation
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 4)  # 1, 2, 4, 8, 1, 2, ...
            self.blocks.append(
                GatedResidualBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence to global features.
        
        Parameters
        ----------
        x : torch.Tensor
            One-hot encoded sequence [B, 4, L]
        
        Returns
        -------
        torch.Tensor
            Global features [B, hidden_dim]
        """
        x = self.embed(x)  # [B, H, L]
        
        for block in self.blocks:
            x = block(x)   # [B, H, L] (preserved)
        
        x = self.pool(x).squeeze(-1)  # [B, H]
        
        return x
```

---

## Data Flow & Dimensionality

### Complete Forward Pass Example

**Configuration**:
- `batch_size = 32`
- `seq_length = 501` (nucleotides)
- `hidden_dim = 128`
- `n_layers = 6`

| Stage | Component | Input Shape | Output Shape | Notes |
|-------|-----------|-------------|--------------|-------|
| 1 | Input (one-hot DNA) | - | `[32, 4, 501]` | A,C,G,T encoding |
| 2 | Embed Conv1d | `[32, 4, 501]` | `[32, 128, 501]` | k=1 projection |
| 3 | GatedResidualBlock×6 | `[32, 128, 501]` | `[32, 128, 501]` | Preserved |
| 4 | AdaptiveAvgPool1d(1) | `[32, 128, 501]` | `[32, 128, 1]` | Global pool |
| 5 | squeeze(-1) | `[32, 128, 1]` | `[32, 128]` | Final embedding |

### Memory Footprint

For a single forward pass with the above configuration:
- Input: `32 × 4 × 501 × 4 bytes = 256 KB`
- After embed: `32 × 128 × 501 × 4 bytes = 8.2 MB`
- Peak (during conv): `32 × 256 × 501 × 4 bytes = 16.4 MB`

---

## Complete Architecture: ValidatedDeltaPredictor

The `GatedCNNEncoder` is used within the `ValidatedDeltaPredictor` for splice variant effect prediction.

### Full Architecture Diagram

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              ValidatedDeltaPredictor                     │
                    └─────────────────────────────────────────────────────────┘
                                              │
            ┌─────────────────────────────────┼─────────────────────────────────┐
            │                                 │                                 │
            ▼                                 ▼                                 ▼
    alt_seq [B, 4, L]               ref_base [B, 4]                   alt_base [B, 4]
            │                                 │                                 │
            ▼                                 └──────────┬──────────────────────┘
┌───────────────────────┐                               │
│   GatedCNNEncoder     │                               ▼
│   (6 layers, H=128)   │                    ┌─────────────────────┐
└───────────────────────┘                    │  concat [B, 8]      │
            │                                └─────────────────────┘
            ▼                                           │
    seq_features [B, H]                                 ▼
            │                                ┌─────────────────────┐
            │                                │  variant_embed      │
            │                                │  Linear(8 → H)      │
            │                                │  ReLU               │
            │                                │  Dropout            │
            │                                │  Linear(H → H)      │
            │                                └─────────────────────┘
            │                                           │
            │                                           ▼
            │                                var_features [B, H]
            │                                           │
            └──────────────────┬────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  concat [B, 2H]     │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  delta_head         │
                    │  Linear(2H → H)     │
                    │  ReLU               │
                    │  Dropout            │
                    │  Linear(H → H/2)    │
                    │  ReLU               │
                    │  Linear(H/2 → 3)    │
                    └─────────────────────┘
                               │
                               ▼
                    delta [B, 3]
                    (Δ_donor, Δ_acceptor, Δ_neither)
```

### Dimensionality Summary

| Tensor | Shape | Description |
|--------|-------|-------------|
| `alt_seq` | `[B, 4, L]` | One-hot encoded alternate sequence |
| `ref_base` | `[B, 4]` | One-hot reference nucleotide |
| `alt_base` | `[B, 4]` | One-hot alternate nucleotide |
| `seq_features` | `[B, H]` | Encoded sequence (H=128) |
| `var_info` | `[B, 8]` | Concatenated ref+alt bases |
| `var_features` | `[B, H]` | Embedded variant info |
| `combined` | `[B, 2H]` | Fused features |
| `delta` | `[B, 3]` | Predicted delta scores |

---

## PyTorch Implementation Details

### Key PyTorch Concepts Demonstrated

1. **`nn.ModuleList`**: For dynamic layer creation
   ```python
   self.blocks = nn.ModuleList([...])  # Properly registers submodules
   ```

2. **`chunk()`**: Splitting tensors along a dimension
   ```python
   out, gate = out.chunk(2, dim=1)  # Split channels in half
   ```

3. **`permute()`**: Reordering dimensions for LayerNorm
   ```python
   out = out.permute(0, 2, 1)  # [B, C, L] → [B, L, C]
   ```

4. **`AdaptiveAvgPool1d`**: Flexible global pooling
   ```python
   self.pool = nn.AdaptiveAvgPool1d(1)  # Any length → 1
   ```

5. **Dilated Convolutions**: Expanding receptive field
   ```python
   nn.Conv1d(..., dilation=dilation, padding=padding)
   ```

### Padding Formula for "Same" Output

For dilated convolutions to preserve sequence length:
```python
padding = (kernel_size - 1) * dilation // 2
```

Example: `kernel_size=15`, `dilation=4`
```
padding = (15 - 1) * 4 // 2 = 28
```

### Why LayerNorm over BatchNorm?

- **LayerNorm**: Normalizes across features for each sample independently
- **BatchNorm**: Normalizes across batch for each feature
- For sequences, LayerNorm is preferred because:
  - Works with variable batch sizes
  - More stable for sequence data
  - No running statistics to maintain

---

## Usage Example

```python
import torch
from validated_delta_predictor import ValidatedDeltaPredictor, one_hot_seq, one_hot_base

# Create model
model = ValidatedDeltaPredictor(
    hidden_dim=128,
    n_layers=6,
    dropout=0.1
)

# Prepare input
seq = "ACGTACGTACGT..." * 40  # 501 nucleotides
alt_seq = torch.tensor(one_hot_seq(seq)).unsqueeze(0)  # [1, 4, 501]
ref_base = torch.tensor(one_hot_base('A')).unsqueeze(0)  # [1, 4]
alt_base = torch.tensor(one_hot_base('G')).unsqueeze(0)  # [1, 4]

# Forward pass
delta = model(alt_seq, ref_base, alt_base)  # [1, 3]

print(f"Δ_donor: {delta[0, 0]:.4f}")
print(f"Δ_acceptor: {delta[0, 1]:.4f}")
print(f"Δ_neither: {delta[0, 2]:.4f}")
```

---

## References

- **Original Project**: MetaSpliceAI - Meta learning layer for alternative splice site detection
- **Gating Mechanism**: Inspired by Gated Linear Units (GLU) from "Language Modeling with Gated Convolutional Networks" (Dauphin et al., 2017)
- **Dilated Convolutions**: "Multi-Scale Context Aggregation by Dilated Convolutions" (Yu & Koltun, 2016)
