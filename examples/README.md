# Examples and Tutorials

This directory contains Jupyter notebooks and example scripts for learning PyTorch and exploring the pytorch_lab modules.

## Notebooks

### Fundamentals
- `C1_M1_Lab_3_tensors.ipynb` - Introduction to PyTorch tensors (from DeepLearning.AI course)
- `tensor_operations_practical_guide.ipynb` - **NEW!** Practical tensor operations for real-world deep learning

### Planned Learning Path
1. **01_tensor_basics.ipynb** - Tensor operations, shapes, and indexing
2. **02_autograd.ipynb** - Automatic differentiation and backpropagation
3. **03_neural_networks.ipynb** - Building basic neural networks
4. **04_training_loops.ipynb** - Training, validation, and testing
5. **05_custom_layers.ipynb** - Creating custom layers and modules
6. **06_data_loading.ipynb** - Dataset classes and DataLoaders
7. **07_computer_vision.ipynb** - CNN architectures and image classification
8. **08_transfer_learning.ipynb** - Fine-tuning pre-trained models
9. **09_advanced_training.ipynb** - Learning rate schedules, mixed precision, etc.
10. **10_experiment_tracking.ipynb** - TensorBoard and W&B integration

## Data Files

- `data.csv` - Sample data for experiments

## Running Notebooks

```bash
# Activate environment
mamba activate pytorch-lab

# Navigate to examples
cd /Users/pleiadian53/work/pytorch-lab/examples

# Start Jupyter
jupyter notebook
```

## Using pytorch_lab in Notebooks

Since the package is installed in editable mode, you can import it directly:

```python
import pytorch_lab
from pytorch_lab.models import MyModel
from pytorch_lab.training import Trainer
```

Changes to the source code are immediately available - just restart the kernel to reload.
