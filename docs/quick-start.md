# Quick Start Guide

## TL;DR - Complete Installation

```bash
# Create and activate environment
mamba create -n pytorch-lab python=3.11 pip -y
mamba activate pytorch-lab

# Install PyTorch (CPU version for Mac)
mamba install -c pytorch pytorch torchvision -y

# Install other dependencies
mamba install -c conda-forge numpy pandas matplotlib seaborn scikit-learn -y
pip install pydantic typer rich tqdm

# Install pytorch-lab in editable mode
cd /Users/pleiadian53/work/pytorch-lab
pip install -e .

# Optional: Install dev tools
pip install -e ".[dev]"

# Verify installation
python -c "import pytorch_lab; print('✓ Success!')"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Daily Workflow

### Starting Work

```bash
# Activate environment
mamba activate pytorch-lab

# Navigate to project
cd /Users/pleiadian53/work/pytorch-lab
```

### Working with the Code

Since the package is installed in **editable mode** (`pip install -e .`):

- ✅ Edit any file in `pytorch_lab/` and changes are immediately available
- ✅ No need to reinstall after code changes
- ✅ Import from anywhere: `from pytorch_lab.models import ...`

### Running Scripts

```bash
# Run a script
python scripts/your_script.py

# Run with module syntax
python -m pytorch_lab.examples.train_mnist

# Run tests (if installed with [dev])
pytest

# Run Jupyter notebook
jupyter notebook examples/
```

### Training Models

```bash
# Basic training
python -m pytorch_lab.training.train --config config.yaml

# Monitor with TensorBoard
tensorboard --logdir=runs/

# Track with W&B (if installed)
python -m pytorch_lab.training.train --use-wandb
```

### Ending Work

```bash
# Deactivate environment
mamba deactivate
```

## Project Structure

```
pytorch-lab/
├── pytorch_lab/          # Main package (installed in editable mode)
│   ├── core/             # Core utilities and base classes
│   ├── models/           # Neural network architectures
│   ├── layers/           # Custom layers
│   ├── losses/           # Loss functions
│   ├── training/         # Training loops and utilities
│   ├── data/             # Data loaders and preprocessing
│   └── utils/            # Helper functions
├── examples/             # Jupyter notebooks and tutorials
├── scripts/              # Utility scripts
├── apps/                 # Applications (e.g., demos)
├── docs/                 # Project documentation
├── dev/                  # Development notes (gitignored)
└── pyproject.toml        # Project configuration
```

## Common Tasks

### Import the Package

```python
# Import the main package
import pytorch_lab

# Import specific modules
from pytorch_lab.models import ResNet18
from pytorch_lab.training import Trainer
from pytorch_lab.data import DataLoader
```

### Run Jupyter Notebook

```bash
# Start Jupyter (if ipykernel is installed)
jupyter notebook

# Or use VS Code's built-in Jupyter support
```

### Format Code

```bash
# Format with black (if installed with [dev])
black pytorch_lab/

# Lint with ruff
ruff check pytorch_lab/

# Type check with mypy
mypy pytorch_lab/
```

### Run Tests

```bash
# Run all tests (if pytest is installed with [dev])
pytest

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=pytorch_lab
```

## Key Concepts

### Editable Mode (`pip install -e .`)

- Creates a **symbolic link** from `site-packages` → your source code
- Changes to code are **immediately visible** (no reinstall needed)
- Package behaves like it's installed, but uses your local files

### Why Mamba + pip?

- **Mamba**: For compiled libraries (PyTorch, numpy, matplotlib)
  - Better dependency resolution
  - Handles binary dependencies
  - Faster than conda
  - Essential for PyTorch with CUDA/MPS
  
- **pip**: For pure-Python packages (pydantic, typer, rich)
  - Often more up-to-date
  - Simpler for packages without compiled extensions

### Device Management

```python
import torch

# Automatic device selection
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

# Move model and data to device
model = model.to(device)
data = data.to(device)
```

## Troubleshooting

### Can't import pytorch_lab

```bash
# Check if environment is activated
which python  # Should show path to pytorch-lab environment

# Verify installation
pip list | grep pytorch-lab

# Check import path
python -c "import pytorch_lab; print(pytorch_lab.__file__)"
```

### Changes not reflected

If using editable mode and changes aren't visible:

```python
# In Python/Jupyter, reload the module
import importlib
import pytorch_lab.models
importlib.reload(pytorch_lab.models)
```

### PyTorch not using GPU

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check MPS (Apple Silicon) availability
python -c "import torch; print(torch.backends.mps.is_available())"

# List available devices
python -c "import torch; print(torch.cuda.device_count())"
```

### Environment issues

```bash
# List all environments
mamba env list

# Recreate environment if needed
mamba env remove -n pytorch-lab
# Then follow installation steps again
```

## Next Steps

- Read the [Installation Guide](installation.md) for detailed explanations
- Check [PyTorch Setup](pytorch-setup.md) for GPU configuration
- Explore examples in `examples/`
- Review the main [README](../README.md)
