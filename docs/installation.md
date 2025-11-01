# Installation Guide

This guide covers the installation process for the PyTorch Lab project using mamba for environment management.

## Prerequisites

- [Miniforge](https://github.com/conda-forge/miniforge) or [Mambaforge](https://github.com/conda-forge/miniforge) installed
- Python 3.9 or higher
- ~5-10 GB disk space for PyTorch and dependencies

## Installation Steps

### Option A: Quick Install (Using environment.yml)

Create the environment from the provided configuration file:

```bash
# From the project root directory
mamba env create -f environment.yml

# Activate the environment
mamba activate pytorch-lab

# Install pytorch-lab package in editable mode
pip install -e .
```

This installs all dependencies at once. Skip to [Verification](#verification) section.

### Option B: Step-by-Step Install (Recommended for customization)

#### 1. Create the Environment

Create a new mamba environment with Python 3.11:

```bash
mamba create -n pytorch-lab python=3.11 pip -y
```

### 2. Activate the Environment

```bash
mamba activate pytorch-lab
```

### 3. Install PyTorch

**For Mac (CPU or Apple Silicon with MPS):**
```bash
mamba install -c pytorch pytorch torchvision -y
```

**For Linux/Windows with CUDA 11.8:**
```bash
mamba install -c pytorch pytorch torchvision pytorch-cuda=11.8 -y
```

**For Linux/Windows with CUDA 12.1:**
```bash
mamba install -c pytorch pytorch torchvision pytorch-cuda=12.1 -y
```

**Why mamba for PyTorch?**
- Better binary dependency resolution
- Handles CUDA dependencies automatically
- Significantly faster than pip for large packages
- Essential for proper GPU support

### 4. Install Other Compiled Dependencies

Install packages with C/C++ extensions via mamba:

```bash
mamba install -c conda-forge \
  numpy pandas matplotlib seaborn scikit-learn -y
```

**Why mamba for these packages?**
- Packages like numpy and matplotlib have compiled extensions
- Mamba handles binary dependencies better than pip
- Avoids compilation errors, especially on Apple Silicon

### 5. Install Pure-Python Dependencies

Install pure-Python packages via pip:

```bash
pip install pydantic typer rich tqdm
```

### 6. Install pytorch-lab in Editable Mode

Navigate to the project directory and install in editable mode:

```bash
cd /Users/pleiadian53/work/pytorch-lab
pip install -e .
```

**What does `-e` (editable mode) do?**
- Creates a symbolic link from your environment's `site-packages` to your local source code
- Code changes are immediately available without reinstalling
- Allows you to import the package from anywhere: `from pytorch_lab.models import ...`

### 7. Install Development Tools (Optional)

Install development dependencies (pytest, black, ruff, jupyter):

```bash
pip install -e ".[dev]"
```

### 8. Install Optional Components

**Computer Vision Tools:**
```bash
pip install -e ".[vision]"
# Installs: pillow, opencv-python
```

**NLP Tools:**
```bash
pip install -e ".[nlp]"
# Installs: transformers, tokenizers
```

**Experiment Tracking:**
```bash
pip install -e ".[experiment]"
# Installs: tensorboard, wandb
```

**Everything:**
```bash
pip install -e ".[all]"
# Installs all optional dependencies
```

## Verification

Verify the installation by importing the package:

```bash
python -c "import pytorch_lab; print('✓ Success!')"
```

Check PyTorch installation:

```bash
# Check PyTorch version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check CUDA availability (for GPU systems)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check MPS availability (for Apple Silicon Macs)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Check that all core dependencies are available:

```bash
python -c "import torch, torchvision, numpy, pandas, matplotlib, seaborn; print('✓ All dependencies loaded!')"
```

## Package Location

With editable mode, your package is installed as a link:

```bash
python -c "import pytorch_lab; print(pytorch_lab.__file__)"
# Output: /Users/pleiadian53/work/pytorch-lab/pytorch_lab/__init__.py
```

This confirms that Python is using your local source code, not a copy in `site-packages`.

## Troubleshooting

### Multiple Top-Level Packages Error

If you encounter an error about multiple top-level packages during `pip install -e .`, ensure your `pyproject.toml` includes:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["pytorch_lab*"]
exclude = ["dev*", "apps*", "scripts*", "examples*", "docs*"]
```

This explicitly tells setuptools to only include the `pytorch_lab` package and exclude other directories.

### Environment Not Found

If `mamba activate pytorch-lab` fails, ensure:
1. Miniforge/Mambaforge is properly installed
2. Your shell is configured for conda/mamba (run `mamba init`)
3. Restart your terminal after initialization

### PyTorch Not Using GPU

**For CUDA systems:**
```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall with correct CUDA version
mamba install -c pytorch pytorch torchvision pytorch-cuda=11.8 -y
```

**For Apple Silicon (M1/M2/M3):**
```bash
# Verify MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"

# MPS should work with default PyTorch installation on Mac
```

### Import Errors

If you can't import pytorch_lab:

```bash
# Ensure environment is activated
which python  # Should show pytorch-lab environment

# Reinstall in editable mode
cd /Users/pleiadian53/work/pytorch-lab
pip install -e .
```

## Updating Dependencies

To update dependencies in the future:

```bash
# Update mamba packages
mamba update -c pytorch pytorch torchvision
mamba update -c conda-forge numpy pandas matplotlib seaborn scikit-learn

# Update pip packages
pip install --upgrade pydantic typer rich tqdm

# Reinstall the package (if dependencies changed)
pip install -e .
```

## Deactivating the Environment

When you're done working:

```bash
mamba deactivate
```

## Removing the Environment

To completely remove the environment:

```bash
mamba env remove -n pytorch-lab
```

## Next Steps

- Read the [Quick Start Guide](quick-start.md) for daily workflow
- Check [Environment Setup Guide](environment-setup-guide.md) for detailed explanations
- Review GPU setup in [PyTorch Setup](pytorch-setup.md)
- Explore examples in `../examples/`
