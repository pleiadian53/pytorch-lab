# Environment Setup Guide

Complete, step-by-step guide for setting up the PyTorch Lab development environment on a new computer.

## Overview

This guide will walk you through:
1. Installing Miniforge (mamba package manager)
2. Creating a dedicated Python environment
3. Installing PyTorch and dependencies
4. Installing the pytorch-lab package in editable mode
5. Verifying the installation

**Estimated time:** 10-20 minutes (depending on internet speed)

---

## Step 1: Install Miniforge

Miniforge provides `mamba`, a faster alternative to `conda` that uses conda-forge by default.

### macOS (Intel or Apple Silicon)

```bash
# Download and install
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# Follow the prompts, answer 'yes' to initialize
# Restart your terminal
```

### Linux

```bash
# Download and install
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# Follow the prompts, answer 'yes' to initialize
# Restart your terminal
```

### Windows

Download and run the installer from: https://github.com/conda-forge/miniforge

### Verify Installation

```bash
mamba --version
# Should show mamba version
```

---

## Step 2: Create the Environment

Create a new isolated Python environment for PyTorch Lab:

```bash
mamba create -n pytorch-lab python=3.11 pip -y
```

**What this does:**
- Creates an environment named `pytorch-lab`
- Installs Python 3.11
- Includes pip for additional package management
- `-y` automatically confirms the installation

---

## Step 3: Activate the Environment

```bash
mamba activate pytorch-lab
```

**Verify activation:**
```bash
which python
# Should show: .../miniforge3/envs/pytorch-lab/bin/python
```

---

## Step 4: Install PyTorch

Choose the appropriate command for your system:

### macOS (CPU or Apple Silicon with MPS)

```bash
mamba install -c pytorch pytorch torchvision -y
```

### Linux/Windows with NVIDIA GPU (CUDA 11.8)

```bash
mamba install -c pytorch pytorch torchvision pytorch-cuda=11.8 -y
```

### Linux/Windows with NVIDIA GPU (CUDA 12.1)

```bash
mamba install -c pytorch pytorch torchvision pytorch-cuda=12.1 -y
```

**Verify PyTorch:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

---

## Step 5: Install Other Dependencies

### Compiled Libraries (via mamba)

```bash
mamba install -c conda-forge \
  numpy pandas matplotlib seaborn scikit-learn -y
```

**Why mamba?**
- These packages have C/C++ compiled extensions
- Mamba provides pre-built binaries
- Faster and more reliable than building from source

### Pure-Python Libraries (via pip)

```bash
pip install pydantic typer rich tqdm
```

**Why pip?**
- These are pure-Python packages
- Often more up-to-date on PyPI
- No binary compatibility issues

---

## Step 6: Install pytorch-lab in Editable Mode

Navigate to the project directory:

```bash
cd /Users/pleiadian53/work/pytorch-lab
```

Install in editable mode:

```bash
pip install -e .
```

**What is editable mode?**
- Creates a symbolic link from `site-packages` to your source code
- Code changes are immediately available without reinstalling
- Essential for active development

**Verify installation:**
```bash
python -c "import pytorch_lab; print(pytorch_lab.__file__)"
# Should show: /Users/pleiadian53/work/pytorch-lab/pytorch_lab/__init__.py
```

---

## Step 7: Install Development Tools (Optional)

```bash
pip install -e ".[dev]"
```

**Includes:**
- `pytest` - Testing framework
- `black` - Code formatter
- `ruff` - Fast linter
- `mypy` - Type checker
- `ipykernel`, `jupyter` - Interactive development

---

## Step 8: Final Verification

Run all verification checks:

```bash
# Check package import
python -c "import pytorch_lab; print('✓ pytorch_lab installed')"

# Check PyTorch
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"

# Check GPU support
python -c "import torch; print(f'✓ CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'✓ MPS: {torch.backends.mps.is_available()}')"

# Check all dependencies
python -c "import torch, torchvision, numpy, pandas, matplotlib, seaborn, scikit_learn, pydantic, typer, rich, tqdm; print('✓ All dependencies loaded')"
```

---

## Understanding the Setup

### Directory Structure

```
pytorch-lab/
├── pytorch_lab/          # Main package (installed in editable mode)
│   ├── __init__.py       # Makes it a Python package
│   ├── core/             # Core utilities
│   ├── models/           # Neural network architectures
│   └── ...
├── examples/             # Learning notebooks (NOT installed)
├── docs/                 # Documentation (NOT installed)
├── scripts/              # Utility scripts (NOT installed)
├── pyproject.toml        # Project configuration
└── requirements.txt      # Dependency reference
```

Only `pytorch_lab/` is installed as a package. Everything else stays in your project directory.

### How Editable Mode Works

```
Your Source Code                    site-packages
────────────────                    ─────────────
/path/to/pytorch-lab/               /env/lib/python3.11/site-packages/
├── pytorch_lab/  ← SOURCE          ├── pytorch-lab.egg-link
│   ├── __init__.py                 │   (points to your source)
│   └── ...                         └── (Python imports from YOUR source!)
```

When you `import pytorch_lab`, Python follows the link to your source code.

### Hybrid Package Management

We use both mamba and pip strategically:

| Package Manager | Used For | Why |
|-----------------|----------|-----|
| **mamba** | PyTorch, numpy, matplotlib | Compiled libraries, binary dependencies |
| **pip** | pydantic, typer, rich | Pure-Python packages, latest versions |
| **pip -e** | pytorch_lab (our package) | Development mode, immediate code changes |

---

## Daily Workflow

### Starting Work

```bash
# Activate environment
mamba activate pytorch-lab

# Navigate to project
cd /Users/pleiadian53/work/pytorch-lab

# Now you can edit code and changes are immediately available!
```

### Making Changes

```bash
# Edit any file in pytorch_lab/
vim pytorch_lab/models/my_model.py

# Changes are immediately available - no reinstall needed!
python -c "from pytorch_lab.models import my_model"
```

### Running Code

```bash
# Run scripts
python scripts/train.py

# Run as module
python -m pytorch_lab.examples.train_mnist

# Run tests
pytest

# Run Jupyter
jupyter notebook examples/
```

### Ending Work

```bash
# Deactivate environment
mamba deactivate
```

---

## Troubleshooting

### Can't import pytorch_lab

**Problem:** `ModuleNotFoundError: No module named 'pytorch_lab'`

**Solution:**
```bash
# Check if environment is activated
which python  # Should show pytorch-lab environment

# Reinstall in editable mode
cd /Users/pleiadian53/work/pytorch-lab
pip install -e .
```

### Changes not reflected

**Problem:** Code changes don't take effect

**Solution:**
```python
# In Python/Jupyter, reload the module
import importlib
import pytorch_lab.models
importlib.reload(pytorch_lab.models)

# Or restart your Python/Jupyter kernel
```

### PyTorch not using GPU

**Problem:** `torch.cuda.is_available()` returns `False`

**Solution for CUDA:**
```bash
# Check CUDA version on your system
nvidia-smi

# Reinstall PyTorch with matching CUDA version
mamba install -c pytorch pytorch torchvision pytorch-cuda=11.8 -y
```

**Solution for Apple Silicon:**
```bash
# MPS should work automatically with PyTorch 2.0+
python -c "import torch; print(torch.backends.mps.is_available())"

# Use MPS in your code:
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### Multiple top-level packages error

**Problem:** Error during `pip install -e .`

**Solution:** Ensure `pyproject.toml` has:
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["pytorch_lab*"]
exclude = ["dev*", "apps*", "scripts*", "examples*", "docs*"]
```

### Environment activation fails

**Problem:** `mamba activate pytorch-lab` doesn't work

**Solution:**
```bash
# Initialize mamba for your shell
mamba init

# Restart your terminal

# Try again
mamba activate pytorch-lab
```

---

## Next Steps

- Read the [Quick Start Guide](quick-start.md) for daily workflow reference
- Check [Installation Guide](installation.md) for detailed command explanations
- Explore examples in `examples/`
- Review the main [README](../README.md)

---

## Summary

You've successfully set up a professional PyTorch development environment with:

✅ Isolated Python environment  
✅ PyTorch with GPU support (if available)  
✅ All necessary dependencies  
✅ Editable package installation  
✅ Development tools  

**Key insight:** Editable mode (`pip install -e .`) creates a link to your source code, making development seamless. Edit → Test → No reinstall needed!
