# Setup Replication Guide

**Quick reference for setting up this project on a new computer.**

---

## Prerequisites

1. Install [Miniforge](https://github.com/conda-forge/miniforge) (provides mamba)
2. Clone this repository
3. Have ~5-10 GB disk space available (for PyTorch and dependencies)

---

## Complete Setup (5-15 minutes)

```bash
# 1. Navigate to project
cd /path/to/pytorch-lab

# 2. Create environment
mamba create -n pytorch-lab python=3.11 pip -y

# 3. Activate environment
mamba activate pytorch-lab

# 4. Install PyTorch (CPU version for M1/M2 Macs or CPU-only machines)
mamba install -c pytorch pytorch torchvision -y

# For CUDA-enabled systems, use:
# mamba install -c pytorch pytorch torchvision pytorch-cuda=11.8 -y

# 5. Install other compiled dependencies (via mamba)
mamba install -c conda-forge \
  numpy pandas matplotlib seaborn scikit-learn -y

# 6. Install pure-Python dependencies (via pip)
pip install pydantic typer rich tqdm

# 7. Install this package in editable mode
pip install -e .

# 8. Install development tools (optional but recommended)
pip install -e ".[dev]"

# 9. Verify installation
python -c "import pytorch_lab; print('✓ Success!')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## What is Editable Mode?

**Command**: `pip install -e .`

**What it does**: Creates a symbolic link from your environment's `site-packages` to your source code.

**Why it matters**: 
- ✅ Code changes are **immediately available** (no reinstall needed)
- ✅ Package is **importable from anywhere**: `from pytorch_lab import ...`
- ✅ Works with **Jupyter notebooks** automatically
- ✅ **Tests use latest code** automatically

**How it works**:
```
Your Source Code                    site-packages
────────────────                    ─────────────
/path/to/pytorch-lab/               /env/lib/python3.11/site-packages/
├── pytorch_lab/  ← SOURCE          ├── pytorch-lab.egg-link
│   ├── __init__.py                 │   (points to your source)
│   └── ...                         └── (Python imports from YOUR source!)
```

---

## Daily Workflow

```bash
# Start work
mamba activate pytorch-lab
cd /path/to/pytorch-lab

# Edit code
vim pytorch_lab/models/my_model.py

# Test immediately (no reinstall!)
python -c "from pytorch_lab.models import MyModel"
# OR
pytest tests/
# OR
jupyter notebook examples/
```

---

## Why This Approach?

### Hybrid Package Management

**Mamba for compiled libraries**:
- PyTorch, torchvision, numpy, pandas, matplotlib, scikit-learn
- Better binary dependency resolution
- Especially important for PyTorch with CUDA support
- Faster and more reliable on Apple Silicon Macs

**pip for pure-Python packages**:
- pydantic, typer, rich, tqdm
- Often more up-to-date on PyPI
- Simpler dependency chains

**pip with `-e` for our package**:
- Enables active development workflow
- Changes immediately available

### Package Configuration

In `pyproject.toml`:
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["pytorch_lab*"]
exclude = ["dev*", "apps*", "scripts*", "examples*", "docs*"]
```

This ensures only `pytorch_lab/` is installed as a package.

---

## Optional Dependencies

### Computer Vision Tools
```bash
pip install -e ".[vision]"
# Installs: pillow, opencv-python
```

### NLP Tools
```bash
pip install -e ".[nlp]"
# Installs: transformers, tokenizers
```

### Experiment Tracking
```bash
pip install -e ".[experiment]"
# Installs: tensorboard, wandb
```

### Everything
```bash
pip install -e ".[all]"
# Installs all optional dependencies
```

---

## Verification

```bash
# Check environment
which python
# Should show: .../miniforge3/envs/pytorch-lab/bin/python

# Check package location (should be your source directory!)
python -c "import pytorch_lab; print(pytorch_lab.__file__)"
# Should show: /path/to/pytorch-lab/pytorch_lab/__init__.py

# Check PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS (Apple Silicon): {torch.backends.mps.is_available()}')"

# Check all core dependencies
python -c "import torch, torchvision, numpy, pandas, matplotlib, seaborn; print('✓ All loaded')"
```

---

## Troubleshooting

### Can't import pytorch_lab
```bash
mamba activate pytorch-lab
cd /path/to/pytorch-lab
pip install -e .
```

### Changes not reflected
```python
# In Python/Jupyter
import importlib
import pytorch_lab.your_module
importlib.reload(pytorch_lab.your_module)
# Or restart Python/Jupyter kernel
```

### PyTorch not using GPU
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with CUDA support
mamba install -c pytorch pytorch torchvision pytorch-cuda=11.8 -y
```

### Apple Silicon (M1/M2) Issues
```bash
# Use MPS (Metal Performance Shaders) backend
python -c "import torch; print(torch.backends.mps.is_available())"

# In your code:
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

---

## Complete Documentation

For detailed explanations, see:

- **[docs/environment-setup-guide.md](docs/environment-setup-guide.md)** - Complete step-by-step guide
- **[docs/installation.md](docs/installation.md)** - Detailed installation guide
- **[docs/quick-start.md](docs/quick-start.md)** - Quick reference
- **[docs/README.md](docs/README.md)** - Documentation index

---

## Key Files

- `pyproject.toml` - Project configuration and dependencies
- `.gitignore` - Excludes `dev/`, model checkpoints, and Python artifacts
- `requirements.txt` - Core dependency list
- `environment.yml` - Conda/Mamba environment specification
- `docs/` - All documentation

---

## Project Structure

```
pytorch-lab/
├── pytorch_lab/          # Main package (installed in editable mode)
│   ├── core/             # Core utilities and base classes
│   ├── models/           # Neural network architectures
│   ├── layers/           # Custom layers
│   ├── losses/           # Loss functions
│   ├── training/         # Training utilities
│   ├── data/             # Data loaders
│   └── utils/            # Helper functions
├── scripts/              # Utility scripts (NOT installed)
├── apps/                 # Applications (NOT installed)
├── examples/             # Jupyter notebooks (NOT installed)
├── docs/                 # Documentation (NOT installed)
├── dev/                  # Dev notes (gitignored, NOT installed)
└── pyproject.toml        # Configuration
```

Only `pytorch_lab/` is installed. Everything else stays in your project directory.

---

## Summary

**The key insight**: `pip install -e .` creates a link (not a copy) from `site-packages` to your source code, making development seamless.

**Result**: Edit code → Test immediately → No reinstall needed!

This is the standard professional workflow for Python package development.
