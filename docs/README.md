# Project Documentation

This directory contains project-level documentation for the PyTorch Lab project.

## Quick Start

**New to this project?** Start here:

1. 📦 **[Environment Setup Guide](environment-setup-guide.md)** - Complete setup instructions for new computers
2. ⚡ **[Quick Start](quick-start.md)** - TL;DR for daily workflow
3. 🔧 **[Installation Guide](installation.md)** - Detailed installation with explanations

## Documentation Index

### Setup & Installation

| Document | Purpose | Audience |
|----------|---------|----------|
| **[Environment Setup Guide](environment-setup-guide.md)** | Complete step-by-step setup for new computers | New developers |
| **[Installation Guide](installation.md)** | Detailed installation with explanations | All developers |
| **[Quick Start](quick-start.md)** | Quick reference for daily workflow | Experienced developers |

### PyTorch Specific

| Document | Purpose | Audience |
|----------|---------|----------|
| **[PyTorch Setup](pytorch-setup.md)** | GPU/CUDA/MPS configuration | All developers |
| **[Model Development](model-development.md)** | Best practices for building models | All developers |

### Key Concepts

#### Editable Mode (`pip install -e .`)

The most important concept for active development:

- **What**: Creates symbolic link from `site-packages` to your source code
- **Why**: Changes are immediately available without reinstalling
- **When**: Always use for active development

#### Hybrid Package Management

We use both mamba and pip:

- **Mamba**: For compiled libraries (PyTorch, numpy, matplotlib)
  - Better binary dependency resolution
  - Especially important for PyTorch with CUDA/MPS
  - Faster on Apple Silicon
  
- **pip**: For pure-Python packages (pydantic, typer, rich)
  - Often more up-to-date
  - Simpler for packages without compiled extensions

## Documentation Structure

This project maintains three levels of documentation:

### 1. Project-Level (`/docs`) ← You are here

High-level project documentation:
- Setup and installation guides
- Architecture and design decisions
- Learning resources
- API reference (future)

### 2. Package-Level (`/pytorch_lab/*/docs`)

Module-specific documentation:
- `/pytorch_lab/models/docs` - Model architectures
- `/pytorch_lab/layers/docs` - Custom layer implementations
- `/pytorch_lab/training/docs` - Training utilities
- `/pytorch_lab/data/docs` - Data loading and preprocessing

### 3. Development Notes (`/dev`)

Private development notes (gitignored, not version controlled):
- Learning notes and experiments
- Temporary documentation
- Research ideas
- Not meant for public sharing

## Common Tasks

### First Time Setup

```bash
# See environment-setup-guide.md for complete instructions
mamba create -n pytorch-lab python=3.11 pip -y
mamba activate pytorch-lab
mamba install -c pytorch pytorch torchvision -y
mamba install -c conda-forge numpy pandas matplotlib seaborn scikit-learn -y
pip install pydantic typer rich tqdm
cd /path/to/pytorch-lab
pip install -e .
pip install -e ".[dev]"
```

### Daily Workflow

```bash
# Activate environment
mamba activate pytorch-lab

# Make changes to code
# Changes are immediately available!

# Test
pytest

# Format and lint
black pytorch_lab/
ruff check pytorch_lab/
```

### Verification

```bash
# Check installation
python -c "import pytorch_lab; print(pytorch_lab.__file__)"
# Should show: /path/to/pytorch-lab/pytorch_lab/__init__.py

# Check PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Environment Details

**Environment Name**: `pytorch-lab`  
**Python Version**: 3.11  
**Package Manager**: Mamba (conda-forge + pytorch channel)  
**Installation Mode**: Editable (`pip install -e .`)

### Core Dependencies

- **torch, torchvision** - Deep learning framework
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **matplotlib, seaborn** - Visualization
- **scikit-learn** - Machine learning utilities
- **pydantic** - Data validation
- **typer, rich** - CLI framework

### Development Tools

- **pytest** - Testing framework
- **black** - Code formatter
- **ruff** - Fast linter
- **mypy** - Type checking
- **ipykernel, jupyter** - Interactive development

## Getting Help

### Documentation

1. Check the relevant guide in this directory
2. Check package-level docs in `/pytorch_lab/*/docs`
3. Check the main [README](../README.md)

### Troubleshooting

Common issues and solutions:

**Can't import pytorch_lab**
```bash
mamba activate pytorch-lab
pip install -e .
```

**Changes not reflected**
```python
import importlib
import pytorch_lab.your_module
importlib.reload(pytorch_lab.your_module)
# Or restart Python/Jupyter
```

**PyTorch not using GPU**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# See pytorch-setup.md for GPU configuration
```

### More Help

- See [Environment Setup Guide](environment-setup-guide.md#troubleshooting)
- See [Quick Start](quick-start.md#troubleshooting)
- Check [PyTorch Setup](pytorch-setup.md) for GPU issues

## Learning Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## Contributing

This is a personal learning repo, but contributions are welcome:
- Add new examples and tutorials
- Implement additional architectures
- Improve documentation
- Report issues

## Related Files

- `../pyproject.toml` - Project configuration
- `../requirements.txt` - Dependency list (reference)
- `../.gitignore` - Git ignore rules
- `../README.md` - Project README
