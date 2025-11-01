
# pytorch-lab

A modular, hands-on laboratory for **learning PyTorch** and building **reusable deep learning modules**.

This repo is designed to be both a learning playground and a collection of production-ready components for various deep learning tasks including computer vision, NLP, time series, and custom architectures.

## Features

- 🧱 **Modular components**: Reusable layers, losses, optimizers, and training loops
- 🎓 **Learning-focused**: Clear examples and experiments for understanding PyTorch concepts
- 📊 **Experiment tracking**: Built-in support for TensorBoard and W&B
- 🔬 **Research-ready**: Easy prototyping with clean abstractions
- 📈 **Visualization**: Rich plotting and diagnostics for model analysis

## Install

```bash
# Recommended: editable install with dev tools
pip install -e ".[dev]"

# Or minimal install
pip install -e .

# With specific extras
pip install -e ".[vision,experiment]"
```

## Quickstart

```bash
# 1) Run a simple training example
python -m pytorch_lab.examples.train_mnist

# 2) Launch Jupyter for interactive learning
jupyter notebook examples/

# 3) Start TensorBoard to visualize training
tensorboard --logdir=runs/
```

## Project Layout

```
pytorch-lab/
 ├─ pytorch_lab/           # Main package
 │   ├─ __init__.py
 │   ├─ core/              # Core utilities and base classes
 │   ├─ models/            # Neural network architectures
 │   ├─ layers/            # Custom layers and modules
 │   ├─ losses/            # Loss functions
 │   ├─ training/          # Training loops and utilities
 │   ├─ data/              # Data loaders and preprocessing
 │   ├─ tensor_ops/        # Tensor operations (masking, preprocessing, feature engineering)
 │   └─ utils/             # Helper functions
 ├─ examples/              # Jupyter notebooks and example scripts
 ├─ scripts/               # Utility scripts
 ├─ apps/                  # Applications (e.g., Streamlit demos)
 ├─ docs/                  # Documentation
 ├─ pyproject.toml         # Project configuration
 └─ README.md
```

## Learning Path

### Current Tutorials

1. **Tensor Fundamentals** - `examples/C1_M1_Lab_3_tensors.ipynb` - Introduction to PyTorch tensors
2. **Practical Tensor Operations** - `examples/tensor_operations_practical_guide.ipynb` - Real-world tensor operations for deep learning

### Planned Tutorials

3. **Autograd & Backprop** - Understanding automatic differentiation
4. **Building Models** - Creating neural network architectures
5. **Training Loops** - Training, validation, and testing
6. **Custom Layers** - Implementing custom modules

## Module Organization

### Core Modules

- `pytorch_lab.core` - Base classes, configs, registry pattern
- `pytorch_lab.models` - Standard architectures (ResNet, Transformer, etc.)
- `pytorch_lab.layers` - Reusable layer implementations
- `pytorch_lab.losses` - Custom loss functions
- `pytorch_lab.training` - Trainers, callbacks, checkpointing
- `pytorch_lab.data` - Dataset utilities and transforms
- `pytorch_lab.tensor_ops` - **NEW!** Tensor operations (masking, preprocessing, feature engineering, batch ops)

### Optional Components
- `pytorch_lab.vision` - Computer vision specific tools
- `pytorch_lab.nlp` - NLP utilities and models
- `pytorch_lab.timeseries` - Time series models

## Development Setup

See [SETUP_REPLICATION.md](SETUP_REPLICATION.md) for complete setup instructions.

Quick version:
```bash
# Create environment
mamba create -n pytorch-lab python=3.11 -y
mamba activate pytorch-lab

# Install PyTorch (adjust for your CUDA version)
mamba install pytorch torchvision -c pytorch -y

# Install other dependencies
mamba install numpy pandas matplotlib seaborn scikit-learn -c conda-forge -y
pip install pydantic typer rich

# Install package in editable mode
pip install -e ".[dev]"

# Verify
python -c "import pytorch_lab; print('✓ Success!')"
```

## Contributing

This is a personal learning repo, but structured for potential collaboration. Feel free to:
- Add new examples and tutorials
- Implement additional model architectures
- Improve documentation
- Report issues or suggest enhancements

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## License

MIT License - feel free to use for learning and projects.
