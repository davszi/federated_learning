# Federated Learning with Flower and PyTorch

This project demonstrates how to implement federated learning using Flower and PyTorch with different data partitioning strategies.

## Features

- CIFAR-10 image classification using a VGG-style CNN model
- Multiple data partitioning strategies
- Configurable federated learning parameters
- Centralized server-side evaluation

## Data Partitioning Strategies

The project supports the following partitioning strategies:

1. **IID (Independent and Identically Distributed)** - Default strategy where data is randomly distributed across clients.
2. **Dirichlet** - Non-IID partitioning with controllable heterogeneity using a Dirichlet distribution.
3. **Pathological** - Each client gets data from a limited number of classes.
4. **Linear** - Partition sizes are linearly correlated with client ID.
5. **Square** - Partition sizes are correlated with the square of client ID.
6. **Exponential** - Partition sizes are correlated with the exponential of client ID.
7. **Shard** - Each partition contains data from specific classes.
8. **Size** - Custom partition sizes defined by the user.

## Configuration

You can configure the partitioning strategy and other parameters in `pyproject.toml`:

```toml
[tool.flwr.app.config]
# Server configuration
num-server-rounds = 100
fraction-fit = 0.3

# Client configuration
local-epochs = 7

# Data partitioning configuration
partitioning-strategy = "iid"
dataset = "uoft-cs/cifar10"

# Optional partitioning parameters (uncomment as needed)
# For Dirichlet partitioning
# dirichlet-alpha = 0.5

# For Pathological partitioning  
# classes-per-partition = 2
```

## Adding New Partitioning Strategies

To add a new partitioning strategy:

1. Update `fed/partitioning.py` to include the new strategy
2. Add the appropriate parameters to `pyproject.toml`
## Performance Comparison

Different partitioning strategies can significantly impact federated learning performance:

To be added ...

- **IID**: Typically provides the best convergence rate and final accuracy
- **Dirichlet**: Performance depends on the α parameter (smaller values = more heterogeneity = slower convergence)
- **Pathological**: Most challenging for federated learning algorithms due to extreme label skew