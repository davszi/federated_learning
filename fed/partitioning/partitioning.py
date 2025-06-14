from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    Partitioner,
    IidPartitioner,
    DirichletPartitioner,
    PathologicalPartitioner,
    ShardPartitioner,
    SizePartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)

from fed.partitioning.complete_pathological_partitioner import CompletePathologicalPartitioner

# Define a partitioning factory to easily switch between strategies
PARTITIONING_STRATEGIES = {
    "iid": IidPartitioner,
    "dirichlet": DirichletPartitioner,
    "pathological": PathologicalPartitioner,
    "shard": ShardPartitioner,
    "size": SizePartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}


def get_partitioner(
    strategy: str,
    num_partitions: int,
    seed: int = None,
    **kwargs
) -> Partitioner:
    """Get a partitioner based on the strategy name.

    Args:
        strategy: The partitioning strategy to use
        num_partitions: Number of partitions to create
        seed: Random seed for reproducibility
        **kwargs: Additional arguments for specific partitioners

    Returns:
        A partitioner instance

    Raises:
        ValueError: If the strategy is not supported
    """
    if strategy not in PARTITIONING_STRATEGIES:
        raise ValueError(
            f"Unknown partitioning strategy: {strategy}. "
            f"Available strategies: {list(PARTITIONING_STRATEGIES.keys())}"
        )

    # Set default partition_by for label-based strategies if not specified
    partition_by = kwargs.get("partition_by", "label")

    # Special handling for strategies that need specific parameters
    if strategy == "dirichlet":
        # Default concentration parameter if not provided
        alpha = kwargs.get("alpha", 0.5)
        return DirichletPartitioner(num_partitions=num_partitions,
                                    alpha=alpha,
                                    partition_by=partition_by,
                                    seed=seed)

    elif strategy == "pathological":
        print(f"Using PathologicalPartitioner with: {kwargs}")
        return CompletePathologicalPartitioner(
            num_partitions=num_partitions,
            num_classes_per_partition=kwargs.get("num_classes_per_partition", 4),
            partition_by=partition_by,
            class_assignment_mode=kwargs.get("class_assignment_mode", "random"),
            shuffle=kwargs.get("shuffle", True),
            complete_mode= kwargs.get("complete_mode", False),
            seed=seed,
        )

    elif strategy == "shard":
        return ShardPartitioner(num_partitions=num_partitions, partition_by=partition_by, seed=seed)

    elif strategy == "size":
        # Require partition_sizes parameter
        partition_sizes = kwargs.get("partition_sizes")
        if partition_sizes is None:
            # Default to equal sizes if not provided
            partition_sizes = [1.0 / num_partitions] * num_partitions
        return SizePartitioner(partition_sizes=partition_sizes)

    # For simpler partitioners that just need num_partitions
    return PARTITIONING_STRATEGIES[strategy](num_partitions=num_partitions)


def load_partitioned_dataset(
        dataset_name: str,
        strategy: str,
        num_partitions: int,
        seed: int,
        **kwargs
) -> FederatedDataset:
    """Load a dataset with the specified partitioning strategy.

    Args:
        dataset_name: Name of the dataset to load
        strategy: Partitioning strategy to use
        num_partitions: Number of partitions to create
        seed: Random seed for reproducibility
        **kwargs: Additional arguments for specific partitioners

    Returns:
        A FederatedDataset with the specified partitioning
    """
    partitioner = get_partitioner(strategy, num_partitions, seed, **kwargs)

    return FederatedDataset(
        dataset=dataset_name,
        partitioners={"train": partitioner},
    )