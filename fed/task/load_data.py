from typing import Optional

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from fed.partitioning import load_partitioned_dataset


# Global cache for FederatedDataset
_fds_cache = {}  # (dataset_name, strategy, num_partitions) -> FederatedDataset


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_name: str,
    partitioning_strategy: str,
    seed: Optional[int] = None,
    test_size: float = 0.2,
    validate_size: float = 0.0,
    batch_size=20,
    **partitioning_kwargs,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    # Set partition_by for label-based strategies if not provided
    if (
        partitioning_strategy in ["dirichlet", "pathological", "shard"]
        and "partition_by" not in partitioning_kwargs
    ):
        partitioning_kwargs["partition_by"] = "label"

    # Create a cache key based on the dataset and partitioning configuration
    cache_key = (
        dataset_name,
        partitioning_strategy,
        num_partitions,
        str(sorted(partitioning_kwargs.items())),
    )

    # Only initialize FederatedDataset once for a given configuration
    global _fds_cache
    if cache_key not in _fds_cache:
        _fds_cache[cache_key] = load_partitioned_dataset(
            dataset_name=dataset_name,
            strategy=partitioning_strategy,
            num_partitions=num_partitions,
            **partitioning_kwargs,
        )

    fds = _fds_cache[cache_key]

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Load the specific partition
    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    train_test_partition = partition.train_test_split(test_size=test_size, seed=seed)
    train_partition = train_test_partition["train"]
    test_partition = train_test_partition["test"].with_transform(
        apply_transforms
    )

    if validate_size > 0.0:
        train_val_partition = train_partition.train_test_split(
            test_size=validate_size, seed=seed
        )
        train_partition = train_val_partition["train"]
        val_partition = train_val_partition["test"].with_transform(
            apply_transforms
        )
        val_loader = DataLoader(
            val_partition, batch_size=batch_size
        )
    else:
        val_loader = None

    train_partition = train_partition.with_transform(apply_transforms)
    trainloader = DataLoader(
        train_partition, batch_size=batch_size, shuffle=True
    )

    testloader = DataLoader(test_partition, batch_size=batch_size)

    return trainloader, val_loader, testloader