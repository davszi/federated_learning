"""Script to automatically visualize data distribution across partitions for all strategies."""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from flwr_datasets import FederatedDataset
from fed.partitioning import get_partitioner


def plot_label_distribution(
    dataset: str,
    strategy: str,
    num_partitions: int,
    file_name: str,
    **kwargs
):
    """Plot the distribution of labels across partitions for a given strategy.

    Args:
        dataset: Name of the dataset
        strategy: Partitioning strategy name
        num_partitions: Number of partitions to create
        **kwargs: Additional parameters for the partitioner
    """
    # Always set partition_by to "label" for strategies that need it
    if strategy in ["dirichlet", "pathological", "shard"] and "partition_by" not in kwargs:
        kwargs["partition_by"] = "label"

    # Get the partitioner
    partitioner = get_partitioner(strategy, num_partitions, **kwargs)

    # Create the federated dataset
    fds = FederatedDataset(
        dataset=dataset,
        partitioners={"train": partitioner},
    )

    # Track label distribution for each partition
    partition_label_counts = []
    partition_sizes = []
    num_classes = 10  # Assuming CIFAR-10

    for i in range(num_partitions):
        partition = fds.load_partition(i)
        partition_sizes.append(len(partition))

        # Directly access the "label" key
        label_counts = Counter(partition["label"])

        # Convert to a normalized distribution
        distribution = np.zeros(num_classes)
        for label, count in label_counts.items():
            distribution[label] = count

        # Normalize to percentages
        if distribution.sum() > 0:
            distribution = (distribution / distribution.sum()) * 100

        partition_label_counts.append(distribution)

    # Plot the distribution
    plt.figure(figsize=(12, 8))

    # Heatmap for label distribution
    plt.subplot(2, 1, 1)
    heatmap = plt.imshow(
        np.array(partition_label_counts),
        aspect='auto',
        cmap='viridis'
    )
    plt.colorbar(heatmap, label='Percentage of samples (%)')
    plt.xlabel('Class Label')
    plt.ylabel('Partition ID')
    plt.title(f'Label Distribution Across Partitions ({strategy})')
    plt.xticks(range(num_classes))
    plt.yticks(range(num_partitions), range(num_partitions))

    # Bar chart for partition sizes
    plt.subplot(2, 1, 2)
    plt.bar(range(num_partitions), partition_sizes)
    plt.xlabel('Partition ID')
    plt.ylabel('Number of Samples')
    plt.title(f'Partition Sizes ({strategy})')
    plt.xticks(range(num_partitions), range(num_partitions))

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()  # Close the figure to free memory

    print(f"Visualization saved to {file_name}")


def generate_all_visualizations(dataset='uoft-cs/cifar10', num_partitions=10):
    """Generate visualizations for all supported partitioning strategies.

    Args:
        dataset: Name of the dataset to use
        num_partitions: Number of partitions to create
    """
    # Define all strategies with their required parameters
    strategies = [
        {
            "name": "iid",
            "params": {}
        },
        {
            "name": "dirichlet",
            "params": {"alpha": 0.5}
        },
        {
            "name": "dirichlet",
            "params": {"alpha": 0.1},
            "suffix": "skewed"  # More skewed distribution
        },
        {
            "name": "pathological",
            "params": {"num_classes_per_partition": 2}
        },
        {
            "name": "pathological",
            "params": {"num_classes_per_partition": 1},
            "suffix": "extreme"
        },
        {
            "name": "shard",
            "params": {"shards_per_partition": 2}
        },
        {
            "name": "linear",
            "params": {}
        },
        {
            "name": "size",
            "params": {"partition_sizes": [0.5, 0.3, 0.2]}
        },
        {
            "name": "square",
            "params": {}
        },
        {
            "name": "exponential",
            "params": {}
        }
    ]

    for strategy_config in strategies:
        strategy_name = strategy_config["name"]
        params = strategy_config["params"]

        # Add suffix to strategy name if provided
        if "suffix" in strategy_config:
            strategy_display = f"{strategy_name}_{strategy_config['suffix']}"
        else:
            strategy_display = strategy_name

        print(f"\n{'='*50}")
        print(f"Generating visualization for {strategy_display} strategy")
        print(f"{'='*50}\n")

        try:
            plot_label_distribution(
                dataset=dataset,
                strategy=strategy_name,  # Use base strategy name for the function
                num_partitions=num_partitions,
                file_name=f'partition_distribution_{strategy_display}.png',
                **params
            )

        except Exception as e:
            print(f"Error generating visualization for {strategy_display}: {e}")

    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    # Call the function to generate all visualizations
    print("Starting automatic generation of federated data partitioning visualizations")
    generate_all_visualizations()
    print("Completed all visualizations!")