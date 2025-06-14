from typing import Literal, Optional, Any

from flwr_datasets.partitioner import PathologicalPartitioner

class CompletePathologicalPartitioner(PathologicalPartitioner):
    """Like PathologicalPartitioner, but can ensure every label is in ≥1 partition."""
    def __init__(
            self,
            num_partitions: int,
            partition_by: str,
            num_classes_per_partition: int,
            class_assignment_mode: Literal["random", "deterministic", "first-deterministic"] = "random",
            shuffle: bool = True,
            seed: Optional[int] = 42,
            complete_mode: bool = False,
    ) -> None:
        """Initialize the complete pathological partitioner.

        Parameters
        ----------
        num_partitions
            Number of partitions (clients) to create.
        partition_by
            Name of the dataset field to use as the label.
        num_classes_per_partition
            Exact number of distinct classes each partition should hold.
        class_assignment_mode
            How to pick the initial k classes per partition:
            - "random": pick k unique labels at random
            - "deterministic": use (partition_id + i) % num_labels
            - "first-deterministic": fix first label deterministically, sample the rest
        shuffle
            Whether to shuffle examples within each partition.
        seed
            Random seed for reproducibility.
        complete_mode
            If True, after the base assignment, any label not yet assigned will be
            swapped in by evicting a “redundant” label (one present in >1 partitions),
            preserving exactly k classes per partition while guaranteeing full coverage.

        Algorithm
        ---------
        1. **Base assignment**: Invoke the parent (`PathologicalPartitioner`) logic to
           select exactly k classes per partition according to `class_assignment_mode`.
        2. **Feasibility check**: Ensure total slots (p × k) ≥ total labels; else error.
        3. **Collect unused labels**: Identify labels never assigned in step 1.
        4. **Swap loop**: For each missing label:
           a. Pick a label with usage >1 (redundant) at random.
           b. Choose one of its partitions and swap it out for the new label.
           c. Update usage counts and reverse maps.
        5. **Result**: Each partition still has k classes, and every class appears at least once.

        NOTE: Partition sample sizes can be affected by this process!!!
        """

        super().__init__(
            num_partitions,
            partition_by,
            num_classes_per_partition,
            class_assignment_mode,
            shuffle,
            seed,
        )
        self.complete_mode = complete_mode

    def _determine_partition_id_to_unique_labels(self) -> None:
        # 1) run base assignment (exactly k per partition, may leave some unused)
        super()._determine_partition_id_to_unique_labels()

        if not self.complete_mode:
            return

        # 2) feasibility check
        num_labels = len(self._unique_labels)
        total_slots = self._num_partitions * self._num_classes_per_partition
        if total_slots < num_labels:
            raise ValueError(
                f"Impossible to cover {num_labels} classes with "
                f"{total_slots} slots (p={self._num_partitions}×k={self._num_classes_per_partition})."
            )

        # 3) compute which labels are used/unused
        used = {
            lbl
            for labels in self._partition_id_to_unique_labels.values()
            for lbl in labels
        }
        unused = [lbl for lbl in self._unique_labels if lbl not in used]
        if not unused:
            return

        # 4) build reverse‐map and usage counts
        label_to_pids: dict[Any, list[int]] = {}
        usage_count: dict[Any, int] = {}
        for pid, labels in self._partition_id_to_unique_labels.items():
            for lbl in labels:
                label_to_pids.setdefault(lbl, []).append(pid)
                usage_count[lbl] = usage_count.get(lbl, 0) + 1

        # 5) for each label missing, swap into a partition that holds a redundant label
        for new_lbl in unused:
            # find any label with count >1
            candidates = [lbl for lbl, cnt in usage_count.items() if cnt > 1]
            if not candidates:
                # shouldn't happen if total_slots >= num_labels
                candidates = list(usage_count.keys())
            evict_lbl = self._rng.choice(candidates)
            # pick one partition that has evict_lbl
            pid = self._rng.choice(label_to_pids[evict_lbl])
            labels = self._partition_id_to_unique_labels[pid]

            # swap
            idx = labels.index(evict_lbl)
            labels[idx] = new_lbl

            # update maps
            # remove pid from old label
            label_to_pids[evict_lbl].remove(pid)
            usage_count[evict_lbl] -= 1
            if usage_count[evict_lbl] == 0:
                del usage_count[evict_lbl]
                del label_to_pids[evict_lbl]

            # add new_lbl
            label_to_pids.setdefault(new_lbl, []).append(pid)
            usage_count[new_lbl] = usage_count.get(new_lbl, 0) + 1