"""
Exact Distributed Query Processing Baseline.
Simulates standard distributed hash-partitioned execution (Apache Spark / MapReduce / Flink):
- Stage 1: Hash partitioning by group key: partition = hash(cell_tower_id) % num_partitions
- Stage 2: Worker-local hash set aggregation for COUNT(DISTINCT) and exact frequency
- Measures:
    - Partition skew (min, max, mean, stddev, coefficient of variation)
    - Straggler task penalty: Job latency = max(partition_time)
    - Peak memory consumption (O(N) exact hash sets)
    - Shuffle network payload (raw rows transferred across network)
"""

import time
import sys
import hashlib
from typing import List, Dict, Any, Tuple
import numpy as np


class ExactDistributedProcessor:
    """
    Simulates exact hash-partitioned distributed query execution.
    """

    def __init__(self, num_partitions: int = 8):
        self.num_partitions = num_partitions

    def _hash_partition(self, key: str) -> int:
        """Standard hash partitioner: hash(key) % W."""
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.num_partitions

    def execute_aggregation(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Executes exact aggregation query:
            SELECT cell_tower_id,
                   COUNT(*) AS total_calls,
                   COUNT(DISTINCT caller_id) AS unique_callers,
                   COUNT(DISTINCT callee_id) AS unique_callees,
                   SUM(duration_sec) AS total_duration
            FROM cdr_records
            GROUP BY cell_tower_id;

        Returns:
            (results_dict, performance_telemetry)
        """
        start_wall_time = time.perf_counter()

        # Step 1: Shuffle Stage (Data partitioning)
        partition_buckets = [[] for _ in range(self.num_partitions)]
        shuffle_bytes_transferred = 0

        for r in records:
            p_idx = self._hash_partition(r["cell_tower_id"])
            partition_buckets[p_idx].append(r)
            # Estimate serialized record size (~80 bytes per CDR row)
            shuffle_bytes_transferred += 80

        # Step 2: Worker Node Execution (simulating parallel workers)
        partition_metrics = []
        final_aggregates = {}

        for p_idx, bucket in enumerate(partition_buckets):
            p_start = time.perf_counter()

            # Worker local aggregation state:
            # tower -> {count, set(callers), set(callees), sum_dur}
            worker_state = {}
            total_elements_in_hashsets = 0

            for row in bucket:
                tower = row["cell_tower_id"]
                caller = row["caller_id"]
                callee = row["callee_id"]
                dur = row["duration_sec"]

                if tower not in worker_state:
                    worker_state[tower] = {
                        "count": 0,
                        "unique_callers": set(),
                        "unique_callees": set(),
                        "duration": 0,
                    }

                worker_state[tower]["count"] += 1
                worker_state[tower]["unique_callers"].add(caller)
                worker_state[tower]["unique_callees"].add(callee)
                worker_state[tower]["duration"] += dur

            # Calculate memory: Python set overhead is ~64 bytes per string reference
            for tower, data in worker_state.items():
                total_elements_in_hashsets += len(data["unique_callers"]) + len(
                    data["unique_callees"]
                )

            p_end = time.perf_counter()
            p_time = p_end - p_start

            # Memory estimate in KB: base overhead + 64 bytes per set item + raw rows buffer
            worker_mem_kb = (
                sys.getsizeof(worker_state)
                + (total_elements_in_hashsets * 64)
                + (len(bucket) * 80)
            ) / 1024.0

            partition_metrics.append(
                {
                    "partition_id": p_idx,
                    "record_count": len(bucket),
                    "execution_time_sec": p_time,
                    "peak_memory_kb": worker_mem_kb,
                }
            )

            # Consolidate results
            for tower, data in worker_state.items():
                final_aggregates[tower] = {
                    "total_calls": data["count"],
                    "distinct_callers": len(data["unique_callers"]),
                    "distinct_callees": len(data["unique_callees"]),
                    "total_duration": data["duration"],
                }

        end_wall_time = time.perf_counter()

        # In distributed computing, the overall stage latency is bounded by the slowest worker (straggler)
        worker_times = [m["execution_time_sec"] for m in partition_metrics]
        record_counts = [m["record_count"] for m in partition_metrics]
        memories = [m["peak_memory_kb"] for m in partition_metrics]

        straggler_time = max(worker_times)
        mean_worker_time = float(np.mean(worker_times)) if worker_times else 0.0
        straggler_ratio = (
            straggler_time / mean_worker_time if mean_worker_time > 0 else 1.0
        )

        std_records = float(np.std(record_counts))
        mean_records = float(np.mean(record_counts))
        coeff_variation = (std_records / mean_records) if mean_records > 0 else 0.0

        telemetry = {
            "mode": "Exact",
            "num_partitions": self.num_partitions,
            "total_records": len(records),
            "total_wall_time_sec": end_wall_time - start_wall_time,
            "effective_distributed_latency_sec": straggler_time,
            "straggler_ratio": straggler_ratio,
            "partition_counts": record_counts,
            "partition_times_sec": worker_times,
            "partition_memory_kb": memories,
            "peak_worker_memory_kb": max(memories),
            "shuffle_bytes": shuffle_bytes_transferred,
            "record_count_stddev": std_records,
            "coefficient_of_variation": coeff_variation,
        }

        return final_aggregates, telemetry
