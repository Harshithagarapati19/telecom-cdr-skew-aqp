"""
Two-Tier Sketch-based Approximate Query Processing (AQP) Processor with Skew Mitigation.
Solves Data Skew and Straggler Bottlenecks in Distributed Systems using:
1. Dynamic Salting via Count-Min Sketch Heavy-Hitter Pre-filtering
2. Local Monoid Pre-Aggregation (Worker-side HLL and CMS)
3. Zero-loss / Bounded-error Hierarchical Sketch Merge (O(1) payload transfer)
"""

import time
import sys
import hashlib
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from sketches import CountMinSketch, HyperLogLog


class SketchAQPDistributedProcessor:
    """
    Distributed AQP Engine utilizing Count-Min Sketch and HyperLogLog
    with Dynamic Key Salting to eliminate partition skew and stragglers.
    """

    def __init__(
        self,
        num_partitions: int = 8,
        hll_precision: int = 10,
        cms_epsilon: float = 0.005,
        cms_delta: float = 0.01,
        enable_salting: bool = True,
        salt_factor: int = 4,
        heavy_hitter_threshold_pct: float = 0.05,
    ):
        """
        :param num_partitions: Number of distributed worker partitions.
        :param hll_precision: HLL precision parameter p (2^p registers, error ~ 1.04 / sqrt(2^p)).
        :param cms_epsilon: Count-Min Sketch error parameter.
        :param cms_delta: Count-Min Sketch failure probability.
        :param enable_salting: Whether to activate dynamic key salting for hot keys.
        :param salt_factor: Number of sub-partitions to split hot keys across.
        :param heavy_hitter_threshold_pct: Threshold fraction of records to designate a key as 'heavy'.
        """
        self.num_partitions = num_partitions
        self.hll_precision = hll_precision
        self.cms_epsilon = cms_epsilon
        self.cms_delta = cms_delta
        self.enable_salting = enable_salting
        self.salt_factor = salt_factor
        self.heavy_threshold_pct = heavy_hitter_threshold_pct

    def _hash_partition(self, key: str) -> int:
        """Hash partition function: hash(key) % W."""
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.num_partitions

    def execute_aggregation(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Executes Skew-Balanced AQP aggregation query:
            SELECT cell_tower_id,
                   COUNT(*) AS approx_total_calls,
                   COUNT(DISTINCT caller_id) AS approx_unique_callers,
                   COUNT(DISTINCT callee_id) AS approx_unique_callees,
                   SUM(duration_sec) AS total_duration
            FROM cdr_records
            GROUP BY cell_tower_id;
        """
        start_wall_time = time.perf_counter()
        total_records = len(records)
        heavy_threshold = int(total_records * self.heavy_threshold_pct)

        # Stage 1: Stream Pre-Filter with Count-Min Sketch to detect Hot Keys (Heavy Hitters)
        hot_keys = set()
        cms_detector = CountMinSketch(epsilon=self.cms_epsilon, delta=self.cms_delta)

        # First pass / streaming sample to identify heavy keys
        sample_step = max(1, total_records // 5000)
        for i in range(0, total_records, sample_step):
            row = records[i]
            tower = row["cell_tower_id"]
            cms_detector.update(tower, sample_step)
            if cms_detector.query(tower) >= heavy_threshold:
                hot_keys.add(tower)

        # Stage 2: Ingestion & Salting Partition Routing
        # Split input into balanced worker queues
        worker_input_queues = [[] for _ in range(self.num_partitions)]
        shuffle_bytes_transferred = 0

        for r in records:
            tower = r["cell_tower_id"]
            if self.enable_salting and tower in hot_keys:
                # Disperse hot tower across salt_factor sub-partitions
                salt = hash(r["caller_id"] + str(r["timestamp"])) % self.salt_factor
                salted_key = f"{tower}__salt_{salt}"
                p_idx = self._hash_partition(salted_key)
                worker_input_queues[p_idx].append((salted_key, tower, r))
            else:
                p_idx = self._hash_partition(tower)
                worker_input_queues[p_idx].append((tower, tower, r))

        # Stage 3: Worker Node Local Sketch Aggregation
        # Each worker updates local Count-Min Sketch and HLLs instead of allocating raw sets
        partition_metrics = []
        intermediate_worker_sketches = (
            []
        )  # List of dict: salted_key -> {hll_caller, hll_callee, count, duration}

        for p_idx, queue in enumerate(worker_input_queues):
            p_start = time.perf_counter()

            worker_sketch_state: Dict[str, Dict[str, Any]] = {}

            for salted_key, orig_tower, row in queue:
                caller = row["caller_id"]
                callee = row["callee_id"]
                dur = row["duration_sec"]

                if salted_key not in worker_sketch_state:
                    worker_sketch_state[salted_key] = {
                        "orig_tower": orig_tower,
                        "count": 0,
                        "duration": 0,
                        "hll_caller": HyperLogLog(p=self.hll_precision),
                        "hll_callee": HyperLogLog(p=self.hll_precision),
                    }

                entry = worker_sketch_state[salted_key]
                entry["count"] += 1
                entry["duration"] += dur
                entry["hll_caller"].add(caller)
                entry["hll_callee"].add(callee)

            p_end = time.perf_counter()
            p_time = p_end - p_start

            # Calculate memory footprint: fixed size per tower (1024 bytes per HLL + counter overhead)
            # Notice memory is INDEPENDENT of number of records processed by this worker!
            num_keys_tracked = len(worker_sketch_state)
            sketch_mem_bytes = num_keys_tracked * (
                2 * (1 << self.hll_precision) + 64
            )
            worker_mem_kb = sketch_mem_bytes / 1024.0

            partition_metrics.append(
                {
                    "partition_id": p_idx,
                    "record_count": len(queue),
                    "execution_time_sec": p_time,
                    "peak_memory_kb": worker_mem_kb,
                }
            )

            intermediate_worker_sketches.append(worker_sketch_state)

        # Stage 4: Hierarchical Sketch Merge (O(1) transfer per key)
        # Only the sketch registers are transferred across the network, NOT raw records!
        merge_start = time.perf_counter()
        final_aggregates: Dict[str, Dict[str, Any]] = {}

        for worker_state in intermediate_worker_sketches:
            for salted_key, state in worker_state.items():
                orig_tower = state["orig_tower"]
                # Payload: 2 HLL sketches (each 2^p bytes) + count (8 bytes) + duration (8 bytes)
                shuffle_bytes_transferred += 2 * (1 << self.hll_precision) + 16

                if orig_tower not in final_aggregates:
                    final_aggregates[orig_tower] = {
                        "count": 0,
                        "duration": 0,
                        "hll_caller": HyperLogLog(p=self.hll_precision),
                        "hll_callee": HyperLogLog(p=self.hll_precision),
                    }

                target = final_aggregates[orig_tower]
                target["count"] += state["count"]
                target["duration"] += state["duration"]
                target["hll_caller"].merge_inplace(state["hll_caller"])
                target["hll_callee"].merge_inplace(state["hll_callee"])

        merge_end = time.perf_counter()
        merge_time = merge_end - merge_start

        # Extract estimates
        final_results = {}
        for tower, state in final_aggregates.items():
            final_results[tower] = {
                "total_calls": state["count"],
                "distinct_callers": round(state["hll_caller"].estimate()),
                "distinct_callees": round(state["hll_callee"].estimate()),
                "total_duration": state["duration"],
            }

        end_wall_time = time.perf_counter()

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
            "mode": "Sketch-AQP",
            "num_partitions": self.num_partitions,
            "total_records": len(records),
            "total_wall_time_sec": end_wall_time - start_wall_time,
            "effective_distributed_latency_sec": straggler_time + merge_time,
            "worker_straggler_time_sec": straggler_time,
            "merge_phase_time_sec": merge_time,
            "straggler_ratio": straggler_ratio,
            "partition_counts": record_counts,
            "partition_times_sec": worker_times,
            "partition_memory_kb": memories,
            "peak_worker_memory_kb": max(memories),
            "shuffle_bytes": shuffle_bytes_transferred,
            "record_count_stddev": std_records,
            "coefficient_of_variation": coeff_variation,
            "hot_keys_detected": list(hot_keys),
        }

        return final_results, telemetry
