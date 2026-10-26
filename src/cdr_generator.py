"""
Synthetic Telecom Call Detail Record (CDR) Generator with Controllable Data Skew.
Simulates real-world Power-Law / Zipfian skew in telecom networks:
- Heavy-hitter cell towers (stadiums, transit hubs, downtown business centers)
- High-frequency automated calling entities (robocallers, notification dispatchers)
- Long-tail regular subscribers
Vectorized with NumPy for high-throughput generation.
"""

import time
import math
from typing import List, Dict, Generator, Tuple, Any
import numpy as np


class CDRGenerator:
    """
    Generates realistic Telecom Call Detail Records (CDRs) with configurable Zipfian skew.
    """

    def __init__(
        self,
        num_towers: int = 100,
        num_subscribers: int = 50_000,
        zipf_alpha: float = 1.5,
        random_seed: int = 42,
    ):
        self.num_towers = num_towers
        self.num_subscribers = num_subscribers
        self.zipf_alpha = zipf_alpha
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Pre-compute tower probability distribution based on Zipf parameter
        self.tower_ids = [f"TOWER_{i:04d}" for i in range(num_towers)]
        if zipf_alpha <= 0.001:
            self.tower_probs = np.ones(num_towers, dtype=np.float64) / num_towers
        else:
            ranks = np.arange(1, num_towers + 1, dtype=np.float64)
            weights = 1.0 / (ranks**zipf_alpha)
            self.tower_probs = weights / np.sum(weights)

        # Pre-compute subscriber phone numbers
        self.subscriber_pool = [
            f"+1-555-{i // 10000:03d}-{i % 10000:04d}"
            for i in range(num_subscribers)
        ]

        # Heavy-hitter callers
        self.num_heavy_callers = max(5, int(num_subscribers * 0.001))

        self.call_types = ["VOICE", "SMS", "DATA", "ROAMING"]
        self.call_type_weights = [0.45, 0.25, 0.25, 0.05]
        self.statuses = ["SUCCESS", "DROPPED", "BUSY", "NO_ANSWER"]
        self.status_weights = [0.85, 0.05, 0.05, 0.05]

    def generate_batch(
        self, num_records: int, start_time: int = 1700000000
    ) -> List[Dict[str, Any]]:
        """Vectorized generation of CDR batches for maximum throughput."""
        tower_indices = self.rng.choice(
            self.num_towers, size=num_records, p=self.tower_probs
        )
        caller_indices = self.rng.integers(0, self.num_subscribers, size=num_records)
        callee_indices = self.rng.integers(0, self.num_subscribers, size=num_records)

        # Heavy hitter caller injection under skew
        if self.zipf_alpha > 0.5:
            heavy_mask = self.rng.random(size=num_records) < 0.35
            heavy_callers = self.rng.integers(0, self.num_heavy_callers, size=num_records)
            caller_indices = np.where(heavy_mask, heavy_callers, caller_indices)

        # Eliminate identical caller and callee
        same_mask = caller_indices == callee_indices
        if np.any(same_mask):
            callee_indices[same_mask] = (callee_indices[same_mask] + 1) % self.num_subscribers

        type_indices = self.rng.choice(
            len(self.call_types), size=num_records, p=self.call_type_weights
        )
        status_indices = self.rng.choice(
            len(self.statuses), size=num_records, p=self.status_weights
        )
        durations = np.maximum(
            0, (self.rng.exponential(scale=120, size=num_records) + 5).astype(int)
        )

        records = [None] * num_records
        tower_ids = self.tower_ids
        sub_pool = self.subscriber_pool
        call_types = self.call_types
        statuses = self.statuses

        for i in range(num_records):
            records[i] = {
                "record_id": i,
                "caller_id": sub_pool[caller_indices[i]],
                "callee_id": sub_pool[callee_indices[i]],
                "cell_tower_id": tower_ids[tower_indices[i]],
                "timestamp": start_time + (i % 3600),
                "call_type": call_types[type_indices[i]],
                "duration_sec": int(durations[i]) if status_indices[i] == 0 else 0,
                "bytes_transferred": 160,
                "status": statuses[status_indices[i]],
            }
        return records

    def compute_ground_truth(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Exact aggregation for baseline ground truth validation."""
        truth = {}
        for r in records:
            tower = r["cell_tower_id"]
            if tower not in truth:
                truth[tower] = {
                    "total_calls": 0,
                    "unique_callers": set(),
                    "unique_callees": set(),
                    "total_duration": 0,
                }
            truth[tower]["total_calls"] += 1
            truth[tower]["unique_callers"].add(r["caller_id"])
            truth[tower]["unique_callees"].add(r["callee_id"])
            truth[tower]["total_duration"] += r["duration_sec"]

        summary = {}
        for tower, data in truth.items():
            summary[tower] = {
                "total_calls": data["total_calls"],
                "distinct_callers": len(data["unique_callers"]),
                "distinct_callees": len(data["unique_callees"]),
                "total_duration": data["total_duration"],
            }
        return summary
