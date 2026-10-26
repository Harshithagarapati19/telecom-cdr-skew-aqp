"""
Comprehensive Benchmarking Pipeline:
Evaluates Exact Distributed Processing vs. Sketch-based Approximate Query Processing (AQP)
Across varying levels of Zipfian Data Skew (alpha in [0.0, 0.5, 1.0, 1.5, 1.8, 2.0]) and data sizes.

Outputs:
  - Console summary tables
  - benchmark_results.json
  - benchmark_results.csv
"""

import json
import csv
import os
import sys
import time
from typing import Dict, Any, List
import numpy as np

# Ensure local src directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cdr_generator import CDRGenerator
from exact_processor import ExactDistributedProcessor
from sketch_aqp_processor import SketchAQPDistributedProcessor


def calculate_mape(
    exact_results: Dict[str, Dict[str, Any]], aqp_results: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """Calculate Mean Absolute Percentage Error (MAPE) between Exact and AQP."""
    caller_errors = []
    callee_errors = []
    count_errors = []

    for tower, truth in exact_results.items():
        if tower not in aqp_results:
            continue
        est = aqp_results[tower]

        # Call count error
        if truth["total_calls"] > 0:
            err = (
                abs(est["total_calls"] - truth["total_calls"])
                / truth["total_calls"]
                * 100.0
            )
            count_errors.append(err)

        # Distinct callers error
        if truth["distinct_callers"] > 0:
            err = (
                abs(est["distinct_callers"] - truth["distinct_callers"])
                / truth["distinct_callers"]
                * 100.0
            )
            caller_errors.append(err)

        # Distinct callees error
        if truth["distinct_callees"] > 0:
            err = (
                abs(est["distinct_callees"] - truth["distinct_callees"])
                / truth["distinct_callees"]
                * 100.0
            )
            callee_errors.append(err)

    return {
        "call_count_mape_pct": float(np.mean(count_errors)) if count_errors else 0.0,
        "caller_distinct_mape_pct": (
            float(np.mean(caller_errors)) if caller_errors else 0.0
        ),
        "callee_distinct_mape_pct": (
            float(np.mean(callee_errors)) if callee_errors else 0.0
        ),
    }


def run_benchmark(
    num_records: int = 100_000,
    num_partitions: int = 8,
    alphas: List[float] = [0.0, 0.5, 1.0, 1.5, 1.8, 2.0],
    output_dir: str = "../results",
) -> Dict[str, Any]:
    """Execute complete multi-skew benchmark."""
    os.makedirs(output_dir, exist_ok=True)
    results_history = []

    print("\n" + "=" * 90)
    print("  BIG DATA ANALYTICS BENCHMARK: DATA SKEW & LOAD BALANCING VIA SKETCH-BASED AQP")
    print(f"  Configuration: {num_records:,} CDR Records | {num_partitions} Worker Partitions")
    print("=" * 90)

    for alpha in alphas:
        skew_label = (
            "Uniform (No Skew)"
            if alpha == 0.0
            else f"Zipf alpha={alpha:.1f}"
            + (" (Extreme)" if alpha >= 1.8 else " (High)" if alpha >= 1.4 else "")
        )
        print(f"\n>>> Testing Scenario: {skew_label}...")

        # 1. Generate Skewed CDR Dataset
        gen = CDRGenerator(
            num_towers=100,
            num_subscribers=25_000,
            zipf_alpha=alpha,
            random_seed=42,
        )
        records = gen.generate_batch(num_records)

        # 2. Exact Distributed Processing
        exact_engine = ExactDistributedProcessor(num_partitions=num_partitions)
        exact_res, exact_metrics = exact_engine.execute_aggregation(records)

        # 3. Sketch-based AQP Distributed Processing (with Dynamic Salting)
        aqp_engine = SketchAQPDistributedProcessor(
            num_partitions=num_partitions,
            hll_precision=10,  # 1024 registers, theoretical SE ~= 3.25%
            enable_salting=True,
            salt_factor=4,
        )
        aqp_res, aqp_metrics = aqp_engine.execute_aggregation(records)

        # 4. Accuracy Evaluation
        accuracy = calculate_mape(exact_res, aqp_res)

        # 5. Speedup and Resource Savings
        straggler_speedup = (
            exact_metrics["effective_distributed_latency_sec"]
            / aqp_metrics["effective_distributed_latency_sec"]
            if aqp_metrics["effective_distributed_latency_sec"] > 0
            else 1.0
        )
        mem_reduction_pct = (
            (
                1.0
                - (
                    aqp_metrics["peak_worker_memory_kb"]
                    / exact_metrics["peak_worker_memory_kb"]
                )
            )
            * 100.0
            if exact_metrics["peak_worker_memory_kb"] > 0
            else 0.0
        )
        shuffle_reduction_pct = (
            (
                1.0
                - (aqp_metrics["shuffle_bytes"] / exact_metrics["shuffle_bytes"])
            )
            * 100.0
            if exact_metrics["shuffle_bytes"] > 0
            else 0.0
        )

        entry = {
            "zipf_alpha": alpha,
            "skew_label": skew_label,
            "num_records": num_records,
            "num_partitions": num_partitions,
            # Exact Metrics
            "exact_latency_sec": exact_metrics["effective_distributed_latency_sec"],
            "exact_straggler_ratio": exact_metrics["straggler_ratio"],
            "exact_peak_memory_kb": exact_metrics["peak_worker_memory_kb"],
            "exact_shuffle_bytes": exact_metrics["shuffle_bytes"],
            "exact_partition_counts": exact_metrics["partition_counts"],
            "exact_count_stddev": exact_metrics["record_count_stddev"],
            "exact_cv": exact_metrics["coefficient_of_variation"],
            # AQP Metrics
            "aqp_latency_sec": aqp_metrics["effective_distributed_latency_sec"],
            "aqp_worker_straggler_sec": aqp_metrics["worker_straggler_time_sec"],
            "aqp_straggler_ratio": aqp_metrics["straggler_ratio"],
            "aqp_peak_memory_kb": aqp_metrics["peak_worker_memory_kb"],
            "aqp_shuffle_bytes": aqp_metrics["shuffle_bytes"],
            "aqp_partition_counts": aqp_metrics["partition_counts"],
            "aqp_count_stddev": aqp_metrics["record_count_stddev"],
            "aqp_cv": aqp_metrics["coefficient_of_variation"],
            # Comparisons
            "straggler_speedup": straggler_speedup,
            "memory_reduction_pct": mem_reduction_pct,
            "shuffle_reduction_pct": shuffle_reduction_pct,
            # Accuracy
            "hll_caller_mape_pct": accuracy["caller_distinct_mape_pct"],
            "hll_callee_mape_pct": accuracy["callee_distinct_mape_pct"],
        }
        results_history.append(entry)

        print(f"  Exact Latency (Straggler Bound): {exact_metrics['effective_distributed_latency_sec']:.4f}s "
              f"| Straggler Ratio: {exact_metrics['straggler_ratio']:.2f}x "
              f"| Peak Mem: {exact_metrics['peak_worker_memory_kb']:.1f} KB")
        print(f"  AQP Latency   (Balanced Stages): {aqp_metrics['effective_distributed_latency_sec']:.4f}s "
              f"| Straggler Ratio: {aqp_metrics['straggler_ratio']:.2f}x "
              f"| Peak Mem: {aqp_metrics['peak_worker_memory_kb']:.1f} KB")
        print(f"  ==> Speedup: {straggler_speedup:.2f}x | Memory Saved: {mem_reduction_pct:.1f}% "
              f"| Shuffle Saved: {shuffle_reduction_pct:.1f}% | HLL Error: {accuracy['caller_distinct_mape_pct']:.2f}%")

    # Save to JSON
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_history, f, indent=2)

    # Save to CSV
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    fieldnames = [
        "zipf_alpha",
        "skew_label",
        "num_records",
        "num_partitions",
        "exact_latency_sec",
        "exact_straggler_ratio",
        "exact_peak_memory_kb",
        "exact_shuffle_bytes",
        "exact_count_stddev",
        "exact_cv",
        "aqp_latency_sec",
        "aqp_straggler_ratio",
        "aqp_peak_memory_kb",
        "aqp_shuffle_bytes",
        "aqp_count_stddev",
        "aqp_cv",
        "straggler_speedup",
        "memory_reduction_pct",
        "shuffle_reduction_pct",
        "hll_caller_mape_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results_history)

    print("\n" + "=" * 90)
    print(f"  Benchmark complete. Saved results to:\n    - {json_path}\n    - {csv_path}")
    print("=" * 90)

    return {"history": results_history, "json_path": json_path, "csv_path": csv_path}


if __name__ == "__main__":
    run_benchmark(num_records=100_000, num_partitions=8)
