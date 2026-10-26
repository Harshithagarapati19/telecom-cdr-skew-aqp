"""
Main Execution Entry Point:
Big Data Analytics - Data Skew and Load Balancing in Distributed Processing
via Approximate Query Processing with Sketch-based Algorithms (HyperLogLog, Count-Min Sketch).
Applied to Telecom Network Call Detail Record (CDR) Analysis.

Academic Project Guide: Mrs. P. Neeraja
"""

import os
import sys
import time

# Ensure src directory is accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
sys.path.insert(0, SRC_DIR)

from benchmark import run_benchmark
from visualizer import plot_all_figures


def main():
    print("=" * 80)
    print("  BIG DATA ANALYTICS: DATA SKEW & LOAD BALANCING VIA SKETCH-BASED AQP")
    print("  Application: Telecom Network Call Detail Record (CDR) Analysis")
    print("  Guide: Mrs. P. Neeraja")
    print("=" * 80)

    # 1. Execute Benchmark across Skew Factors
    print("\n[STEP 1/3] Running Distributed Execution Benchmark across Skew Levels...")
    t_start = time.time()
    benchmark_res = run_benchmark(
        num_records=100_000,
        num_partitions=8,
        alphas=[0.0, 0.5, 1.0, 1.5, 1.8, 2.0],
        output_dir=RESULTS_DIR,
    )
    t_elapsed = time.time() - t_start
    print(f"[STEP 1/3 COMPLETED] Benchmark completed in {t_elapsed:.2f}s.")

    # 2. Generate Visualizations
    print("\n[STEP 2/3] Generating Academic Figures & Plots...")
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    plot_all_figures(
        benchmark_json_path=benchmark_res["json_path"], output_dir=figures_dir
    )
    print(f"[STEP 2/3 COMPLETED] Figures generated in {figures_dir}")

    # 3. Print Executive Summary Table
    print("\n[STEP 3/3] Performance & Accuracy Summary Table:")
    print("-" * 105)
    print(
        f"{'Skew Factor':<12} | {'Exact Latency':<14} | {'AQP Latency':<12} | "
        f"{'Speedup':<9} | {'Mem Saved':<10} | {'Shuffle Saved':<14} | {'HLL Error':<10}"
    )
    print("-" * 105)
    for entry in benchmark_res["history"]:
        skew_name = f"alpha={entry['zipf_alpha']:.1f}"
        print(
            f"{skew_name:<12} | {entry['exact_latency_sec']:>10.4f} s    | {entry['aqp_latency_sec']:>8.4f} s   | "
            f"{entry['straggler_speedup']:>7.2f}x  | {entry['memory_reduction_pct']:>8.1f}%  | "
            f"{entry['shuffle_reduction_pct']:>12.1f}% | {entry['hll_caller_mape_pct']:>8.2f}%"
        )
    print("-" * 105)

    print("\nAll deliverables generated successfully!")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Figures Directory: {figures_dir}")


if __name__ == "__main__":
    main()
