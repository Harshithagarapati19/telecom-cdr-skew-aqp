"""
Visualization Suite for Telecom CDR Distributed Skew Mitigation and AQP Benchmark.
Generates publication-quality charts for academic papers and presentation slides:
  1. Worker Load Distribution (Exact vs. AQP under severe skew)
  2. Straggler Task Latency vs. Zipfian Skew Factor (alpha)
  3. Memory Footprint and Network Shuffle Bytes (Log scale)
  4. HLL Empirical Error vs. Theoretical Error Bounds
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Style settings for academic figures
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})


def plot_all_figures(
    benchmark_json_path: str = "../results/benchmark_results.json",
    output_dir: str = "../results/figures",
):
    """Generate all analytical figures from benchmark telemetry."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(benchmark_json_path):
        print(f"[ERROR] Benchmark results file not found at: {benchmark_json_path}")
        return

    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n[VISUALIZER] Generating figures in: {output_dir}")

    alphas = [d["zipf_alpha"] for d in data]
    exact_latencies = [d["exact_latency_sec"] for d in data]
    aqp_latencies = [d["aqp_latency_sec"] for d in data]
    exact_stragglers = [d["exact_straggler_ratio"] for d in data]
    aqp_stragglers = [d["aqp_straggler_ratio"] for d in data]
    exact_mems = [d["exact_peak_memory_kb"] for d in data]
    aqp_mems = [d["aqp_peak_memory_kb"] for d in data]
    exact_shuffles = [d["exact_shuffle_bytes"] / 1024.0 for d in data]  # KB
    aqp_shuffles = [d["aqp_shuffle_bytes"] / 1024.0 for d in data]  # KB
    hll_errors = [d["hll_caller_mape_pct"] for d in data]

    # Find extreme skew entry (e.g. alpha = 1.8 or max alpha)
    skew_entry = next((d for d in data if d["zipf_alpha"] >= 1.8), data[-1])

    # =========================================================================
    # Figure 1: Worker Partition Load Distribution (Exact vs. AQP with Salting)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    num_parts = len(skew_entry["exact_partition_counts"])
    x = np.arange(num_parts)
    width = 0.38

    rects1 = ax.bar(
        x - width / 2,
        skew_entry["exact_partition_counts"],
        width,
        label=f"Exact Hash Partitioning (CV={skew_entry['exact_cv']:.2f})",
        color="#D9534F",
        edgecolor="black",
        linewidth=0.8,
    )
    rects2 = ax.bar(
        x + width / 2,
        skew_entry["aqp_partition_counts"],
        width,
        label=f"Proposed AQP with Salting (CV={skew_entry['aqp_cv']:.2f})",
        color="#2E7D32",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xlabel("Distributed Worker Partition ID")
    ax.set_ylabel("Records Assigned per Worker")
    ax.set_title(
        f"Worker Load Balancing Under Extreme Skew (Zipf $\\alpha={skew_entry['zipf_alpha']:.1f}$)\n"
        f"Total Records = {skew_entry['num_records']:,} | Salting Factor = 4"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Worker {i}" for i in range(num_parts)])
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, axis="y")

    # Annotate max straggler
    max_exact = max(skew_entry["exact_partition_counts"])
    max_worker = skew_entry["exact_partition_counts"].index(max_exact)
    ax.annotate(
        f"Straggler Worker!\n{max_exact:,} records\n({max_exact / skew_entry['num_records'] * 100:.1f}% of total)",
        xy=(max_worker - width / 2, max_exact),
        xytext=(max_worker - 0.5, max_exact * 0.85),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=7),
        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.8),
        fontsize=9,
        fontweight="bold",
    )

    plt.tight_layout()
    f1_path = os.path.join(output_dir, "fig1_worker_load_distribution.png")
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {f1_path}")

    # =========================================================================
    # Figure 2: Effective Latency & Straggler Ratio vs. Data Skew (alpha)
    # =========================================================================
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_exact = "#D9534F"
    color_aqp = "#1E88E5"

    ax1.plot(
        alphas,
        exact_latencies,
        marker="o",
        linewidth=2.2,
        color=color_exact,
        label="Exact Stage Latency (Straggler Bound)",
    )
    ax1.plot(
        alphas,
        aqp_latencies,
        marker="s",
        linewidth=2.2,
        color=color_aqp,
        label="AQP Stage Latency (Balanced Execution)",
    )
    ax1.set_xlabel("Data Skew Parameter (Zipf $\\alpha$) [0.0 = Uniform, 2.0 = Heavy Skew]")
    ax1.set_ylabel("Distributed Job Latency (seconds)", color="#222222")
    ax1.tick_params(axis="y")
    ax1.grid(True)

    # Secondary axis: Straggler Ratio
    ax2 = ax1.twinx()
    ax2.plot(
        alphas,
        exact_stragglers,
        marker="^",
        linestyle="--",
        color="#8E24AA",
        linewidth=1.8,
        label="Exact Straggler Ratio (Max / Mean)",
    )
    ax2.plot(
        alphas,
        aqp_stragglers,
        marker="v",
        linestyle="--",
        color="#43A047",
        linewidth=1.8,
        label="AQP Straggler Ratio",
    )
    ax2.set_ylabel("Straggler Ratio ($\\max(T_i) / \\mathrm{mean}(T_i)$)", color="#8E24AA")
    ax2.tick_params(axis="y", labelcolor="#8E24AA")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.92)

    ax1.set_title("Distributed Stage Latency and Straggler Ratio vs. Data Skew ($\\alpha$)")
    plt.tight_layout()
    f2_path = os.path.join(output_dir, "fig2_latency_straggler_vs_skew.png")
    plt.savefig(f2_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {f2_path}")

    # =========================================================================
    # Figure 3: Peak Worker Memory & Shuffle Bytes (Log Scale Comparison)
    # =========================================================================
    fig, (ax_mem, ax_shuf) = plt.subplots(1, 2, figsize=(12, 5))

    x_indices = np.arange(len(alphas))
    bwidth = 0.35

    # Memory plot
    ax_mem.bar(
        x_indices - bwidth / 2,
        exact_mems,
        bwidth,
        label="Exact (O(N) Hash Sets)",
        color="#D9534F",
        edgecolor="black",
        linewidth=0.8,
    )
    ax_mem.bar(
        x_indices + bwidth / 2,
        aqp_mems,
        bwidth,
        label="Sketch-AQP (Fixed Register Array)",
        color="#2E7D32",
        edgecolor="black",
        linewidth=0.8,
    )
    ax_mem.set_xlabel("Zipf Skew ($\\alpha$)")
    ax_mem.set_ylabel("Peak Worker Memory (KB)")
    ax_mem.set_title("Peak Worker Memory Footprint")
    ax_mem.set_xticks(x_indices)
    ax_mem.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax_mem.legend()
    ax_mem.grid(True, axis="y")

    # Shuffle bytes plot (Log Scale)
    ax_shuf.bar(
        x_indices - bwidth / 2,
        exact_shuffles,
        bwidth,
        label="Exact (Raw Row Shuffle)",
        color="#E65100",
        edgecolor="black",
        linewidth=0.8,
    )
    ax_shuf.bar(
        x_indices + bwidth / 2,
        aqp_shuffles,
        bwidth,
        label="AQP (Compact Sketch Exchange)",
        color="#0288D1",
        edgecolor="black",
        linewidth=0.8,
    )
    ax_shuf.set_yscale("log")
    ax_shuf.set_xlabel("Zipf Skew ($\\alpha$)")
    ax_shuf.set_ylabel("Network Shuffle Payload (KB, Log Scale)")
    ax_shuf.set_title("Network Shuffle Volume Reduction")
    ax_shuf.set_xticks(x_indices)
    ax_shuf.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax_shuf.legend()
    ax_shuf.grid(True, which="both", axis="y")

    plt.tight_layout()
    f3_path = os.path.join(output_dir, "fig3_memory_and_shuffle_comparison.png")
    plt.savefig(f3_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {f3_path}")

    # =========================================================================
    # Figure 4: HyperLogLog Accuracy and Theoretical Error Bounds
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 5))

    precisions = [8, 10, 12, 14, 16]
    registers = [1 << p for p in precisions]
    theoretical_errors = [(1.04 / np.sqrt(m)) * 100.0 for m in registers]

    ax.plot(
        precisions,
        theoretical_errors,
        marker="s",
        color="#C2185B",
        linewidth=2.2,
        label=r"Theoretical Bound: $\sigma = \frac{1.04}{\sqrt{2^p}} \times 100\%$",
    )

    # Empirical point from this benchmark (p=10)
    avg_emp_error = float(np.mean(hll_errors))
    ax.scatter(
        [10],
        [avg_emp_error],
        color="#00796B",
        s=140,
        zorder=5,
        label=f"Empirical Mean Absolute Error (p=10): {avg_emp_error:.2f}%",
    )

    ax.set_xlabel("HLL Precision Parameter ($p$, Registers $m = 2^p$)")
    ax.set_ylabel("Relative Error Percentage (%)")
    ax.set_title("HyperLogLog Cardinality Accuracy: Empirical vs. Theoretical Bound")
    ax.set_xticks(precisions)
    ax.set_xticklabels([f"p={p}\n(m={1<<p})" for p in precisions])
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    f4_path = os.path.join(output_dir, "fig4_hll_error_bounds.png")
    plt.savefig(f4_path, dpi=300)
    plt.close()
    print(f"  [+] Saved: {f4_path}")

    print("[VISUALIZER] Successfully created all 4 publication charts.")


if __name__ == "__main__":
    plot_all_figures()
