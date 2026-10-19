"""
run_evaluation.py

Two experiments that strengthen the evaluation beyond a single data point:

1. Threshold sweep: runs MemoryAwareAdmissionPolicy across a range of
   safe_threshold values, showing the OOM-rate vs. throughput tradeoff.
   This justifies WHY 0.85 (or whatever value) is a reasonable choice,
   rather than an arbitrary pick.

2. Multi-load comparison: runs all three policies (no-control baseline,
   fixed-concurrency-cap baseline, memory-aware policy) across light/medium/
   heavy load levels, so results aren't cherry-picked from one scenario.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scheduler"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "traces"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from memory_model import ModelConfig, MemoryModel
from policy import NoOpBaselinePolicy, FixedConcurrencyCapPolicy, MemoryAwareAdmissionPolicy
from simulator import run_simulation
from generator import generate_trace


# Single-seed quick checks; the reported multi-seed plots come from run_full_simulation_eval.py
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "single_seed")
os.makedirs(RESULTS_DIR, exist_ok=True)


def experiment_threshold_sweep():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Threshold Sensitivity Sweep")
    print("=" * 70)

    model_config = ModelConfig()
    gpu_kv_cache_pool_bytes = 2 * (1024 ** 3)
    memory_model = MemoryModel(model_config, gpu_kv_cache_pool_bytes)

    trace = generate_trace(num_requests=800, arrival_rate_per_step=8.0, seed=7)

    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
    oom_rates = []
    throughputs = []
    completed_counts = []

    for thresh in thresholds:
        policy = MemoryAwareAdmissionPolicy(memory_model, safe_threshold=thresh, max_wait_steps=50)
        result = run_simulation(trace, policy, memory_model)
        summary = result.summary(gpu_kv_cache_pool_bytes)
        oom_rates.append(summary["oom_rate_pct"])
        throughputs.append(summary["throughput_tokens_completed"])
        completed_counts.append(summary["completed"])
        print(f"  threshold={thresh:.2f}  ->  OOM rate={summary['oom_rate_pct']:5.1f}%  "
              f"completed={summary['completed']:4d}  tokens={summary['throughput_tokens_completed']:6d}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(thresholds, oom_rates, marker="o", color="#c0392b")
    ax1.set_xlabel("Safe threshold")
    ax1.set_ylabel("OOM rate (%)")
    ax1.set_title("OOM Rate vs. Safety Threshold")
    ax1.grid(alpha=0.3)

    ax2.plot(thresholds, completed_counts, marker="o", color="#2d5e2d")
    ax2.set_xlabel("Safe threshold")
    ax2.set_ylabel("Requests completed")
    ax2.set_title("Throughput vs. Safety Threshold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "threshold_sensitivity.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved: {out_path}")
    print("\nInterpretation: lower thresholds (more conservative) drive OOM to zero "
          "but may leave some throughput on the table. Higher thresholds risk OOM "
          "creeping back up. The 'knee' of the OOM curve is the defensible choice "
          "for safe_threshold -- pick the lowest threshold that still achieves 0% OOM, "
          "rather than an arbitrary round number.")


def experiment_multi_load():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Multi-Load-Level Comparison")
    print("=" * 70)

    model_config = ModelConfig()
    gpu_kv_cache_pool_bytes = 2 * (1024 ** 3)
    memory_model = MemoryModel(model_config, gpu_kv_cache_pool_bytes)

    load_levels = {
        "Light":  dict(num_requests=500, arrival_rate_per_step=2.0, seed=7),
        "Medium": dict(num_requests=700, arrival_rate_per_step=5.0, seed=7),
        "Heavy":  dict(num_requests=800, arrival_rate_per_step=8.0, seed=7),
    }

    policies_factory = {
        "No admission control": lambda: NoOpBaselinePolicy(memory_model),
        "Fixed concurrency cap (64)": lambda: FixedConcurrencyCapPolicy(memory_model, max_concurrent=64),
        "Memory-aware admission control": lambda: MemoryAwareAdmissionPolicy(memory_model, safe_threshold=0.85),
    }

    rows = []
    for load_name, load_params in load_levels.items():
        trace = generate_trace(**load_params)
        for policy_name, factory in policies_factory.items():
            policy = factory()
            result = run_simulation(trace, policy, memory_model)
            summary = result.summary(gpu_kv_cache_pool_bytes)
            rows.append((load_name, policy_name, summary))

    # Print table
    print(f"\n{'Load':<8}{'Policy':<32}{'OOM%':<8}{'Completed':<12}{'PeakUtil%':<10}")
    print("-" * 70)
    for load_name, policy_name, summary in rows:
        print(f"{load_name:<8}{policy_name:<32}{summary['oom_rate_pct']:<8}"
              f"{summary['completed']:<12}{summary['peak_utilization_pct']:<10}")

    # Plot: OOM rate by load level and policy
    fig, ax = plt.subplots(figsize=(10, 6))
    load_names = list(load_levels.keys())
    policy_names = list(policies_factory.keys())
    x = range(len(load_names))
    width = 0.25
    colors = ["#c0392b", "#e0a95c", "#2d5e2d"]

    for i, policy_name in enumerate(policy_names):
        vals = [s["oom_rate_pct"] for (l, p, s) in rows if p == policy_name]
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], vals, width, label=policy_name, color=colors[i])

    ax.set_xticks(list(x))
    ax.set_xticklabels(load_names)
    ax.set_ylabel("OOM rate (%)")
    ax.set_title("OOM Rate Across Load Levels and Policies")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "multi_load_comparison.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved: {out_path}")

    return rows


if __name__ == "__main__":
    experiment_threshold_sweep()
    experiment_multi_load()
