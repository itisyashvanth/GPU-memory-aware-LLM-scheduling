"""
run_simulation.py

Runs the baseline (no admission control, mirrors default vLLM batching)
against our memory-aware admission control policy, on the same trace,
and reports comparison metrics + plots.

Usage:
    python3 run_simulation.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scheduler"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "traces"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from memory_model import ModelConfig, MemoryModel
from policy import NoOpBaselinePolicy, MemoryAwareAdmissionPolicy
from simulator import run_simulation
from generator import generate_trace


def main():
    # --- Setup: model + GPU capacity ---
    model_config = ModelConfig()  # defaults ~ Qwen2.5-1.5B-class
    # Simulate a GPU KV-cache pool -- e.g. ~2GB reserved for KV cache on a shared/smaller
    # allocation (rest of VRAM goes to model weights + activations + other tenants)
    gpu_kv_cache_pool_bytes = 2 * (1024 ** 3)
    memory_model = MemoryModel(model_config, gpu_kv_cache_pool_bytes)

    print(f"Model: {model_config}")
    print(f"Bytes per token (KV cache): {model_config.bytes_per_token:,.0f}")
    print(f"GPU KV-cache pool: {gpu_kv_cache_pool_bytes / (1024**3):.1f} GB")
    print(f"=> Max tokens the pool can hold at once: "
          f"{gpu_kv_cache_pool_bytes / model_config.bytes_per_token:,.0f}")
    print()

    # --- Generate a bursty trace (heavy load, to actually stress memory capacity) ---
    trace = generate_trace(
        num_requests=800,
        arrival_rate_per_step=8.0,   # bursty: 8 requests/step on average -- deliberately heavy
        seed=7,
    )

    # --- Baseline: no admission control ---
    baseline_policy = NoOpBaselinePolicy(memory_model)
    baseline_result = run_simulation(trace, baseline_policy, memory_model)

    # --- Our policy: memory-aware admission control ---
    our_policy = MemoryAwareAdmissionPolicy(memory_model, safe_threshold=0.85, max_wait_steps=50)
    our_result = run_simulation(trace, our_policy, memory_model)

    # --- Report ---
    baseline_summary = baseline_result.summary(gpu_kv_cache_pool_bytes)
    our_summary = our_result.summary(gpu_kv_cache_pool_bytes)

    print("=" * 70)
    print(f"{'Metric':<32}{'Baseline (no control)':<22}{'Memory-Aware Policy':<20}")
    print("=" * 70)
    for key in baseline_summary:
        print(f"{key:<32}{str(baseline_summary[key]):<22}{str(our_summary[key]):<20}")
    print("=" * 70)

    # --- Plot: memory usage over time ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for ax, result, label, color in [
        (axes[0], baseline_result, "Baseline (no admission control)", "#c0392b"),
        (axes[1], our_result, "Memory-Aware Admission Control", "#2d5e2d"),
    ]:
        times = [t for t, _ in result.memory_trace]
        used_gb = [b / (1024**3) for _, b in result.memory_trace]
        ax.plot(times, used_gb, color=color, linewidth=1)
        ax.axhline(y=gpu_kv_cache_pool_bytes / (1024**3), color="black", linestyle="--",
                   linewidth=1, label="GPU KV-cache capacity")
        ax.set_title(label)
        ax.set_xlabel("Time step")
        ax.set_ylabel("KV-cache memory used (GB)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "memory_usage_comparison.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot: {out_path}")

    # --- Plot: outcome comparison bar chart ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    categories = ["Completed", "OOM Dropped", "Rejected"]
    baseline_vals = [baseline_summary["completed"], baseline_summary["oom_dropped"], baseline_summary["rejected"]]
    our_vals = [our_summary["completed"], our_summary["oom_dropped"], our_summary["rejected"]]

    x = range(len(categories))
    width = 0.35
    ax2.bar([i - width/2 for i in x], baseline_vals, width, label="Baseline", color="#c0392b")
    ax2.bar([i + width/2 for i in x], our_vals, width, label="Memory-Aware Policy", color="#2d5e2d")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Number of requests")
    ax2.set_title("Request Outcomes: Baseline vs. Memory-Aware Admission Control")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out_path2 = os.path.join(os.path.dirname(__file__), "..", "results", "outcome_comparison.png")
    plt.savefig(out_path2, dpi=120)
    print(f"Saved plot: {out_path2}")


if __name__ == "__main__":
    main()
