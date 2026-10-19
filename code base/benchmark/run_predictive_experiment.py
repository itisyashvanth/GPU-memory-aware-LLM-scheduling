"""
run_predictive_experiment.py

Compares our two candidate policies:
1. MemoryAwareAdmissionPolicy -- reserves worst-case memory, zero OOM by design
2. PredictiveBatchingPolicy -- reserves predicted memory, trades some OOM
   risk for higher throughput

Sweeps predictor accuracy (noise_sigma) to show how prediction quality
affects the tradeoff -- this is the real research question for this angle:
"how good does a predictor need to be before predictive batching is worth
the added OOM risk, if any?"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scheduler"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "traces"))

import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from memory_model import ModelConfig, MemoryModel
from policy import MemoryAwareAdmissionPolicy, PredictiveBatchingPolicy
from predictor import SimulatedLengthPredictor, TrainedLengthPredictor
from simulator import run_simulation
from generator import generate_trace

# Single-seed quick checks; the reported multi-seed plots come from run_full_simulation_eval.py
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "single_seed")
os.makedirs(RESULTS_DIR, exist_ok=True)
TRAINED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "scheduler", "trained_predictor.pkl")


def load_trained_predictor():
    if not os.path.exists(TRAINED_MODEL_PATH):
        print("No trained model found -- run scheduler/train_predictor.py first. Skipping.")
        return None
    with open(TRAINED_MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    print(f"Loaded trained predictor: {saved['model_name']} "
          f"(test MAE={saved['test_mae']:.1f} tokens, R^2={saved['test_r2']:.3f})")
    return TrainedLengthPredictor(saved["model"], None, saved["feature_names"])


def main():
    model_config = ModelConfig()
    gpu_kv_cache_pool_bytes = 2 * (1024 ** 3)
    memory_model = MemoryModel(model_config, gpu_kv_cache_pool_bytes)

    trace = generate_trace(num_requests=800, arrival_rate_per_step=8.0, seed=7)

    # Admission control (worst-case reservation) -- our existing baseline-of-the-two
    admission_policy = MemoryAwareAdmissionPolicy(memory_model, safe_threshold=0.85)
    admission_result = run_simulation(trace, admission_policy, memory_model)
    admission_summary = admission_result.summary(gpu_kv_cache_pool_bytes)

    # Predictive batching across a range of predictor accuracy levels
    noise_levels = {
        "Perfect predictor (0.0)": 0.0,
        "Accurate predictor (0.2)": 0.2,
        "Moderate predictor (0.5)": 0.5,
        "Poor predictor (1.0)": 1.0,
    }

    predictive_summaries = {}
    for label, noise_sigma in noise_levels.items():
        predictor = SimulatedLengthPredictor(noise_sigma=noise_sigma, seed=123)
        policy = PredictiveBatchingPolicy(memory_model, predictor, safe_threshold=0.85)
        result = run_simulation(trace, policy, memory_model)
        predictive_summaries[label] = result.summary(gpu_kv_cache_pool_bytes)

    # --- Real trained predictor (train_predictor.py), not simulated noise ---
    trained_predictor = load_trained_predictor()
    if trained_predictor is not None:
        policy = PredictiveBatchingPolicy(memory_model, trained_predictor, safe_threshold=0.85)
        result = run_simulation(trace, policy, memory_model)
        predictive_summaries["REAL trained predictor (LinReg)"] = result.summary(gpu_kv_cache_pool_bytes)

    # --- Print comparison table ---
    print("=" * 85)
    print(f"{'Policy':<32}{'OOM%':<10}{'Completed':<12}{'Tokens':<10}{'PeakUtil%':<10}")
    print("-" * 85)
    print(f"{'Memory-Aware Admission (worst-case)':<32}{admission_summary['oom_rate_pct']:<10}"
          f"{admission_summary['completed']:<12}{admission_summary['throughput_tokens_completed']:<10}"
          f"{admission_summary['peak_utilization_pct']:<10}")
    for label, summary in predictive_summaries.items():
        print(f"{'Predictive: ' + label:<32}{summary['oom_rate_pct']:<10}"
              f"{summary['completed']:<12}{summary['throughput_tokens_completed']:<10}"
              f"{summary['peak_utilization_pct']:<10}")
    print("=" * 85)

    # --- Plot: throughput vs OOM risk tradeoff ---
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(admission_summary["oom_rate_pct"], admission_summary["completed"],
               s=150, marker="*", color="#2d5e2d", label="Memory-Aware Admission (worst-case)", zorder=5)

    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#1D9E75"]
    for (label, summary), color in zip(predictive_summaries.items(), colors):
        ax.scatter(summary["oom_rate_pct"], summary["completed"], s=100, color=color, label=label)

    ax.set_xlabel("OOM rate (%)")
    ax.set_ylabel("Requests completed")
    ax.set_title("Throughput vs. OOM Risk: Admission Control vs. Predictive Batching")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "predictive_vs_admission_tradeoff.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved: {out_path}")

    print("\nInterpretation: this plot is the actual research finding for this angle.")
    print("If predictive batching with a REALISTIC predictor (0.2-0.5 noise) sits")
    print("noticeably up-and-right of the admission-control star with near-zero OOM,")
    print("that's evidence predictive batching is worth the engineering complexity.")
    print("If it takes a near-perfect predictor to beat admission control meaningfully,")
    print("that's an equally valid (and honest) finding: worst-case reservation may be")
    print("the better real-world choice given how hard length prediction actually is.")


if __name__ == "__main__":
    main()
