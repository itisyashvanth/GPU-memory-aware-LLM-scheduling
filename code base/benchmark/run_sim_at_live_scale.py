"""
run_sim_at_live_scale.py

Sim-to-real check: runs the SIMULATOR configured exactly like the real
llama.cpp experiment (Qwen2.5-0.5B: 12,288 bytes/token; a 16,384-token KV
pool; the same 300-request traces, seeds 7/11/23) to see whether the
simulator predicts what the real engine did. Writes results/sim_at_live_scale.json.
"""

import json
import os
import pickle
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scheduler"))
sys.path.insert(0, os.path.join(HERE, "..", "traces"))

from memory_model import ModelConfig, MemoryModel
from policy import NoOpBaselinePolicy, MemoryAwareAdmissionPolicy, PredictiveBatchingPolicy
from predictor import TrainedLengthPredictor
from simulator import run_simulation
from generator import generate_trace

cfg = ModelConfig(num_layers=24, num_kv_heads=2, head_dim=64, dtype_bytes=2)
POOL_TOKENS = 16384
mm = MemoryModel(cfg, POOL_TOKENS * cfg.bytes_per_token)
with open(os.path.join(HERE, "..", "scheduler", "trained_predictor.pkl"), "rb") as f:
    saved = pickle.load(f)
trained = TrainedLengthPredictor(saved["model"], None, saved["feature_names"])

policies = {
    "direct (no admission control)": lambda: NoOpBaselinePolicy(mm),
    "admission, current-usage": lambda: MemoryAwareAdmissionPolicy(mm, 0.85),
    "admission, reserved": lambda: MemoryAwareAdmissionPolicy(mm, 0.85, accounting="reserved"),
    "predictive, current-usage": lambda: PredictiveBatchingPolicy(mm, trained, 0.85),
    "predictive, reserved": lambda: PredictiveBatchingPolicy(mm, trained, 0.85, accounting="reserved"),
}
out = {}
print(f"{'Policy':<34}{'Completed':>12}{'OOM-dropped':>13}{'Rejected':>10}")
for name, make in policies.items():
    rows = [run_simulation(generate_trace(300, 8.0, seed=s), make(), mm).summary(mm.total_capacity_bytes)
            for s in (7, 11, 23)]
    out[name] = {k: round(statistics.mean(r[k] for r in rows), 1) for k in ("completed", "oom_dropped", "rejected")}
    print(f"{name:<34}{out[name]['completed']:>12}{out[name]['oom_dropped']:>13}{out[name]['rejected']:>10}")
with open(os.path.join(HERE, "..", "results", "sim_at_live_scale.json"), "w") as f:
    json.dump({"pool_tokens": POOL_TOKENS, "bytes_per_token": cfg.bytes_per_token, "seeds": [7, 11, 23],
               "mean_over_seeds": out}, f, indent=1)
