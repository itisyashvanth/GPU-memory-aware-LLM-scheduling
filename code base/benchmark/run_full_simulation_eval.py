"""
run_full_simulation_eval.py

The consolidated simulation evaluation. Every simulation number quoted in
the README / report comes from results/sim_summary.json written here.
Each experiment is repeated over several trace seeds and reported as
mean ± standard deviation, so no result rests on one lucky trace.

Experiments
  E1 main      -- all policies on the heavy default trace (800 req, 8/step)
  E2 stress    -- same, on a long-output-heavy mix (50% long_form): does the
                  original current-usage check still hold?
  E3 threshold -- safe_threshold sweep, current vs reserved accounting
  E4 ordering  -- FIFO vs smallest-first (honest) vs smallest-first (oracle,
                  true lengths -- reference only) + fairness by category
  E5 predictor -- predictor accuracy sweep (simulated noise + trained model)
  E6 load      -- light / medium / heavy load levels

Usage: python3 run_full_simulation_eval.py [--seeds 1 2 3 4 5]
"""

import argparse
import json
import os
import pickle
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scheduler"))
sys.path.insert(0, os.path.join(HERE, "..", "traces"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from memory_model import ModelConfig, MemoryModel
from policy import (NoOpBaselinePolicy, FixedConcurrencyCapPolicy, MemoryAwareAdmissionPolicy,
                    PredictiveBatchingPolicy)
from predictor import SimulatedLengthPredictor, TrainedLengthPredictor
from simulator import run_simulation
from generator import generate_trace, REQUEST_CATEGORIES

RESULTS_DIR = os.path.join(HERE, "..", "results")
CAPACITY = 2 * (1024 ** 3)
MM = MemoryModel(ModelConfig(), CAPACITY)
HEAVY = dict(num_requests=800, arrival_rate_per_step=8.0)
LONG_HEAVY_MIX = {"short_qa": 0.1, "chat": 0.2, "summarization": 0.2, "long_form": 0.5}
METRICS = ("completed", "rejected", "oom_dropped", "oom_rate_pct", "throughput_tokens_completed",
           "avg_latency_steps", "p95_latency_steps", "peak_utilization_pct", "avg_utilization_pct")


def load_trained():
    path = os.path.join(HERE, "..", "scheduler", "trained_predictor.pkl")
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return TrainedLengthPredictor(saved["model"], None, saved["feature_names"]), saved


def policy_factories(trained):
    return {
        "No admission control": lambda: NoOpBaselinePolicy(MM),
        "Fixed concurrency cap (64)": lambda: FixedConcurrencyCapPolicy(MM, max_concurrent=64),
        "Memory-aware, current-usage": lambda: MemoryAwareAdmissionPolicy(MM, 0.85),
        "Memory-aware, reserved": lambda: MemoryAwareAdmissionPolicy(MM, 0.85, accounting="reserved"),
        "Predictive (trained), current-usage": lambda: PredictiveBatchingPolicy(MM, trained, 0.85),
        "Predictive (trained), reserved": lambda: PredictiveBatchingPolicy(MM, trained, 0.85,
                                                                            accounting="reserved"),
    }


def run(trace, policy, order="fifo"):
    return run_simulation(trace, policy, MM, admission_order=order)


def agg(summaries):
    out = {}
    for m in METRICS:
        vals = [s[m] for s in summaries]
        out[m] = {"mean": round(statistics.mean(vals), 2), "sd": round(statistics.pstdev(vals), 2),
                  "min": min(vals), "max": max(vals)}
    return out


def fmt(a, m, digits=1):
    return f"{a[m]['mean']:.{digits}f}±{a[m]['sd']:.{digits}f}"


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'Policy':<40}{'Completed':>14}{'OOM %':>13}{'Rejected':>14}{'PeakUtil%':>12}{'p95 lat':>13}")
    for name, a in rows.items():
        print(f"{name:<40}{fmt(a, 'completed'):>14}{fmt(a, 'oom_rate_pct', 2):>13}{fmt(a, 'rejected'):>14}"
              f"{fmt(a, 'peak_utilization_pct'):>12}{fmt(a, 'p95_latency_steps'):>13}")


def comparison(seeds, trace_kwargs, factories):
    rows = {}
    for name, make in factories.items():
        rows[name] = agg([run(generate_trace(seed=s, **trace_kwargs), make()).summary(CAPACITY) for s in seeds])
    return rows


def bar_plot(rows, title, path, highlight_oom=True):
    names = list(rows)
    short = [n.replace(", ", "\n").replace(" (", "\n(") for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, label, color in [(axes[0], "completed", "Requests completed", "#2d5e2d"),
                                     (axes[1], "oom_rate_pct", "OOM rate (%)", "#c0392b")]:
        means = [rows[n][metric]["mean"] for n in names]
        sds = [rows[n][metric]["sd"] for n in names]
        bars = ax.bar(range(len(names)), means, yerr=sds, capsize=4, color=color)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.1f}" if metric != "completed" else f"{m:.0f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short, fontsize=7)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def plot_predictor_sweep(out):
    """E5 as grouped bars: completions per predictor accuracy, current vs
    reserved accounting, with the worst-case policies as reference lines.
    (OOM was 0% for every configuration, so a throughput-vs-OOM scatter
    would collapse onto a single vertical line.)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, trace_name in zip(axes, ("default", "long_heavy")):
        names = list(out["E5_predictor"][f"{trace_name}/current"])
        width = 0.38
        for i, (acc, color) in enumerate((("current", "#7F77DD"), ("reserved", "#1D9E75"))):
            series = out["E5_predictor"][f"{trace_name}/{acc}"]
            vals = [series[n]["completed"]["mean"] for n in names]
            sds = [series[n]["completed"]["sd"] for n in names]
            ooms = [series[n]["oom_rate_pct"]["mean"] for n in names]
            bars = ax.bar([x + (i - 0.5) * width for x in range(len(names))], vals, width, yerr=sds, capsize=3,
                          color=color, label=f"predictive, {acc} accounting")
            for b, v, o in zip(bars, vals, ooms):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}" + (f"\n{o:.1f}% OOM" if o else ""),
                        ha="center", va="bottom", fontsize=7)
        base_key = "E1_main_heavy" if trace_name == "default" else "E2_stress_long_heavy"
        for name, style in (("Memory-aware, current-usage", "--"), ("Memory-aware, reserved", ":")):
            ax.axhline(out[base_key][name]["completed"]["mean"], color="#2d5e2d", linestyle=style,
                       label=f"worst-case cap, {name.split(', ')[1]}")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
        ax.set_ylabel("Requests completed")
        ax.set_title(f"{trace_name} trace (OOM = 0% unless labelled)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7, loc="lower right")
    n = len(out["meta"]["seeds"])
    fig.suptitle(f"Predictive batching vs predictor accuracy, mean ± sd over {n} seeds")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "predictive_vs_admission_tradeoff.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--plots-only", action="store_true",
                    help="redraw the E5 predictor plot from an existing results/sim_summary.json")
    args = ap.parse_args()
    if args.plots_only:
        with open(os.path.join(RESULTS_DIR, "sim_summary.json")) as f:
            plot_predictor_sweep(json.load(f))
        return
    seeds = args.seeds
    t0 = time.time()

    trained, saved = load_trained()
    factories = policy_factories(trained)
    out = {"meta": {"seeds": seeds, "capacity_bytes": CAPACITY,
                    "bytes_per_token": MM.model_config.bytes_per_token,
                    "capacity_tokens": int(CAPACITY / MM.model_config.bytes_per_token),
                    "safe_threshold": 0.85, "max_wait_steps": 50, "max_possible_output_len": 2000,
                    "trained_predictor": {"model": saved["model_name"], "test_mae": round(saved["test_mae"], 1),
                                          "test_r2": round(saved["test_r2"], 3)}}}

    # E1 main comparison
    out["E1_main_heavy"] = comparison(seeds, HEAVY, factories)
    print_table(f"E1  Heavy load (800 req, 8/step), {len(seeds)} seeds", out["E1_main_heavy"])
    bar_plot(out["E1_main_heavy"], f"Simulation, heavy load -- mean ± sd over {len(seeds)} seeds",
             os.path.join(RESULTS_DIR, "sim_main_comparison.png"))

    # E2 stress: long-output-heavy traffic
    out["E2_stress_long_heavy"] = comparison(seeds, dict(HEAVY, category_weights=LONG_HEAVY_MIX), factories)
    print_table("E2  Stress: long-output-heavy mix (50% long_form)", out["E2_stress_long_heavy"])
    bar_plot(out["E2_stress_long_heavy"], "Simulation, long-output-heavy stress trace",
             os.path.join(RESULTS_DIR, "sim_stress_comparison.png"))

    # E3 threshold sweep, both accounting modes, both traces
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
    out["E3_threshold"] = {}
    for trace_name, tk in [("default", HEAVY), ("long_heavy", dict(HEAVY, category_weights=LONG_HEAVY_MIX))]:
        for acc in ("current", "reserved"):
            out["E3_threshold"][f"{trace_name}/{acc}"] = {
                str(th): agg([run(generate_trace(seed=s, **tk),
                                  MemoryAwareAdmissionPolicy(MM, th, accounting=acc)).summary(CAPACITY)
                              for s in seeds]) for th in thresholds}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key, style in [("default/current", "o-"), ("default/reserved", "s--"),
                       ("long_heavy/current", "o-"), ("long_heavy/reserved", "s--")]:
        series = out["E3_threshold"][key]
        axes[0].plot(thresholds, [series[str(t)]["oom_rate_pct"]["mean"] for t in thresholds], style, label=key)
        axes[1].plot(thresholds, [series[str(t)]["completed"]["mean"] for t in thresholds], style, label=key)
    axes[0].set_ylabel("OOM rate (%)")
    axes[1].set_ylabel("Requests completed")
    for ax in axes:
        ax.set_xlabel("safe_threshold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Threshold sweep, memory-aware admission (trace / accounting), mean over {len(seeds)} seeds")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "threshold_sensitivity.png"), dpi=120)
    plt.close(fig)
    print("\nE3  Threshold sweep (mean completed / mean OOM%)")
    for key, series in out["E3_threshold"].items():
        print(f"  {key:<22}" + "  ".join(f"{t}:{series[str(t)]['completed']['mean']:.0f}/"
                                         f"{series[str(t)]['oom_rate_pct']['mean']:.1f}" for t in thresholds))

    # E4 ordering + fairness
    out["E4_ordering"] = {}
    category_rejection = {}
    for pname in ("Memory-aware, current-usage", "Predictive (trained), current-usage", "Memory-aware, reserved"):
        for order in ("fifo", "smallest_first", "smallest_first_oracle"):
            summaries, rej_size, rej_by_cat, n_by_cat = [], [], {}, {}
            for s in seeds:
                trace = generate_trace(seed=s, **HEAVY)
                res = run(trace, factories[pname](), order)
                summaries.append(res.summary(CAPACITY))
                rej_size += [r.prompt_len + r.output_len for r in res.rejected]
                for r in trace:
                    n_by_cat[r["category"]] = n_by_cat.get(r["category"], 0) + 1
                for r in res.rejected:
                    rej_by_cat[r.category] = rej_by_cat.get(r.category, 0) + 1
            a = agg(summaries)
            a["avg_rejected_size_tokens"] = round(statistics.mean(rej_size), 1) if rej_size else 0
            a["rejection_rate_pct_by_category"] = {c: round(100 * rej_by_cat.get(c, 0) / n_by_cat[c], 1)
                                                   for c in REQUEST_CATEGORIES}
            out["E4_ordering"][f"{pname} | {order}"] = a
            category_rejection[f"{pname} | {order}"] = a["rejection_rate_pct_by_category"]
    print("\nE4  Admission ordering (oracle = uses true output_len, NOT achievable; reference only)")
    print(f"{'Policy | order':<62}{'Completed':>14}{'Rejected':>12}{'AvgRejSize':>12}  rejection % by category")
    for k, a in out["E4_ordering"].items():
        print(f"{k:<62}{fmt(a, 'completed'):>14}{a['rejected']['mean']:>12.1f}{a['avg_rejected_size_tokens']:>12}  "
              f"{a['rejection_rate_pct_by_category']}")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pnames = ["Memory-aware, current-usage", "Predictive (trained), current-usage", "Memory-aware, reserved"]
    orders = [("fifo", "FIFO", "#7F77DD"), ("smallest_first", "Smallest-first (honest)", "#1D9E75"),
              ("smallest_first_oracle", "Smallest-first (oracle, reference)", "#bbbbbb")]
    width = 0.26
    for i, (order, label, color) in enumerate(orders):
        vals = [out["E4_ordering"][f"{p} | {order}"]["completed"]["mean"] for p in pnames]
        sds = [out["E4_ordering"][f"{p} | {order}"]["completed"]["sd"] for p in pnames]
        bars = ax.bar([x + (i - 1) * width for x in range(len(pnames))], vals, width, yerr=sds, capsize=3,
                      label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(pnames)))
    ax.set_xticklabels(pnames)
    ax.set_ylabel("Requests completed")
    ax.set_title(f"Admission order: FIFO vs smallest-first, heavy load, mean ± sd over {len(seeds)} seeds")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "optimization_fifo_vs_smallest_first.png"), dpi=120)
    plt.close(fig)

    # E5 predictor accuracy sweep
    out["E5_predictor"] = {}
    predictors = {"perfect (sigma 0.0)": lambda: SimulatedLengthPredictor(0.0, seed=123),
                  "accurate (sigma 0.2)": lambda: SimulatedLengthPredictor(0.2, seed=123),
                  "moderate (sigma 0.5)": lambda: SimulatedLengthPredictor(0.5, seed=123),
                  "poor (sigma 1.0)": lambda: SimulatedLengthPredictor(1.0, seed=123),
                  "trained LinReg": lambda: trained}
    for trace_name, tk in [("default", HEAVY), ("long_heavy", dict(HEAVY, category_weights=LONG_HEAVY_MIX))]:
        for acc in ("current", "reserved"):
            out["E5_predictor"][f"{trace_name}/{acc}"] = {
                pname: agg([run(generate_trace(seed=s, **tk),
                                PredictiveBatchingPolicy(MM, make(), 0.85, accounting=acc)).summary(CAPACITY)
                            for s in seeds]) for pname, make in predictors.items()}
    print("\nE5  Predictor accuracy (mean completed / mean OOM%)")
    for key, series in out["E5_predictor"].items():
        print(f"  {key:<22}" + "  ".join(f"{p}: {a['completed']['mean']:.0f}/{a['oom_rate_pct']['mean']:.2f}"
                                         for p, a in series.items()))
    plot_predictor_sweep(out)

    # E6 load levels
    loads = {"Light": dict(num_requests=500, arrival_rate_per_step=2.0),
             "Medium": dict(num_requests=700, arrival_rate_per_step=5.0),
             "Heavy": HEAVY}
    subset = {k: factories[k] for k in ("No admission control", "Fixed concurrency cap (64)",
                                        "Memory-aware, current-usage", "Memory-aware, reserved",
                                        "Predictive (trained), reserved")}
    out["E6_load"] = {ln: comparison(seeds, lk, subset) for ln, lk in loads.items()}
    print("\nE6  Load levels")
    for ln, rows in out["E6_load"].items():
        print_table(f"  {ln}", rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.16
    colors = ["#c0392b", "#e0a95c", "#7F77DD", "#2d5e2d", "#1D9E75"]
    for ax, metric, label in [(axes[0], "oom_rate_pct", "OOM rate (%)"), (axes[1], "completed", "Requests completed")]:
        for i, (pname, color) in enumerate(zip(subset, colors)):
            vals = [out["E6_load"][ln][pname][metric]["mean"] for ln in loads]
            ax.bar([x + (i - 2) * width for x in range(len(loads))], vals, width, label=pname, color=color)
        ax.set_xticks(range(len(loads)))
        ax.set_xticklabels(list(loads))
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Across load levels, mean over {len(seeds)} seeds")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "multi_load_comparison.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "sim_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved results/sim_summary.json and plots ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
