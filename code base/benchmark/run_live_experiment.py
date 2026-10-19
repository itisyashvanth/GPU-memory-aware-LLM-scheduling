"""
run_live_experiment.py

The real-engine counterpart of run_simulation.py / run_evaluation.py:
replays the same kind of generated trace as REAL HTTP traffic against a
running llama-server (llama.cpp, Metal GPU on Apple Silicon), once directly
(no admission control) and once through the memory-aware gateway for each
policy variant, over several seeds. Every number reported for the real
engine comes from the JSON this script writes.

Prerequisite -- start the engine with a fixed, shared KV-cache pool:

    llama-server -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf -c 16384 -np 64 \\
        --kv-unified --no-context-shift --metrics --no-webui -cram 0 -ngl 99 --port 8000

Then:

    python3 run_live_experiment.py --seeds 7 11 23 --speed 10

Outputs (results/live/): one raw JSON per run, live_summary.json,
live_summary.csv, live_comparison.png.
"""

import argparse
import asyncio
import csv
import json
import os
import socket
import statistics
import subprocess
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "traces"))

from replay_trace_live import replay, summarize, load_trace  # noqa: E402
from generator import generate_trace, save_trace_csv  # noqa: E402

OUT_DIR = os.path.join(REPO, "results", "live")
DEFAULT_MODEL = {"vllm": "Qwen/Qwen2.5-0.5B-Instruct", "llamacpp": "qwen2.5-0.5b-instruct"}


def prom_sum(text: str, name: str):
    """Sum of every sample of a Prometheus metric (all label sets), or None."""
    import re
    vals = re.findall(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)\s*$", text, re.MULTILINE)
    return sum(float(v) for v in vals) if vals else None


def vllm_kv_pool_tokens(metrics_text: str) -> int:
    """KV pool size from vllm:cache_config_info. Newer builds label it
    kv_cache_size_tokens; otherwise it is num_gpu_blocks x block_size."""
    import re
    m = re.search(r'kv_cache_size_tokens="(\d+)"', metrics_text)
    if m:
        return int(m.group(1))
    blocks = re.search(r'num_gpu_blocks="(\d+)"', metrics_text)
    size = re.search(r'block_size="(\d+)"', metrics_text)
    if blocks and size:
        return int(blocks.group(1)) * int(size.group(1))
    return 0

# label -> gateway policy (None = send straight to the engine)
CONFIGS = {
    "direct (no admission control)": None,
    "gateway: admission, current-usage": "admission_control",
    "gateway: admission, reserved": "admission_control_reserved",
    "gateway: predictive, current-usage": "predictive",
    "gateway: predictive, reserved": "predictive_reserved",
}


def wait_for_port(port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError(f"nothing listening on :{port} after {timeout}s")


def wait_for_idle_engine(engine_url: str, backend: str = "llamacpp", timeout: float = 600.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if backend == "vllm":
            # Idle = nothing running or waiting. (KV usage is not a reliable
            # signal: prefix-cached blocks can stay allocated after a run.)
            text = httpx.get(f"{engine_url}/metrics", timeout=5.0).text
            busy = (prom_sum(text, "vllm:num_requests_running") or 0) + \
                   (prom_sum(text, "vllm:num_requests_waiting") or 0)
            if busy == 0:
                return
        else:
            slots = httpx.get(f"{engine_url}/slots", timeout=5.0).json()
            if not any(s.get("is_processing") for s in slots):
                return
        time.sleep(1.0)
    raise RuntimeError("engine still busy -- previous run did not drain")


def run_one(trace_path, label, policy, args):
    target = args.engine_url
    gateway_proc = None
    if policy is not None:
        backend = getattr(args, "backend", "llamacpp")
        cmd = [sys.executable, "-u", os.path.join(REPO, "scheduler", "vllm_integration.py"),
               "--backend", backend, "--policy", policy, "--port", str(args.gateway_port),
               "--engine-url", args.engine_url, "--threshold", str(args.threshold),
               "--num-layers", str(args.num_layers), "--num-kv-heads", str(args.num_kv_heads),
               "--head-dim", str(args.head_dim)]
        if not getattr(args, "oracle_client_cap", False):
            # The replay sets max_tokens to the trace's TRUE output length (so
            # ignore_eos generates exactly that many tokens). The policy must
            # not see it, or its "worst case" is the true length.
            cmd += ["--ignore-client-max-tokens"]
        if hasattr(args, "dtype_bytes") and args.dtype_bytes != 2:
            cmd += ["--dtype-bytes", str(args.dtype_bytes)]
        gateway_log = open(os.path.join(OUT_DIR, f"gateway{args.tag}.log"), "a")
        gateway_log.write(f"\n=== {label} ===\n")
        gateway_log.flush()
        gateway_proc = subprocess.Popen(cmd, cwd=os.path.join(REPO, "scheduler"),
                                        stdout=gateway_log, stderr=subprocess.STDOUT)
        wait_for_port(args.gateway_port)
        target = f"http://127.0.0.1:{args.gateway_port}"
    try:
        trace = load_trace(trace_path)
        backend = getattr(args, "backend", "llamacpp")
        model_name = getattr(args, "model_name", None) or DEFAULT_MODEL[backend]
        # vLLM preempts instead of failing, so preemptions are its failure signal
        preemptions_before = 0
        if backend == "vllm":
            text = httpx.get(f"{args.engine_url}/metrics", timeout=5.0).text
            preemptions_before = prom_sum(text, "vllm:num_preemptions_total") or 0

        outcomes, wall = asyncio.run(replay(trace, target, model_name, args.speed,
                                            category_prompts=args.category_prompts))
        summary = summarize(outcomes, wall)

        if backend == "vllm":
            text = httpx.get(f"{args.engine_url}/metrics", timeout=5.0).text
            summary["vllm_preemptions"] = int((prom_sum(text, "vllm:num_preemptions_total") or 0)
                                              - preemptions_before)

        if gateway_proc is not None:
            summary["gateway_stats"] = httpx.get(f"{target}/gateway/stats", timeout=5.0).json()
    finally:
        if gateway_proc is not None:
            gateway_proc.terminate()
            gateway_proc.wait(timeout=10)
    return summary, outcomes


ENGINE_LABEL = ["Qwen2.5-0.5B, Apple M3"]  # set from --engine-label in main()


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(CONFIGS)
    short = ["Direct\n(no control)", "Admission\ncurrent", "Admission\nreserved",
             "Predictive\ncurrent", "Predictive\nreserved"]
    # vLLM preempts instead of failing, so plot preemptions in place of failures
    if any("vllm_preemptions" in r for r in rows):
        middle = ("vllm_preemptions", "vLLM preemptions (KV exhausted)")
    else:
        middle = ("engine_failures", "Engine failures (KV exhausted)")
    metrics = [("completed", "Completed requests"), middle, ("rejected", "Rejected by gateway (429)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (key, title) in zip(axes, metrics):
        means = [statistics.mean(r[key] for r in rows if r["config"] == l) for l in labels]
        sds = [statistics.pstdev([r[key] for r in rows if r["config"] == l]) for l in labels]
        bars = ax.bar(range(len(labels)), means, yerr=sds, capsize=4,
                      color=["#c0392b", "#e0a95c", "#2d5e2d", "#7F77DD", "#1D9E75"])
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(short, fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    n_seeds = len({r["seed"] for r in rows})
    fig.suptitle(f"Real engine ({ENGINE_LABEL[0]}) -- mean ± sd over {n_seeds} seeds")
    plt.tight_layout()
    plt.savefig(path, dpi=120)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    p.add_argument("--num-requests", type=int, default=300)
    p.add_argument("--arrival-rate", type=float, default=8.0)
    p.add_argument("--speed", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--engine-url", default="http://127.0.0.1:8000")
    p.add_argument("--gateway-port", type=int, default=8080)
    p.add_argument("--num-layers", type=int, default=24)      # Qwen2.5-0.5B
    p.add_argument("--num-kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--category-prompts", action="store_true",
                   help="prompts open with a category phrase the gateway's categorizer can detect")
    p.add_argument("--oracle-client-cap", action="store_true",
                   help="let the gateway policy see max_tokens (= the TRUE output length in a replay); "
                        "reproduces the original, oracle-informed runs. Off by default.")
    p.add_argument("--tag", default="", help="suffix for output file names (keeps runs separate)")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), help="subset of config labels")
    p.add_argument("--backend", choices=["vllm", "llamacpp"], default="llamacpp",
                   help="which LLM engine backend to use")
    p.add_argument("--dtype-bytes", type=int, default=2, help="dtype size for KV cache (2=fp16, 4=fp32)")
    p.add_argument("--model-name", default=None,
                   help="model name sent in each request (vLLM: the HF id it serves; "
                        "default Qwen/Qwen2.5-0.5B-Instruct for vllm)")
    p.add_argument("--engine-label", default=None,
                   help="text for the plot title, e.g. 'Qwen2.5-0.5B, vLLM, RTX 5080'")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.engine_label:
        ENGINE_LABEL[0] = args.engine_label
    if args.backend == "vllm":
        # For vLLM, read capacity from /metrics
        metrics_text = httpx.get(f"{args.engine_url}/metrics", timeout=5.0).text
        kv_tokens = vllm_kv_pool_tokens(metrics_text)
        if not kv_tokens:
            raise SystemExit("could not read the KV pool size from vllm:cache_config_info -- "
                             "check `curl <engine>/metrics | grep cache_config`")
        print(f"Engine: vLLM, KV pool {kv_tokens} tokens, model {args.model_name or DEFAULT_MODEL['vllm']}")
    else:
        engine_props = httpx.get(f"{args.engine_url}/slots", timeout=5.0).json()
        print(f"Engine: {len(engine_props)} slots, shared KV pool n_ctx={engine_props[0]['n_ctx']} tokens")

    rows = []
    for seed in args.seeds:
        trace_path = os.path.join(OUT_DIR, f"trace_seed{seed}.csv")
        save_trace_csv(generate_trace(args.num_requests, args.arrival_rate, seed=seed), trace_path)
        for label in args.configs:
            wait_for_idle_engine(args.engine_url, backend=args.backend)
            print(f"[seed {seed}] {label} ...", flush=True)
            summary, outcomes = run_one(trace_path, label, CONFIGS[label], args)
            slug = label.split("(")[0].strip().replace(":", "").replace(",", "").replace(" ", "_")
            with open(os.path.join(OUT_DIR, f"run{args.tag}_seed{seed}_{slug}.json"), "w") as f:
                json.dump({"config": label, "seed": seed, "summary": summary,
                           "outcomes": [o.__dict__ for o in outcomes]}, f, indent=1)
            row = {"config": label, "seed": seed, **{k: v for k, v in summary.items() if k != "gateway_stats"}}
            rows.append(row)
            print(f"    completed={row['completed']} engine_failures={row['engine_failures']} "
                  f"rejected={row['rejected']} goodput={row['goodput_tokens_per_second']} tok/s", flush=True)

    meta = {k: v for k, v in vars(args).items()}
    if args.backend == "vllm":
        meta["engine_type"] = "vllm"
        meta["engine_kv_pool_tokens"] = kv_tokens
    else:
        meta["engine_type"] = "llamacpp"
        meta["engine_slots"] = len(engine_props)
        meta["engine_kv_pool_tokens"] = engine_props[0]["n_ctx"]
    with open(os.path.join(OUT_DIR, f"live_summary{args.tag}.json"), "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    with open(os.path.join(OUT_DIR, f"live_summary{args.tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 128)
    print(f"{'Config':<38}{'Completed':>12}{'EngineFail':>12}{'Preempt':>10}{'Rejected':>10}"
          f"{'Goodput tok/s':>15}{'p95 lat s':>11}{'p99 lat s':>11}")
    print("-" * 128)
    for label in args.configs:
        rs = [r for r in rows if r["config"] == label]
        def ms(key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return f"{statistics.mean(vals):.1f}±{statistics.pstdev(vals):.1f}" if vals else "-"
        print(f"{label:<38}{ms('completed'):>12}{ms('engine_failures'):>12}{ms('vllm_preemptions'):>10}"
              f"{ms('rejected'):>10}{ms('goodput_tokens_per_second'):>15}{ms('p95_latency_seconds'):>11}"
              f"{ms('p99_latency_seconds'):>11}")
    print("=" * 128)

    if set(args.configs) == set(CONFIGS):
        plot(rows, os.path.join(OUT_DIR, f"live_comparison{args.tag}.png"))
        print(f"Saved: {os.path.join(OUT_DIR, f'live_comparison{args.tag}.png')}")


if __name__ == "__main__":
    main()
