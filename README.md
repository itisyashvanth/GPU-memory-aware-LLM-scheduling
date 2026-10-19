  # GPU Memory-Aware Scheduling for Large-Batch LLM Inference

Cloud Computing / HPC course project: a memory-aware **admission controller**
(ACCEPT / DELAY / REJECT) in front of an LLM inference engine, driven by
KV-cache headroom, so that bursty traffic cannot exhaust the GPU's KV-cache
pool.

## Team
- [Sai Yashvanth Kanakam] — policy design, simulation, report
- [Makkena Venkat Praheesh] — engine integration, experiments

## What is in this repo

```
scheduler/    policy.py (the policies), simulator.py, memory_model.py,
              predictor.py + train_predictor.py (+ trained_predictor.pkl),
              categorizer.py, vllm_integration.py (the real gateway; vLLM or llama.cpp),
              mock_vllm_server.py, test_policy_simulator.py, test_vllm_integration.py
traces/       generator.py -- Poisson arrivals, lognormal lengths, 4 request categories
benchmarks/   run_full_simulation_eval.py  -- ALL reported simulation numbers (5 seeds)
              run_live_experiment.py       -- ALL reported real-engine numbers (3 seeds)
              run_sim_at_live_scale.py     -- simulator configured exactly like the live run
              replay_trace_live.py         -- replays a trace as real HTTP traffic
              run_simulation.py, run_evaluation.py, run_predictive_experiment.py,
              run_optimization_experiment.py -- original single-seed scripts (quick checks)
results/      sim_summary.json, sim_at_live_scale.json, live/live_summary*.json + plots
              archive_pre_fix/ -- the plots as they were before the October fixes
docs/         problem_statement.md, design_rationale.md, setup.md, literature_tracker.csv
report/       REPORT.md -- the written report
```

## The policies

All policies implement one `decide()` interface, used unchanged by the
simulator and by the real gateway.

| Policy | What it charges a new request | What it compares against |
|---|---|---|
| No admission control | -- | admits everything |
| Fixed concurrency cap (64) | -- | number of active requests |
| **Memory-aware, current-usage** (original) | prompt + 2000-token cap | memory in use *now* |
| **Memory-aware, reserved** (added) | prompt + 2000-token cap | memory *promised* to active requests |
| **Predictive, current / reserved** | prompt + predicted length (LinReg) | as above |

`safe_threshold = 0.85`, `max_wait_steps = 50` (simulation) / 15 s (gateway).
Optional smallest-first ordering of the waiting queue.

**Correction to earlier versions of this README:** the original policy was
described as guaranteeing zero OOM "by construction". It does not: it adds
the new request's worst case to the memory used *right now*, but requests
already admitted keep growing afterwards. Only the **reserved** variant
(added in October) is a guarantee -- with the worst-case cap, total usage
can never exceed `safe_threshold × capacity`.

## Results

All numbers below are copied from the JSON files named; nothing is
hand-entered. Simulation: 2 GB pool (74,898 tokens at 28,672 B/token,
Qwen2.5-1.5B-class), mean ± sd over 5 trace seeds. Real engine: see below.

### Simulation, heavy load (800 requests, 8/step) -- `results/sim_summary.json` E1

| Policy | Completed | OOM % | Rejected | Peak util % |
|---|---|---|---|---|
| No admission control | 569.2 ± 47.3 | **28.85 ± 5.91** | 0 | 104.2 |
| Fixed concurrency cap (64) | 162.4 ± 8.0 | 0 | 637.6 | 21.5 |
| Memory-aware, current-usage | 581.6 ± 49.3 | 0 | 218.4 | 83.6 |
| Memory-aware, reserved | 76.6 ± 5.3 | 0 | 723.4 | 10.1 |
| Predictive (trained), current-usage | **626.4 ± 51.6** | 0 | 173.6 | 86.3 |
| Predictive (trained), reserved | 522.4 ± 36.0 | 0 | 277.6 | 58.0 |

On a long-output-heavy stress mix (50% long_form, E2) the no-control
baseline loses **58.8%** of requests; every admission policy stays at 0%
OOM at threshold 0.85. Current-usage accounting starts to OOM on that mix
at threshold ≥ 0.90 (0.2% at 0.90, 1.4% at 0.95, 2.7% at 0.98, E3);
reserved accounting stays at 0% at every threshold.

Reading: in simulation, reserving the worst case for every request is
safe but very wasteful (cap 2000 vs ~250 average tokens/request); a
**predicted** reservation recovers most of that (522 vs 77 completed)
while staying at 0% OOM.

### Smallest-first ordering -- E4

| Policy | FIFO | Smallest-first | Oracle (true lengths, *reference only*) |
|---|---|---|---|
| Memory-aware, current-usage | 581.6 | **704.4 (+21%)** | 705.6 |
| Predictive, current-usage | 626.4 | **713.0 (+14%)** | 714.0 |
| Memory-aware, reserved | 76.6 | 124.8 | 149.8 |

Earlier versions sorted by the request's **true** output length (not
known at arrival). Ordering now uses only the policy's own estimate; for
the current-usage policies the honest gain is within 1–2 requests of the
oracle. **Fairness cost** (current-usage, smallest-first): rejection rate
2.3% for short_qa vs 33.2% for summarization (FIFO: 25–31% for all);
average size of rejected requests rises from 277 to 478 tokens.

### Predictor accuracy -- E5

Trained linear regression (prompt_len + category): test MAE 46.6 tokens,
R² 0.619 (naive mean: MAE 106.4). Under current-usage accounting, predictor
quality barely matters (623–627 completed from perfect to poor). Under
reserved accounting it does: 513 (perfect) → 466 (poor, σ=1.0); the
trained model gets 522 -- slightly above "perfect" because it
under-predicts long requests and therefore reserves less (and so is
less safe, not more accurate).

### Real engine -- `results/live/live_summary_catprompts_nooracle.json`

Setup: Apple M3 (16 GB), llama.cpp `llama-server` 0.6.0 on Metal,
Qwen2.5-0.5B-Instruct Q4_K_M, **one shared 16,384-token KV pool**, 64
parallel slots. 300-request traces (8 req/step, replayed at 10 steps/s, so
a ~4 s burst), `ignore_eos` so every request generates exactly its trace
length. Prompts open with a category phrase the categorizer can read. The
gateway runs with `--ignore-client-max-tokens`, so it reserves for the
2000-token cap and never sees the true length. Mean ± sd over seeds 7, 11, 23.

When llama.cpp's pool is exhausted its decode step fails and **every
in-flight request** returns HTTP 500 ("Context size has been exceeded"):
failures arrive in whole-batch chunks of ~64.

| Configuration | Completed | Engine failures | Rejected (429) | Goodput tok/s | p95 latency s |
|---|---|---|---|---|---|
| Direct, no admission control | 172.0 ± 0.0 | **128.0 ± 0.0** | 0 | 203.6 ± 32.8 | 37.6 ± 6.7 |
| Gateway: worst case, current-usage | 185.0 ± 7.3 | 42.7 ± 30.2 | 72.3 | 295.4 ± 120.0 | 39.3 ± 1.6 |
| Gateway: worst case, reserved | 47.3 ± 4.7 | **0.0 ± 0.0** | 252.7 | 188.3 ± 20.1 | 20.2 ± 3.5 |
| Gateway: predictive, current-usage | 201.7 ± 20.4 | 85.0 ± 29.7 | 13.3 | 197.1 ± 15.9 | 44.0 ± 7.9 |
| **Gateway: predictive, reserved** | 155.3 ± 17.2 | **0.0 ± 0.0** | 144.7 | **301.0 ± 28.2** | 27.4 ± 3.5 |

![live](results/live/live_comparison_catprompts_nooracle.png)

1. Without admission control, 42.7% of requests are killed by KV-pool exhaustion.
2. **Current-usage accounting does not transfer to a real engine**: both
   variants still lose whole batches. A decision log of a diagnostic run
   shows why: the gateway admitted up to ~230 concurrent requests while the
   engine had only 64 slots. The excess sat in llama.cpp's internal queue,
   invisible in its usage numbers, and overflowed the pool once it started.
   The simulator has no engine-side queue, so configured exactly like the
   live run it still predicts 0 OOM for this policy
   (`results/sim_at_live_scale.json`).
3. **Reserved accounting has 0 failures in every run.** With worst-case
   lengths it is far too cautious (47 of 300 completed). With predicted
   lengths it completes 155, rejects the rest cleanly, and raises goodput
   by 48% over no control with lower p95 latency.

**Filler prompts** (`results/live/live_summary_nooracle.json`): with no
category signal for the predictor, the pattern holds -- predictive-reserved
0 failures in 3 of 3 seeds, 128.7 completed, goodput 220.8 vs 148.9 tok/s
(+48%), p95 37.8 vs 49.9 s; worst-case reserved completes 45.

**Correction.** The first live runs (`live_summary.json`,
`live_summary_catprompts.json`) let the policy see each request's true
output length through `max_tokens` -- the replay sets it to the trace length.
Their worst-case-reserved numbers (~166-172 completed, +41% goodput) are
therefore oracle results and should not be cited. See report §7.4 and
`docs/design_rationale.md`.

### Real engine, vLLM on an NVIDIA GPU -- `results/live/live_summary_vllm_gpu_{qwen2_5-0_5b,llama-3_2-1b}-instruct.json`

Setup: RTX 5080 Laptop GPU (16 GB, sm_120), WSL2 Ubuntu 24.04, **vLLM 0.31.0**
(PyTorch 2.13, CUDA 13.2), **two models in bf16**: Qwen2.5-0.5B-Instruct and Meta's
Llama-3.2-1B-Instruct (weights from the ungated `unsloth/` re-upload), **16,384-token KV pool**
(`--num-gpu-blocks-override 1024`, block 16), `--max-num-seqs 64`, prefix caching
off. 300 requests per run, speed 10, θ = 0.85, category prompts,
`--ignore-client-max-tokens`, **5 seeds** (7, 11, 23, 31, 47). Calibration
(1 seed) gave 101 (Qwen) and 79 (Llama) preemptions without control, so these
settings already create real pressure.

vLLM **preempts** (evicts and recomputes) instead of failing when the pool
fills, so the metric to compare is **preemptions** and **tail latency**.

**Qwen2.5-0.5B-Instruct**

| Configuration | Preemptions | Completed | Rejected | Engine failures | Goodput tok/s | p95 s | p99 s |
|---|---|---|---|---|---|---|---|
| No admission control | 74.0 ± 20.9 | 300.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2699.4 ± 535.9 | 5.8 ± 0.7 | 8.4 ± 0.9 |
| Worst case, current-usage | 0.0 ± 0.0 | 300.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2556.9 ± 412.1 | 7.9 ± 1.0 | 9.7 ± 1.2 |
| Worst case, reserved | 0.0 ± 0.0 | 173.6 ± 15.9 | 126.4 ± 15.9 | 0.0 ± 0.0 | 1022.5 ± 55.5 | 15.6 ± 0.1 | 17.2 ± 0.9 |
| Predicted, current-usage | 0.0 ± 0.0 | 300.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2417.2 ± 383.5 | 6.6 ± 0.4 | 9.9 ± 1.8 |
| Predicted, reserved | 0.0 ± 0.0 | 300.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2345.3 ± 484.6 | 7.3 ± 0.9 | 10.8 ± 1.8 |

Source: `code_repo/results/live/live_summary_vllm_gpu_qwen2_5-0_5b-instruct.json`, seeds [7, 11, 23, 31, 47], pool 16384 tokens, speed 10, 300 requests per run.

**Llama-3.2-1B-Instruct**

| Configuration | Preemptions | Completed | Rejected | Engine failures | Goodput tok/s | p95 s | p99 s |
|---|---|---|---|---|---|---|---|
| No admission control | 90.0 ± 16.1 | 300.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1290.4 ± 289.5 | 13.7 ± 1.6 | 18.2 ± 1.9 |
| Worst case, current-usage | 0.0 ± 0.0 | 297.4 ± 2.1 | 2.6 ± 2.1 | 0.0 ± 0.0 | 1289.4 ± 180.6 | 15.8 ± 0.8 | 20.1 ± 2.3 |
| Worst case, reserved | 0.0 ± 0.0 | 96.0 ± 10.0 | 204.0 ± 10.0 | 0.0 ± 0.0 | 496.2 ± 45.4 | 16.6 ± 0.3 | 20.8 ± 4.0 |
| Predicted, current-usage | 0.0 ± 0.0 | 300.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1286.9 ± 286.7 | 14.4 ± 1.0 | 19.8 ± 2.7 |
| Predicted, reserved | 0.0 ± 0.0 | 298.0 ± 2.4 | 2.0 ± 2.4 | 0.0 ± 0.0 | 1185.1 ± 206.0 | 16.0 ± 0.7 | 21.2 ± 2.0 |

Source: `code_repo/results/live/live_summary_vllm_gpu_llama-3_2-1b-instruct.json`, seeds [7, 11, 23, 31, 47], pool 16384 tokens, speed 10, 300 requests per run.

What this shows (details in report §7.5):

1. **Every gateway configuration removed all preemptions** for both models
   (0 in 5 of 5 seeds each); no control had 46 to 103 (Qwen) and 65 to 115
   (Llama) per run.
2. **But admission control did not pay for itself on vLLM.** No control had
   the lowest p95/p99. Predicted reservation had 13% lower goodput and higher
   p99 than no control in 5 of 5 seeds; the gateway's queueing (143 requests
   held back per run) costs more than vLLM's cheap recomputation of a 0.5B
   model on this GPU. With Llama-3.2-1B the gap narrows (predicted reserved
   −7% goodput, p99 21.2 vs 18.2 s; predicted current-usage matches no
   control's goodput) but does not reverse.
3. **Current-usage accounting did not fail here** (0 preemptions), unlike on
   llama.cpp. Our reading, not measured: requests drain 4-5x faster than on the
   M3, so the engine-queue blind spot was never reached at this load.
4. Worst-case reserved: safe, but 126.4 of 300 rejected and 61% lower goodput.

The older one-seed vLLM **CPU** run on the Mac (`live_summary_vllm.json`)
predates the oracle fix and is superseded; do not cite it.

## How to run

```bash
# from the project root
python3.13 -m venv .venv && .venv/bin/pip install numpy pandas scikit-learn matplotlib fastapi uvicorn httpx
cd code_repo/scheduler && ../../.venv/bin/python test_policy_simulator.py && ../../.venv/bin/python test_vllm_integration.py
cd ../benchmarks && ../../.venv/bin/python run_full_simulation_eval.py      # ~9 min
```

Real engine (Mac): see `docs/setup.md` for llama.cpp (Metal GPU, fast).

vLLM on an NVIDIA GPU (Linux or WSL2, a vLLM build for CUDA 12.8+ on Blackwell):

```bash
bash gpu/check_env.sh                 # must show capability (12, 0) and sm_120
bash gpu/start_vllm.sh                # Qwen2.5-0.5B, 1024 x 16 = 16,384-token pool, 64 seqs
bash gpu/run_vllm_matrix.sh calib     # 1 seed: direct must show tens of preemptions
bash gpu/run_vllm_matrix.sh full      # 5 seeds x 5 configs
```

Notes from the RTX 5080 run: Triton needs Python headers (`Python.h`); we used a
uv-managed Python 3.12 venv instead of `apt install python3.12-dev`. The vLLM
wheel's `torchaudio` was built for a different CUDA than its torch and broke
the CLI; removing it is harmless for a text model. Set `HF_HUB_OFFLINE=1`
once the model is cached. The exact WSL scripts used are in `gpu/wsl/`.

## Base paper
Zhang, Y., Yang, N., Pan, C., & Yuan, D. (2026). "Joint Optimization of
Resource Allocation and Request Batching for Multi-Tenant Inference Serving
on GPU." IEEE Transactions on Parallel and Distributed Systems.
https://ieeexplore.ieee.org/document/11223102/ -- full list in `../papers/PAPERS.md`.

## Limitations
- Single GPU / single engine; no cluster placement or Kubernetes component.
- Real-engine validation uses small models: Qwen2.5-0.5B on llama.cpp (Apple
  Silicon Metal, failures on exhaustion) and Qwen2.5-0.5B + Llama-3.2-1B on vLLM
  (RTX 5080 Laptop GPU, preemption on exhaustion). Models above 1B were not tested.
- Traffic is synthetic (Poisson / lognormal / 4 categories); the predictor is
  trained on that synthetic data. Not validated on a production trace.
- The simulator advances every request one token per step and has no
  engine-side queue or prefill phase; its zero-OOM result for current-usage
  accounting did not transfer to the real engine.
