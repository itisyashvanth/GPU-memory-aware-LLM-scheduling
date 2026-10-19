# Project Overview

**Title:** GPU Memory-Aware Scheduling for Large-Batch LLM Inference
**Course:** Cloud Computing (HPC / Cloud) — due 2nd week of October 2026
**Team:** 2 members, low budget
**Status (2026-10-14):** implementation, simulation, real-engine evaluation on llama.cpp (Apple M3) and on vLLM (NVIDIA RTX 5080, 5 seeds, Qwen2.5-0.5B and Llama-3.2-1B), report and IEEE-style paper done. See `code_repo/report/REPORT.md`.

## 1. The project in one paragraph
When an LLM answers requests, every request keeps a "KV cache" in GPU memory that grows with each generated token. If a server accepts too many requests at once, the KV-cache pool runs out and requests fail; if it is too cautious, the GPU sits underused. We built an **admission controller that decides ACCEPT / DELAY / REJECT for each request from KV-cache headroom**, ran it in a simulator and as a real HTTP gateway in front of an LLM engine, and found which way of *accounting* for memory actually prevents failures on real hardware.

## 2. The problem (and why it happens)
- LLM decoding is limited by GPU memory (the KV cache), not compute.
- KV cache per token = `2 × layers × kv_heads × head_dim × dtype_bytes` (28,672 B for the simulated 1.5B-class model; 12,288 B for the Qwen2.5-0.5B used on the real engine).
- A request's final output length is **unknown at arrival**, so its final memory need is unknown; only the cap (`max_tokens` ≤ 2000) is known.

## 3. The gap
Base paper: Zhang, Yang, Pan, Yuan, IEEE TPDS 37(1), 2026 — DRS, a deep-RL scheduler that jointly sets GPU resource share and batch size for multi-tenant DNN inference. It treats per-request cost as roughly fixed and does not model memory that grows during a request or per-request admission on live memory. We specialise the idea to the LLM KV cache.

## 4. What we built
1. **Admission policy** (`code_repo/scheduler/policy.py`): accept if `B + estimate(new request) ≤ 0.85 × capacity`, else delay; reject after waiting too long. Two ways to compute `B`:
   - **current-usage** — memory in use now (the original design; *not* a guarantee),
   - **reserved** — memory promised to every admitted request (worst case: zero OOM by construction).
2. **Predictive variant**: estimate = predicted output length from a linear regression on prompt length + category (MAE 46.6 tokens, R² 0.619) instead of the 2000-token worst case.
3. **Smallest-first** queue ordering (by the policy's own estimate).
4. **One policy, two bodies:** the same `decide()` runs in the simulator and in a real gateway (`vllm_integration.py`) that works in front of vLLM or llama.cpp, with an in-flight ledger, fail-closed metrics, enforced output caps and real status codes.

## 5. Key results (details and sources in `code_repo/README.md`)
**Real engine** (llama.cpp on the Apple M3 GPU, Qwen2.5-0.5B, 16,384-token shared KV pool, 300-request traces, 3 seeds, policy never sees the true output length):
- No admission control: **128 of 300 requests (42.7%) killed** by KV-pool exhaustion, in whole-batch chunks.
- Current-usage policies: still fail (worst case 2 of 3 seeds, predicted 3 of 3). Cause found: requests queued inside the engine are invisible to its memory metrics.
- **Reserved accounting: 0 failures in all 12 runs.** Worst-case lengths are too cautious (47 of 300 completed); **predicted lengths complete 155, +48% goodput, p95 37.6 → 27.4 s.**
- Correction: the first live runs leaked each request's true length to the policy via `max_tokens` (an oracle). Fixed with `--ignore-client-max-tokens`; all numbers above are from the corrected runs.

**Simulation** (2 GB pool, 800-request heavy trace, 5 seeds):
- No control 28.9% OOM; every admission policy 0% at threshold 0.85.
- Worst-case reserved is safe but completes only 77 / 800; **predicted reserved completes 522 with 0% OOM**.
- Smallest-first: **+21%** completions (current-usage), with a fairness cost (2% vs 33% rejection for short vs summarisation requests).

**vLLM on an NVIDIA GPU** (vLLM 0.31.0, RTX 5080 Laptop, 16,384-token pool, 64 seqs, 300 requests, 5 seeds; Qwen2.5-0.5B in `live_summary_vllm_gpu_qwen2_5-0_5b-instruct.json`, Meta's Llama-3.2-1B in `live_summary_vllm_gpu_llama-3_2-1b-instruct.json`):
- vLLM never fails requests; it **preempts**. No control: **74.0 ± 20.9 preemptions per run** (46–103).
- **Every gateway configuration: 0 preemptions in 5 of 5 seeds.**
- But no control had the **lowest tail latency**; predicted reserved had **13% lower goodput** and **p99 10.8 s vs 8.4 s** (worse in 5 of 5 seeds). Preemption of a 0.5B model on this GPU is cheaper than waiting in the gateway.
- **Llama-3.2-1B, same pattern, smaller gap:** 90.0 ± 16.1 preemptions without control → 0 with any gateway; predicted reserved −7% goodput, p99 21.2 vs 18.2 s; predicted current-usage matches no control's goodput.
- Contrary to our hypothesis, **current-usage accounting also had 0 preemptions** on vLLM (likely because requests drain 4–5× faster than on the M3; not directly measured).
- Takeaway: admission control pays off when the engine's own recovery is expensive (llama.cpp), not when it is cheap (vLLM, small model, fast GPU).

## 6. Honest limitations
- Real engines serve small models (0.5B and 1B) with a small pool; models above 1B on vLLM (where preemption costs more) were not tested.
- On vLLM the scheduler prevents preemptions but costs some goodput and tail latency; vLLM's internal queue was not instrumented.
- Traffic and predictor training data are synthetic.
- The simulator has no engine queue or prefill phase and was optimistic for current-usage accounting.
- Single engine; no cluster / Kubernetes component.
- Literature: the 6 journal citations are verified; the broader ~40-paper survey was not expanded.

## 7. Where things are
- Paper (IEEE conference format): `paper/paper.pdf` (source `paper/paper.tex`, figures from `paper/make_figures.py`)
- Report: `code_repo/report/REPORT.md`
- How to reproduce: `code_repo/README.md`, `code_repo/docs/setup.md`
- Numbers: `code_repo/results/sim_summary.json`, `code_repo/results/live/live_summary*_nooracle.json`
- vLLM (GPU) results: `code_repo/results/live/live_summary_vllm_gpu_qwen2_5-0_5b-instruct.json`, `..._llama-3_2-1b-instruct.json` (the old CPU run `live_summary_vllm.json` is superseded)
- Papers: `papers/PAPERS.md` (authoritative journal-only list)

## 8. Note on the overview PDF
`docs_overview/…Overview.pdf` is the earlier planning document. It frames the problem partly as memory *bandwidth* and promises cluster/Kubernetes work; the delivered project addresses KV-cache *capacity* on a single engine. Use this file and the report for what was actually built.
