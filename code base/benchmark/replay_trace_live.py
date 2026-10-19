"""
replay_trace_live.py

Replays a generated trace as REAL HTTP traffic against a running server
(either the memory-aware gateway on :8080, or vLLM directly on :8000 for
baseline comparison) -- and collects REAL results (latency, success/failure,
OOM/429 rejections) in the same summary format as the simulator, so
simulation results and real results are directly comparable.

This is the bridge between everything we validated in simulation and the
real hardware run. Usage:

    # Against the memory-aware gateway (our policy):
    python3 replay_trace_live.py --trace sample_trace.csv --target http://localhost:8080 --speed 10

    # Directly against vLLM, no gateway (baseline, no admission control):
    python3 replay_trace_live.py --trace sample_trace.csv --target http://localhost:8000 --speed 10

--speed scales simulated time-steps to real seconds (e.g. --speed 10 means
1 simulated time-step = 0.1 real seconds) so a trace with thousands of
steps doesn't take absurdly long to replay -- tune this based on how fast
your model actually generates tokens in practice.

Outcome classes: "completed" (HTTP 200), "rejected" (HTTP 429 from the
gateway -- a clean, immediate, retryable refusal), "engine_failure" (HTTP
5xx from the engine, e.g. llama.cpp's "Context size has been exceeded" when
its KV-cache pool is exhausted -- the real-hardware equivalent of the
simulator's oom_dropped), and "error" (timeouts / connection failures).

--ignore-eos (default on) makes the engine generate exactly max_tokens, so
each request's real output length matches the trace's output_len instead
of stopping early at an end-of-sequence token.
"""

import argparse
import asyncio
import csv
import time
import json
from dataclasses import dataclass, asdict

import httpx


@dataclass
class RequestOutcome:
    request_id: int
    arrival_time: int
    prompt_len: int
    output_len: int
    status: str          # "completed" | "rejected" | "engine_failure" | "error"
    latency_seconds: float = None
    http_status: int = None
    error: str = None
    completion_tokens: int = None
    gateway_wait_seconds: float = None
    category: str = None


def load_trace(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return [
            {
                "request_id": int(row["request_id"]),
                "arrival_time": int(row["arrival_time"]),
                "prompt_len": int(row["prompt_len"]),
                "output_len": int(row["output_len"]),
                "category": row.get("category") or None,
            }
            for row in reader
        ]


FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog. "  # 10 tokens (Qwen2.5 tokenizer)


# Opening phrases that let categorizer.py recover each request's trace
# category from the prompt text (used with --category-prompts).
CATEGORY_OPENERS = {
    "summarization": "Summarize the following text. ",
    "long_form": "Write an essay in detail about the following. ",
    "short_qa": "What is this? ",
    "chat": "Hey, let's chat. ",
}


def make_prompt(prompt_len: int, request_id: int = 0, category: str = None) -> str:
    """
    Builds a filler prompt of approximately prompt_len tokens. The filler
    sentence tokenizes to 10 tokens; the remainder is padded with single-
    token words. A per-request tag keeps prompts distinct so the engine's
    prompt cache can't make requests artificially cheap. Not meant to be
    semantically meaningful text.
    """
    opener = CATEGORY_OPENERS.get(category, "") if category else ""
    if category == "short_qa":
        # categorizer.py only labels short (<120 chars) questions as short_qa
        return f"{request_id}: What is this? " + "dog " * max(prompt_len - 8, 0)
    opener_tokens = len(opener.split()) + 1 if opener else 0
    sentences, remainder = divmod(max(prompt_len - 3 - opener_tokens, 1), 10)
    return f"Request {request_id}: " + opener + FILLER_SENTENCE * sentences + "dog " * remainder


async def send_request(client: httpx.AsyncClient, target: str, req: dict, model_name: str,
                       ignore_eos: bool = True, category_prompts: bool = False) -> RequestOutcome:
    start = time.monotonic()
    body = {
        "model": model_name,
        "prompt": make_prompt(req["prompt_len"], req["request_id"],
                              req.get("category") if category_prompts else None),
        "max_tokens": req["output_len"],
    }
    if ignore_eos:
        body["ignore_eos"] = True
    try:
        resp = await client.post(f"{target}/v1/completions", json=body, timeout=900.0)
        latency = time.monotonic() - start
        waited = resp.headers.get("X-Gateway-Waited-Seconds")
        waited = float(waited) if waited is not None else None
        if resp.status_code == 429:
            return RequestOutcome(**req, status="rejected", latency_seconds=latency, http_status=429)
        if resp.status_code >= 500:
            try:
                message = resp.json().get("error", {}).get("message", resp.text[:200])
            except Exception:
                message = resp.text[:200]
            return RequestOutcome(**req, status="engine_failure", latency_seconds=latency,
                                  http_status=resp.status_code, error=str(message)[:200],
                                  gateway_wait_seconds=waited)
        resp.raise_for_status()
        usage = resp.json().get("usage") or {}
        return RequestOutcome(**req, status="completed", latency_seconds=latency, http_status=resp.status_code,
                              completion_tokens=usage.get("completion_tokens"), gateway_wait_seconds=waited)
    except Exception as e:
        latency = time.monotonic() - start
        return RequestOutcome(**req, status="error", latency_seconds=latency, error=repr(e)[:200])


async def replay(trace, target: str, model_name: str, speed: float, ignore_eos: bool = True,
                 category_prompts: bool = False):
    outcomes = []
    tasks = []
    replay_start = time.monotonic()

    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=200)
    async with httpx.AsyncClient(limits=limits) as client:
        for req in trace:
            target_real_time = req["arrival_time"] / speed
            elapsed = time.monotonic() - replay_start
            wait = target_real_time - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            # Fire-and-track: don't block the arrival loop waiting for completion
            tasks.append(asyncio.create_task(send_request(client, target, req, model_name, ignore_eos, category_prompts)))

        print(f"All {len(trace)} requests dispatched, waiting for completions...")
        outcomes = await asyncio.gather(*tasks)

    return outcomes, time.monotonic() - replay_start


def summarize(outcomes, wall_seconds: float = None):
    completed = [o for o in outcomes if o.status == "completed"]
    rejected = [o for o in outcomes if o.status == "rejected"]
    errored = [o for o in outcomes if o.status == "error"]
    failed = [o for o in outcomes if o.status == "engine_failure"]
    latencies = [o.latency_seconds for o in completed]
    total = len(outcomes)
    tokens = sum(o.completion_tokens or 0 for o in completed)

    summary = {
        "total_requests": total,
        "completed": len(completed),
        "rejected": len(rejected),
        "engine_failures": len(failed),
        "errored": len(errored),
        "engine_failure_rate_pct": round(100 * len(failed) / total, 2) if total else 0,
        "error_rate_pct": round(100 * len(errored) / total, 2) if total else 0,
        "rejection_rate_pct": round(100 * len(rejected) / total, 2) if total else 0,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "p95_latency_seconds": round(sorted(latencies)[int(0.95 * len(latencies))], 3) if latencies else None,
        "p99_latency_seconds": round(sorted(latencies)[int(0.99 * len(latencies))], 3) if latencies else None,
        "completed_output_tokens": tokens,
        "wall_seconds": round(wall_seconds, 1) if wall_seconds else None,
        "goodput_tokens_per_second": round(tokens / wall_seconds, 1) if wall_seconds else None,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Replay a trace as real HTTP traffic")
    parser.add_argument("--trace", required=True, help="Path to trace CSV (from traces/generator.py)")
    parser.add_argument("--target", required=True, help="Base URL: gateway (:8080) or vLLM directly (:8000)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name as vLLM expects it")
    parser.add_argument("--speed", type=float, default=10.0,
                        help="Time-step compression factor (higher = faster replay)")
    parser.add_argument("--output", default="live_results.json", help="Where to save raw outcomes")
    parser.add_argument("--no-ignore-eos", action="store_true",
                        help="Let the model stop at end-of-sequence instead of generating exactly output_len")
    parser.add_argument("--category-prompts", action="store_true",
                        help="Open each prompt with a phrase matching its trace category, so the "
                             "gateway's categorizer (and the predictor) can see the category")
    parser.add_argument("--limit", type=int, default=None, help="Only replay the first N requests")
    args = parser.parse_args()

    trace = load_trace(args.trace)[:args.limit]
    print(f"Loaded {len(trace)} requests from {args.trace}")
    print(f"Replaying against {args.target} at speed={args.speed}x")

    outcomes, wall_seconds = asyncio.run(replay(trace, args.target, args.model, args.speed,
                                                not args.no_ignore_eos, args.category_prompts))

    summary = summarize(outcomes, wall_seconds)
    print("\n" + "=" * 50)
    print("LIVE REPLAY RESULTS")
    print("=" * 50)
    for k, v in summary.items():
        print(f"{k:<28}{v}")

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "config": vars(args), "outcomes": [asdict(o) for o in outcomes]}, f, indent=2)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
