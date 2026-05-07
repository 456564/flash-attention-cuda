#!/usr/bin/env python3
"""LLM inference benchmark — fixed prompts, multiple runs, averaged results."""

import subprocess
import json
import re
import time
import sys
import os
from datetime import datetime

# === CONFIG ===
LLAMA_BIN = "/home/jetson/lm-inference/llama.cpp/build/bin/llama-simple"
MODEL_PATH = "/home/jetson/lm-inference/models/qwen2-0_5b-instruct-q8_0.gguf"
N_RUNS = 10
MAX_TOKENS = 128

PROMPTS = [
    "Explain the concept of Flash Attention in one paragraph.",
    "What is the difference between FP16 and INT8 quantization?",
    "Write a simple CUDA kernel that adds two vectors.",
    "Describe the Ampere GPU architecture in 3 sentences.",
    "What is Grouped Query Attention and why is it used?",
]


def run_inference(prompt: str) -> dict:
    """Run single inference via llama-simple, return parsed metrics."""
    t0 = time.perf_counter()
    result = subprocess.run(
        [LLAMA_BIN, "-m", MODEL_PATH, "-p", prompt, "-n", str(MAX_TOKENS)],
        capture_output=True, text=True, timeout=300,
        input=""  # non-interactive
    )
    elapsed = time.perf_counter() - t0

    output = result.stdout + result.stderr
    metrics = {
        "prompt": prompt[:100],
        "wall_time_s": round(elapsed, 3),
    }

    # llama-simple output:
    #   "decoded N tokens in X s, speed: Z t/s"
    m = re.search(r"decoded\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s+s,\s+speed:\s+([\d.]+)\s+t/s", output)
    if m:
        metrics["gen_tokens"] = int(m.group(1))
        metrics["gen_time_s"] = float(m.group(2))
        metrics["gen_tps"] = float(m.group(3))

    # prompt eval time = X ms / N tokens (Y ms per token, Z tokens per second)
    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens.*?([\d.]+)\s+tokens per second", output)
    if m:
        metrics["prompt_time_ms"] = float(m.group(1))
        metrics["prompt_tokens"] = int(m.group(2))
        metrics["prompt_tps"] = float(m.group(3))

    # eval time = X ms / N runs (Y ms per token, Z tokens per second)
    m = re.search(r"eval time\s*=\s*([\d.]+)\s+ms\s*/\s*(\d+)\s+runs.*?([\d.]+)\s+tokens per second", output)
    if m:
        metrics["eval_time_ms"] = float(m.group(1))
        metrics["eval_runs"] = int(m.group(2))
        metrics["eval_tps"] = float(m.group(3))

    # total time = X ms / N tokens
    m = re.search(r"total time\s*=\s*([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens", output)
    if m:
        metrics["total_time_ms"] = float(m.group(1))
        metrics["total_tokens"] = int(m.group(2))

    return metrics


def main():
    if not os.path.exists(LLAMA_BIN):
        print(f"ERROR: binary not found: {LLAMA_BIN}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found: {MODEL_PATH}")
        sys.exit(1)

    all_results = []
    for i in range(N_RUNS):
        print(f"\n=== Run {i + 1}/{N_RUNS} ===")
        for j, prompt in enumerate(PROMPTS):
            print(f"  Prompt {j + 1}/{len(PROMPTS)}...", end=" ", flush=True)
            metrics = run_inference(prompt)
            metrics["run_id"] = i
            metrics["prompt_id"] = j
            all_results.append(metrics)
            tps = metrics.get("eval_tps", "N/A")
            print(f"{tps} tok/s")

    # Aggregate
    gen_values = [m["eval_tps"] for m in all_results if "eval_tps" in m]
    if gen_values:
        avg_tps = round(sum(gen_values) / len(gen_values), 2)
        min_tps = round(min(gen_values), 2)
        max_tps = round(max(gen_values), 2)
    else:
        avg_tps = min_tps = max_tps = 0

    summary = {
        "model": MODEL_PATH,
        "max_tokens": MAX_TOKENS,
        "n_runs": N_RUNS,
        "n_prompts": len(PROMPTS),
        "total_samples": len(all_results),
        "timestamp": datetime.now().isoformat(),
        "avg_eval_tps": avg_tps,
        "min_eval_tps": min_tps,
        "max_eval_tps": max_tps,
    }

    print(f"\n=== SUMMARY ===")
    print(f"Average eval: {summary['avg_eval_tps']} tok/s")
    print(f"Min:     {summary['min_eval_tps']} tok/s")
    print(f"Max:     {summary['max_eval_tps']} tok/s")

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "baseline_q8_0.json")
    with open(result_path, "w") as f:
        json.dump({"summary": summary, "runs": all_results}, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
