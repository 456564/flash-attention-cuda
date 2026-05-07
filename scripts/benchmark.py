#!/usr/bin/env python3
"""LLM 推理基准测试 — 固定 prompt，多次运行取平均。
用法: python3 scripts/benchmark.py --model models/qwen2-xxx.gguf --output results/xxx.json
"""

import subprocess
import json
import re
import time
import sys
import os
import argparse
from datetime import datetime

# ==================== 配置 ====================
LLAMA_BIN = "/home/jetson/lm-inference/llama.cpp/build/bin/llama-simple"
N_RUNS = 10          # 每个 prompt 跑多少轮
MAX_TOKENS = 128     # 每次推理最多生成 token 数

PROMPTS = [
    "Explain the concept of Flash Attention in one paragraph.",
    "What is the difference between FP16 and INT8 quantization?",
    "Write a simple CUDA kernel that adds two vectors.",
    "Describe the Ampere GPU architecture in 3 sentences.",
    "What is Grouped Query Attention and why is it used?",
]


def run_inference(model_path: str, prompt: str) -> dict:
    """调 llama-simple 做单次推理，解析性能指标返回 dict。"""
    t0 = time.perf_counter()
    result = subprocess.run(
        [LLAMA_BIN, "-m", model_path, "-p", prompt, "-n", str(MAX_TOKENS)],
        capture_output=True, text=True, timeout=300,
        input=""  # 关 stdin，非交互模式
    )
    elapsed = time.perf_counter() - t0

    output = result.stdout + result.stderr
    metrics = {
        "prompt": prompt[:100],
        "wall_time_s": round(elapsed, 3),
    }

    # llama-simple 输出格式：
    #   "decoded N tokens in X s, speed: Z t/s"
    m = re.search(r"decoded\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s+s,\s+speed:\s+([\d.]+)\s+t/s", output)
    if m:
        metrics["gen_tokens"] = int(m.group(1))
        metrics["gen_time_s"] = float(m.group(2))
        metrics["gen_tps"] = float(m.group(3))

    # "prompt eval time = X ms / N tokens (Y ms per token, Z tokens per second)"
    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens.*?([\d.]+)\s+tokens per second", output)
    if m:
        metrics["prompt_time_ms"] = float(m.group(1))
        metrics["prompt_tokens"] = int(m.group(2))
        metrics["prompt_tps"] = float(m.group(3))

    # "eval time = X ms / N runs (Y ms per token, Z tokens per second)"
    m = re.search(r"eval time\s*=\s*([\d.]+)\s+ms\s*/\s*(\d+)\s+runs.*?([\d.]+)\s+tokens per second", output)
    if m:
        metrics["eval_time_ms"] = float(m.group(1))
        metrics["eval_runs"] = int(m.group(2))
        metrics["eval_tps"] = float(m.group(3))

    # "total time = X ms / N tokens"
    m = re.search(r"total time\s*=\s*([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens", output)
    if m:
        metrics["total_time_ms"] = float(m.group(1))
        metrics["total_tokens"] = int(m.group(2))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="LLM 推理基准测试")
    parser.add_argument("--model", required=True, help="GGUF 模型路径")
    parser.add_argument("--output", required=True, help="结果 JSON 输出路径")
    parser.add_argument("--runs", type=int, default=N_RUNS, help="每个 prompt 跑几轮")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(LLAMA_BIN):
        print(f"ERROR: 找不到 llama-simple: {LLAMA_BIN}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"ERROR: 找不到模型文件: {model_path}")
        sys.exit(1)

    print(f"模型: {model_path}")
    print(f"输出: {output_path}")
    print(f"轮数: {args.runs}")

    all_results = []
    for i in range(args.runs):
        print(f"\n=== 第 {i + 1}/{args.runs} 轮 ===")
        for j, prompt in enumerate(PROMPTS):
            print(f"  Prompt {j + 1}/{len(PROMPTS)}...", end=" ", flush=True)
            metrics = run_inference(model_path, prompt)
            metrics["run_id"] = i
            metrics["prompt_id"] = j
            all_results.append(metrics)
            tps = metrics.get("eval_tps", "N/A")
            print(f"{tps} tok/s")

    # 汇总统计
    gen_values = [m["eval_tps"] for m in all_results if "eval_tps" in m]
    if gen_values:
        avg_tps = round(sum(gen_values) / len(gen_values), 2)
        min_tps = round(min(gen_values), 2)
        max_tps = round(max(gen_values), 2)
    else:
        avg_tps = min_tps = max_tps = 0

    summary = {
        "model": model_path,
        "max_tokens": MAX_TOKENS,
        "n_runs": args.runs,
        "n_prompts": len(PROMPTS),
        "total_samples": len(all_results),
        "timestamp": datetime.now().isoformat(),
        "avg_eval_tps": avg_tps,
        "min_eval_tps": min_tps,
        "max_eval_tps": max_tps,
    }

    print(f"\n=== 基准测试结果 ===")
    print(f"平均 eval: {summary['avg_eval_tps']} tok/s")
    print(f"最低:     {summary['min_eval_tps']} tok/s")
    print(f"最高:     {summary['max_eval_tps']} tok/s")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "runs": all_results}, f, indent=2)
    print(f"\n结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
