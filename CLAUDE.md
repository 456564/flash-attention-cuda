# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 协作规则

1. **变量/函数名英文，注释中文**
2. **每步标注同步**：板子操作前标注 `[需要同步]` 或 `[无需同步]`，说明原因
3. **优先国内源**：下载大文件用 ModelScope、hf-mirror、清华镜像。国外快时可直接用
4. **只聚焦本项目**：不读 career-ops、esp32s3-minist4 等无关目录

## 开发工作流

板子：Jetson Orin Nano Super (JetPack 6, CUDA 12.6 sm_87)，VSCode SFTP 同步。

开发在本地 PC，代码推到 `/home/jetson/lm-inference/`。
- 本地 → 板子：`Ctrl+Shift+P` → `SFTP: Sync Local -> Remote`
- 板子 → 本地：`Ctrl+Shift+P` → `SFTP: Sync Remote -> Local`

## 常用命令

```bash
# 基准测试
cd ~/lm-inference && python3 scripts/benchmark.py --model models/xxx.gguf --output results/xxx.json

# 单次推理（调试用，非交互）
echo "Hello" | ~/lm-inference/llama.cpp/build/bin/llama-simple \
  -m ~/lm-inference/models/qwen2-0_5b-instruct-q8_0.gguf -n 32

# 量化模型
~/lm-inference/llama.cpp/build/bin/llama-quantize \
  /path/to/model-fp16.gguf /path/to/output-q4_k_m.gguf Q4_K_M

# 重编译 llama.cpp
cd ~/lm-inference/llama.cpp && cmake --build build --config Release -j$(nproc)
```

## 关键坑

- **llama-cli 默认交互模式**，子进程调用会卡住。用 `llama-simple` 做非交互推理
- **路径注意**：远端是 `lm-inference`（没有双 L），不是 `llm-inference`
- **量化源必须是 FP16**，不能从 Q8_0 二次量化为 Q4_0

## 项目详情

详见 [PROJECT.md](PROJECT.md)
