# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 开发工作流

板子：Jetson Orin Nano Super，VSCode SFTP 同步。开发在本地 PC，代码通过 SFTP 推到 `/home/jetson/lm-inference/`。

```bash
# 本地 → 板子：VSCode Ctrl+Shift+P → SFTP: Sync Local -> Remote
# 板子 → 本地：VSCode Ctrl+Shift+P → SFTP: Sync Remote -> Local
```

板子环境：JetPack 6 (R36.5.0), CUDA 12.6 (sm_87), cmake 3.22.1, g++ 11.4.0, Python 3.10.12。

## 常用命令

```bash
# 基准测试
cd ~/lm-inference && python3 scripts/benchmark.py
# 结果：results/baseline_*.json

# 单次推理（调试用）
echo "Hello" | ~/lm-inference/llama.cpp/build/bin/llama-simple \
  -m ~/lm-inference/models/qwen2-0_5b-instruct-q8_0.gguf -n 32

# 量化模型
~/lm-inference/llama.cpp/build/bin/llama-quantize \
  /path/to/model-fp16.gguf /path/to/output-q4_k_m.gguf Q4_K_M

# 重编译 llama.cpp（改代码后）
cd ~/lm-inference/llama.cpp && cmake --build build --config Release -j$(nproc)
```

## 关键坑

- **llama-cli 默认交互模式**，子进程调用会卡住。用 `llama-simple` 做非交互推理。
- **路径注意**：远端是 `lm-inference`（没有双 L），不是 `llm-inference`。
- **模型下载**：HuggingFace 速度慢，用 ModelScope 直链 wget。

## 项目结构

```
├── scripts/benchmark.py    # 基准测试（llama-simple, regex 解析 tok/s）
├── results/                # 基准数据 JSON
├── llama.cpp/              # (远端) git clone 的推理框架
├── models/                 # (远端) GGUF 模型文件
├── flash_attention/        # 第3周：手写 CUDA kernel
├── quant/                  # 第2周：量化分析
└── nsight/                 # 第4周：性能分析
```

## 目标

在 Orin Nano Super 8GB 上部署 Qwen2-0.5B，用 CUDA 算子优化做到极致 token/s。
输出招聘外企（Qualcomm/Intel/ARM/NVIDIA）的面试级项目素材。

## 硬件

- Jetson Orin Nano Super 8GB
- 1024 CUDA Cores, 32 Tensor Cores, Ampere 架构
- Shared Memory: 128KB/SM
- INT8 dp4a 指令支持

## 项目路线（4-5周）

### 第1周：Baseline ✅
- [x] llama.cpp 编译适配 Orin (JetPack 6) — CUDA 12.6, sm_87, VRAM 7619 MiB
- [x] Qwen2-0.5B-Instruct Q8_0 跑通 (等效 FP16)
- [x] 记录：avg **74.2 tok/s**, min 64.7, max 82.68
- [x] benchmark 脚本 (llama-simple, 5 prompts × 10 runs)

### 第2周：INT8 量化
- [ ] Qwen2-0.5B INT8 量化（llama.cpp 内置或自定义）
- [ ] 对比 FP16 vs INT8: token/s, 显存, 困惑度
- [ ] 精度损失分析表

### 第3周：手写 Flash Attention CUDA Kernel
- [ ] 理解 llama.cpp 原生 attention 实现
- [ ] 手写 Flash Attention（简化版）：tile QKV, SRAM resident, online softmax
- [ ] tile 选型分析：16x16 vs 32x8 vs 32x16，为何选当前值
- [ ] 集成到 llama.cpp，替换原生 attention
- [ ] 对比：原生 attention vs Flash Attention — token/s, 显存

### 第4周：nsight 分析
- [ ] nsight systems 全流程 profile（GPU util, mem BW）
- [ ] nsight compute kernel 级分析（occupancy, bank conflict, compute/memory bound）
- [ ] Roofline model
- [ ] 写分析报告

### 第5周（可选）：收尾
- [ ] GitHub README 中英文
- [ ] 性能对比表：FP16 vs INT8 vs Flash Attention
- [ ] OpenVINO 对比（Intel 面试用）
- [ ] 面试 PPT / 演讲提纲

## 面试对位

| 优化点 | 能讲的 | 命中公司 |
|--------|--------|---------|
| Flash Attention tile 选择 | shared memory size, occupancy, bank conflict | NVIDIA, ARM |
| INT8 量化精度分析 | 量化误差来源, per-channel vs per-tensor | Qualcomm, Intel |
| nsight roofline | compute bound vs memory bound 判断 | 全部 |
| llama.cpp 集成 | 框架源码修改, 性能收益 | 全部 |

## 个人

- 刘鑫凯，大三（2027届），端侧AI方向
- 目标外企实习：Qualcomm/Intel/ARM/NVIDIA
- 暑假前完成投递，6/30 截止
