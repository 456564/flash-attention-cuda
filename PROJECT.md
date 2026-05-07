# Jetson Orin Nano Super — LLM 推理优化项目

## 目标

在 Orin Nano Super 8GB 上部署 Qwen2-0.5B，用 CUDA 算子优化做到极致 token/s。
输出招聘外企（Qualcomm/Intel/ARM/NVIDIA）的面试级项目素材。

## 硬件

- Jetson Orin Nano Super 8GB
- 1024 CUDA Cores, 32 Tensor Cores, Ampere 架构
- Shared Memory: 128KB/SM
- INT8 dp4a 指令支持

## 项目结构

```
├── CLAUDE.md                # 协作规则和常用命令
├── PROJECT.md               # 本文件：项目详情
├── scripts/benchmark.py     # 基准测试
├── results/                 # 基准数据 JSON
├── llama.cpp/               # (远端) 推理框架
├── models/                  # (远端) GGUF 模型
├── flash_attention/         # 第3周：手写 CUDA kernel
├── quant/                   # 第2周：量化分析
├── nsight/                   # 第4周：性能分析
└── docs/                    # 第5周：文档和面试材料
```

## 项目路线（4-5周）

### 第1周：Baseline ✅
- [x] llama.cpp 编译适配 Orin (JetPack 6) — CUDA 12.6, sm_87, VRAM 7619 MiB
- [x] Qwen2-0.5B-Instruct Q8_0 跑通 (等效 FP16)
- [x] 记录：avg **74.2 tok/s**, min 64.7, max 82.68
- [x] benchmark 脚本 (llama-simple, 5 prompts × 10 runs)

| 模型 | 大小 | avg tok/s | min | max |
|------|------|-----------|-----|-----|
| Q8_0 | 507M | 74.2 | 64.7 | 82.7 |

### 第2周：INT8 量化 ✅
- [x] FP16 → Q4_0 / Q4_K_M 量化
- [x] 基准测试对比
- [x] 困惑度测试（perplexity）
- [x] 精度损失分析报告 → `quant/quantize_analysis.md`

| 模型 | 大小 | tok/s | PPL | vs Q8_0 速度 | PPL 损失 |
|------|------|-------|-----|-------------|---------|
| Q8_0 | 507M | 74.2 | 1.0282 | baseline | baseline |
| Q4_K_M | 380M | 77.37 | 1.0317 | +4.3% | +0.34% |
| Q4_0 | 336M | 83.18 | 1.0448 | +12.1% | +1.61% |
| Q4_0 | 336M | 83.18 | +12.1% | 3-5% |
| Q4_K_M | 380M | 77.37 | +4.3% | 1-2% |

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
