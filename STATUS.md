# Jetson Orin Nano Super — LLM 推理优化项目状态

## 项目目标

在 Jetson Orin Nano Super 8GB 上部署 Qwen2-0.5B，用 CUDA 算子优化做到极致 token/s。产出面试级项目素材，面向 Qualcomm/Intel/ARM/NVIDIA 外企实习。

## 全局规则

全局 CLAUDE.md (`C:\Users\888\.claude\CLAUDE.md`)：
- 变量/函数名英文，注释中文
- 板子操作前标注 `[需要同步]` / `[无需同步]`
- 大文件优先国内源
- 讲解用因果关系链：因为XX→所以YY→要做ZZ
- 项目文件 CLAUDE.md(稳定) + PROGRESS.md(动态)

## 项目文件架构

```
D:\download\code\jetson orin nano super\
├── CLAUDE.md          # 稳定层：规则、环境、命令、坑
├── PROGRESS.md        # 动态层：路线图、进度、基准数据、面试矩阵
├── scripts/benchmark.py   # 基准测试（argparse传参）
├── results/           # 基准数据 JSON
├── quant/quantize_analysis.md  # 第2周量化分析
├── flash_attention/           # 第3周
│   ├── flash_attn_mini.cu     # 完整可工作的 Flash Attention (我写的参考)
│   ├── flash_attn_mini_s1.cu  # 用户手写版 (Step 1-3 完成，Step 4 一半)
│   ├── tile_analysis.md       # tile 选型分析
│   └── LEARNING_LOG.md        # 学习档案
├── llama.cpp/       # (远端) 推理框架
└── models/          # (远端) GGUF 模型
```

## 工作流

- 本地 PC 开发
- 板子：Jetson Orin Nano Super, JetPack 6, CUDA 12.6 sm_87
- VSCode SFTP 同步：`/home/jetson/lm-inference/`
- Sync Local→Remote 推送代码，Sync Remote→Local 拉回结果
- 用户是 C++/CUDA 初学者，需要逐步讲解

## 硬件规格（板上 cudaGetDeviceProperties 实测）

- GPU: Orin (Ampere)
- SMs: 8
- Max Threads/SM: 1536
- Warp: 32
- Shared Memory: 128 KB/SM, 48 KB/Block (默认)
- Registers: 65536/Block
- Global Memory: 7619 MiB
- Core Clock: 1.02 GHz
- Memory: 128-bit bus, 1020 MHz, LPDDR5

## 已完成的基准数据

### 第 1 周 Baseline (已完成)
- Qwen2-0.5B Q8_0: avg 74.2 tok/s, min 64.7, max 82.68
- 5 prompts × 10 runs benchmark

### 第 2 周 INT8 量化 (已完成)

| 模型 | 大小 | tok/s | PPL | vs Q8_0速度 | PPL损失 |
|------|------|-------|-----|------------|--------|
| Q8_0 | 507M | 74.2 | 1.0282 | baseline | baseline |
| Q4_K_M | 380M | 77.37 | 1.0317 | +4.3% | +0.34% |
| Q4_0 | 336M | 83.18 | 1.0448 | +12.1% | +1.61% |

- PPL 用中文测试文本 6282 bytes，`llama-perplexity --ctx-size 512`
- 模型参数：896维, 24层, 14 Q heads, 2 KV heads (GQA 7:1)
- FFN 中间维度 4864，词表 151936
- 分析文档：`quant/quantize_analysis.md`

## 第 3 周 Flash Attention (当前，进度约 60%)

### 已完成的理论理解

用户已深入理解以下概念（用因果关系讲解）：

**Attention 基础**：
- QKV 哲学三问：Q=我要找谁，K=我是谁，V=我有什么
- Tokenizer → Embedding → Transformer×24 → Output → softmax → 下一个token
- QK 点积 = 64维逐维乘加求相似度
- scale = 1/√64 防止点积爆炸
- causal mask：用 -1e9f 遮住未来 token
- softmax = exp(分数-max)/sum → 微小差异放大为概率分配
- exp 物理意义 = 以 e 为基准的连续放大器（不是离散的翻倍 2）
- VKQ/sum = 加权平均所有 token 的 V → 综合理解

**Flash Attention 原理**：
- HBM vs SRAM vs 寄存器：速度差 10-40 倍
- 瓶颈在搬运，不在计算
- Q→SRAM 常驻，K/V 从 HBM 分块搬进 SRAM，用完覆盖
- online softmax：旧max < 新max → 旧 sum × exp(旧max-新max) 贬值

**硬件认知**：
- half2 = 一次读 2 个 FP16，带宽翻倍
- cudaMalloc = GPU HBM, new = CPU 堆
- `__global__` = GPU kernel, `<<<grid,block,SRAM>>>` = 启动配置
- `extern __shared__` = 运行时分配 SRAM

### 代码进度 (flash_attn_mini_s1.cu)

**Step 1 ✅** — 数据初始化
- CPU float 数组生成随机 QKV，half 临时数组转 FP16
- cudaMalloc + cudaMemcpy 搬到 GPU

**Step 2 ✅** — CPU 参考 attention
- 三重循环 (i,j,d)，因果 mask 用 j≤i 优化
- O_h 为标准答案

**Step 3 ✅** — 朴素 GPU kernel
- `naive_attn_kernel<<<1,256>>>`，每线程处理 1 token
- half2 点积（head_dim_half=32 次循环替代 64 次）
- 误差 0.000015

**Step 4 🔄** — Tiled Flash Attention kernel (写了一半)
- `flash_attn_kernel` 声明 OK
- `extern __shared__` + Q_tile/KV_tile 分区 OK
- Q_tile 协作加载 OK（32线程 × 32元素）
- Br=32, Bc=64
- **未完成**：online softmax 累加器初始化、K/V 分块循环、V 加权累加

### 完整参考 kernel 在哪

`flash_attn_mini.cu` — 已调通，板上 0 误差运行：
- SRAM 占用 12 KB
- `nvcc -arch=sm_87 -O3` 编译验证过
- CuDA kernel + CPU 参考完整实现

## 用户特点

- 大三学生（2027届），端侧 AI 方向
- 中文母语，C++/CUDA 初学者
- 目标外企实习：Qualcomm/Intel/ARM/NVIDIA，暑假前投递，6/30 截止
- 偏好：因果关系讲解（因为XX→所以YY→要做ZZ），不要背书，要理解物理意义
- 当前状态：已学大量内容，有点学不动了，可能需要换项目或换方式

## 当前 git 状态

```
master 分支
ee350ee docs: learning log for Step 1 flash attention
87686c7 docs: Orin device specs from cudaGetDeviceProperties
6a8e1d2 feat: tile analysis doc + fix KQ warning
9304453 feat: Week 3 mini Flash Attention kernel (standalone)
36eed47 refactor: align with global CLAUDE.md structure
```

未提交：flash_attn_mini_s1.cu 的最新修改（Step 4 部分）
