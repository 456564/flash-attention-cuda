# Jetson Orin Nano Super — LLM 推理优化项目状态

## 项目目标

在 Jetson Orin Nano Super 8GB 上部署 Qwen2-0.5B，CUDA 算子优化 token/s。面向 Qualcomm/Intel/ARM/NVIDIA 外企实习。

## 全局规则

全局 CLAUDE.md (`C:\Users\888\.claude\CLAUDE.md`)：
- 变量/函数名英文，注释中文
- `[需要同步]` / `[无需同步]` 标注板子操作
- 大文件优先国内源
- **因果链讲解**：先因后果 → 再说怎么做。绝对不允许跳步骤（教学事故）
- 项目文件 CLAUDE.md(稳定) + PROGRESS.md(动态)

## 项目文件架构

```
D:\download\code\jetson orin nano super\
├── CLAUDE.md              # 稳定层：规则、环境、命令
├── PROGRESS.md            # 动态层：路线图、基准数据
├── STATUS.md              # 本文件：上下文交接快照
├── scripts/benchmark.py   # 基准测试
├── results/               # 基准 JSON
├── quant/quantize_analysis.md  # 第2周 量化分析
├── flash_attention/       # 第3周
│   ├── flash_attn_mini.cu     # 完整可工作的参考 (我写的)
│   ├── flash_attn_mini_s1.cu  # 用户手写版 (当前文件)
│   ├── tile_analysis.md       # tile 选型分析
│   └── LEARNING_LOG.md        # 学习档案
├── llama.cpp/             # (远端)
└── models/                # (远端)

远端：/home/jetson/lm-inference/
开发：本地 PC，JetPack 6, CUDA 12.6 sm_87
同步：VSCode SFTP
```

## 硬件规格 (cudaGetDeviceProperties)

8 SMs, 1536 threads/SM, warp=32, 128 KB SRAM/SM (48 KB/block 默认)
65536 regs/block, 7619 MiB VRAM, 128-bit bus, 1020 MHz
Core: 1.02 GHz Ampere

## 已完成的基准

### 第 1 周 Baseline ✅
Q8_0: 74.2 tok/s avg

### 第 2 周 INT8 量化 ✅
Q4_K_M: 77.37 tok/s (+4.3%), PPL 1.0317 (+0.34%)
Q4_0:   83.18 tok/s (+12.1%), PPL 1.0448 (+1.61%)

## 第 3 周 Flash Attention — 理论学习 ✅，代码 🔄 60%

### 理论：已完全理解

**Attention 原理**：
- QKV 哲学三问（Q=找谁, K=我是谁, V=我有什么）
- Tokenizer→Embedding→Transformer×24→Output→softmax→next token
- QK 点积 = 64 维逐位乘加求相似度
- scale = 1/√64 防止点积爆炸
- causal mask 用 -1e9f 遮未来
- softmax = exp(x-max)/sum 把微小差异放大为概率分配
- exp 物理意义 = 以 e 为基准的连续放大器（不是 2 的离散翻倍）
- VKQ/sum = 加权平均 = 理解上下文

**Flash Attention 原理**：
- HBM 太慢(300 cycles) → Q 先搬进 SRAM(20 cycles) 常驻
- K/V 太大放不下 SRAM → 分块搬入 KV_tile，用完覆盖
- online softmax: 旧max<新max → 旧sum×exp(旧max-新max) 贬值
- tile 选型: Br=32, Bc=64 是因为 SRAM 48KB + 32 线程对齐 + 循环 4 次平衡

**CUDA 基础**：
- GPU 内存层次：寄存器(0) > SRAM(20) > L2 > HBM(300)
- half2 = 一次读 2 个 FP16，带宽翻倍
- __global__ = GPU kernel, <<<grid,block,sram>>> = 启动配置
- extern __shared__ = 运行时分配 SRAM
- block = 同一 SM 上的线程组，可 __syncthreads()
- warp = 32 线程，硬件同时调度
- 多个 block/SM 可交替隐藏内存延迟

### 代码：flash_attn_mini_s1.cu 当前状态

**Step 1 ✅** — CPU 数据初始化
- float Q_h/K_h/V_h 随机生成，half 临时数组转 FP16 传 GPU
- cudaMalloc + cudaMemcpy

**Step 2 ✅** — CPU 参考 attention
- 三重循环 (i,j,d)，因果 mask 用 j<=i 优化
- O_h = 标准答案

**Step 3 ✅** — 朴素 GPU kernel
- naive_attn_kernel<<<1,256>>>，每线程 1 token
- half2 点积，误差 0.000015

**Step 4 🔄** — Flash Attention kernel (正在写)
- extern __shared__ sram 分区：Q_tile(4KB) + KV_tile(8KB)
- Q_tile 协作加载 ✅ (32 线程 × 32 列 = 1024 元素)
- online softmax 初始化 ✅ (VKQ, KQ_max, KQ_sum)
- K/V 外循环骨架 ✅ (kv_start += Bc)
- **K 加载** ✅ (95-99 行)
- **QK 点积** 🔄 (103-110 行有 bug：缺 j 循环、dot 未声明、half2 直接 .x.y)
- **online softmax** ⬜ 下一步
- **V 加载 + VKQ 累加** ⬜
- **归一化输出** ⬜

### 完整参考在哪

`flash_attn_mini.cu` — 已调通，板子 0 误差运行。

## 当前待修复的代码 (103-110 行)

```cpp
// ---- 当前有 bug 的 QK 点积 ----
for(int d = 0; d < head_dim_half; d++){
    half2 qv = Q_tile[my_row * head_dim_half + d];
    half2 kv = KV_tile[j * head_dim_half + d];  // j 不存在！
    float2 qf = __half22float2(qv);
    float2 kf = __half22float2(kv);
    dot += qv.x * kv.x + qv.y * kv.y;  // half2 不能直接用 .x .y！
}

// 需要改成：
// 1. 外层加 for (int j = 0; j < kv_len; j++)
// 2. 声明 float KQ_local[64] + new_max
// 3. half2 → float2 后访问 qf.x kf.x
// 4. dot 做 scale + causal mask
// 5. 每算一个分数存 KQ_local[j] 并更新 new_max
```

## 用户特点和教学偏好

- 大三学生(2027届)，端侧AI，C++/CUDA 初学者
- 目标外企实习：Qualcomm/Intel/ARM/NVIDIA，6/30 截止
- **必须因果链讲解**：先物理原因→推理出设计→才写代码
- "不要只丢解释，要配合代码逐步因果推理"
- 当前状态：学了很多理论有倦怠感，代码语法层面还在熟悉
- "第一次不会全记住" — 用户接受渐进学习
- "你不要全写，解释下一步就行" — 只解释逻辑，用户自己动手写代码
