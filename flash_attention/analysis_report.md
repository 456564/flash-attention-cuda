# Flash Attention CUDA Kernel: 从零实现到性能分析

**Jetson Orin Nano Super (Ampere sm_87, 8 SM, 1.02 TFLOPS, 32 GB/s HBM)**  
**刘鑫凯 | 2026.05**

---

## 1. 项目概述

独立实现 CUDA Flash Attention tiled kernel，覆盖单 head → 多 head GQA → 变长 KV cache 完整链路。  
对比 llama.cpp `fattn-tile.cuh` 生产代码做工程分析。Nsight Compute 量化 profiling + Roofline 定位瓶颈。

## 2. 实现架构

### 2.1 算法核心

```
Flash Attention = Q SRAM 常驻 + K/V 分块复用 + online softmax

SRAM 分区 (6 KB):
  Q_tile(Br=32×32 half2 = 2KB)  ← 常驻
  KV_tile(Bc=64×32 half2 = 4KB) ← 分块: 先装K→QK点积→覆盖装V→VKQ累加

online softmax:
  new_max > old_max → scale = exp(old_max - new_max) → VKQ 和 KQ_sum 等比贬值
  数学: exp(dot-new) = exp(dot-old) × exp(old-new)  ← 指数性质
```

### 2.2 实现阶段

| Step | 内容 | 验证 |
|------|------|------|
| CPU 参考 | 三重循环 + causal mask (j≤i) | 基准答案 |
| Naive GPU | <<<1,256>>>, 全从 HBM 读, half2 点积 | 误差 0.000015 |
| Flash 单 head | tiled SRAM + online softmax | 误差 0.000015 |
| 多 head + GQA | 14 Q heads × 2 KV heads, blockIdx.z 并行 | 14 head 全对 |
| KV cache 变长 | kv_max 参数, kv 循环动态截断 | kv=128/256 均正确 |

### 2.3 与 llama.cpp 对比

逐段对照 `fattn-tile.cuh` (1114 行)，10 维差异：线程模型(1D→2D warp)、KQ 存储(寄存器→SRAM)、stride 布局(flat→nb** 字节步长)、mask(硬编码因果→外部数组+ALiBi)、GQA 列并行等。核心算法链路完全一致。

## 3. 性能分析

### 3.1 Nsight Compute 数据 (单 head, basic set)

| 指标 | Naive | Flash | 变化 |
|------|-------|-------|------|
| Grid | 1×256 | 8×32 | 用满 8 SM |
| Compute Throughput | 1.61% | 3.94% | +2.4× |
| Memory Throughput | 9.62% | 25.87% | SRAM 贡献 |
| L1/TEX Throughput | 77.0% | 26.0% | **−3×** (核心证据) |
| Theoretical Occ | 100% | 25% | SRAM 12KB/block 限制 |
| Achieved Occ | 12.2% | 2.1% | kernel 过小 |

**关键发现**: Naive 的 L1 77% → HBM→L1 数据搬运是主瓶颈。Flash 用 SRAM 绕过 L1→HBM 路径，L1 降到 26%。

### 3.2 Roofline 定位

```
算术密度:
  Naive: 1 FLOP/byte     → 贴 HBM 带宽屋顶线 (32 GB/s)
  Flash: ~60 FLOP/byte   → 越过临界点 (32 FLOP/byte)

硬件临界点: 1020 GFLOPS / 32 GB/s = 32 FLOP/byte
  AI < 32 → memory bound
  AI > 32 → compute/occ bound

结论: Naive memory bound, Flash 因 occupancy 受限 (2% achieved)
      在更大 SEQ_LEN 下 Flash 会表现出 compute bound
```

### 3.3 计时基准

| Kernel | 平均耗时 | 加速比 |
|--------|---------|--------|
| Naive (单 head) | 2.67 ms | 1× |
| Flash (单 head) | ~0.004 ms | ~700× |
| Flash (14 head) | ~0.0003 ms/head | — |

## 4. 工程细节

### 4.1 Tile 选型因果链

Br=32: warp=32 线程, 每线程 1 行 Q → 零线程浪费。  
Bc=64: Q_tile(2KB)+KV_tile(4KB)=6KB < 48KB → 留余量给调度器。256÷64=4 次循环, 搬运量合理。

### 4.2 GQA (Grouped Query Attention)

Q 14 heads × 64 = 896, K/V 2 heads × 64 = 128 (GQA ratio=7)。  
`head_kv = blockIdx.z / 7`: head 0..6 → KV₀, head 7..13 → KV₁。  
KV cache 省 7× 显存, 精度损失可忽略。

### 4.3 Occupancy 受限根因

SRAM 12 KB/block → 128 KB/SM ÷ 12 = 每 SM 最多 10 block → 10 warp/48 = 25% theoretical。  
Achieved 仅 2% — kernel 计算量太小, SM 调度器来不及填充 warp 槽位。  
在 SEQ_LEN≥1024、nheads=14 的生产场景下自然提升。

### 4.4 ncu 兼容性修复

Jetson ncu (2024.3.1) 对 `extern __shared__` 存在 launch config 丢失 bug。  
修复: `extern __shared__ half2 sram[]` → `__shared__ half2 sram[(Br+Bc)*(HEAD_DIM/2)]` (编译期固定大小), launch 去第三个参数。

## 5. 技术栈与能力

- **CUDA**: kernel 编写, shared memory 管理, half2 向量化, 线程协作
- **Nsight Compute**: SpeedOfLight / Occupancy / Workload 指标解读
- **Roofline Model**: 算术密度 → memory/compute/occ bound 三段判断
- **llama.cpp 源码**: fattn-tile.cuh / fattn-mma-f16.cuh 阅读与对比分析
- **GQA / KV cache**: 推理引擎核心机制的 kernel 级实现验证
