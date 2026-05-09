# Flash Attention Tile 选型分析

## 环境

- Orin Nano Super 8GB, 8 SMs, 1536 threads/SM, warp=32
- Shared Memory: 128 KB/SM, 48 KB/Block (默认)
- Qwen2-0.5B: head_dim=64, 14 Q heads, 2 KV heads (GQA 7:1)

## SRAM 占用公式

```
sram_usage = Br × head_dim_half × sizeof(half2)   # Q tile
           + Bc × head_dim_half × sizeof(half2)   # KV tile
           = Br × 32 × 4 + Bc × 32 × 4
           = (Br + Bc) × 128 bytes

half2 = 4 bytes, head_dim_half = 64/2 = 32
```

## 候选 tile 方案

| Br (Q tile) | Bc (KV tile) | SRAM 占用 | 占 128KB | occupancy |
|-------------|-------------|-----------|---------|-----------|
| 16 | 64 | 10.0 KB | 7.8% | 4+ |
| 32 | 64 | 12.0 KB | 9.4% | 4+ |
| 32 | 128 | 20.0 KB | 15.6% | 4 |
| 64 | 64 | 16.0 KB | 12.5% | 4 |
| 64 | 128 | 24.0 KB | 18.8% | 4 |
| 128 | 64 | 24.0 KB | 18.8% | 4 |
| 128 | 128 | 32.0 KB | 25.0% | 4 |
| 256 | 256 | 64.0 KB | 50.0% | 2 |

## 选型分析

### Br 越大 → 一次处理更多 Q 行

Q 块大 = KV 循环次数少 = 减少 HBM→SRAM 搬运
但 Q 块大 = 更多线程空闲（decode 时只有 1 token）

prefill 阶段（batch 大）：Br 大有利
decode 阶段（单 token）：Br 大浪费

### Bc 越大 → 一次处理更多 KV

KV 块大 = 外循环次数少 = 减少 KV 搬运
但 KV 块大 = SRAM 占用多 = occupancy 降低

### 最优 tile 选型逻辑

```
decode (1 token):
  Br = 1 (单 token) — Q 不用分块
  Bc = 可用 SRAM / KV_size_per_row
     = (128KB - Q(0.25KB)) / (64×2bytes) ≈ 1022
  Bc 取 64-128 即可，太大浪费

prefill (N tokens, N > Br):
  Br = 限制因素 — thread 数 / 每行处理量
  Orin 每 SM 最多 1536 threads
  每个 Q 行需要至少 32 threads (一个 warp 做 d=64 的点积)
  Br_max = 1536 / 32 = 48
  取 Br=32 或 64
```

## 当前选择的理由 (Br=32, Bc=64)

```
SRAM: 12 KB — 余量 90%+，允许高 occupancy
Br=32: prefill 时 32 threads 并行处理 Q 行
Bc=64: 128 token 循环 4 次 (256/64)，减少循环开销
       64×64=4096 个元素，2 warp 隐藏延迟

这个选择由以下几点决定：
  1. SM 128 KB → tile 足够小的任意组合都行
  2. warp=32 → nbatch_fa 取 32 的倍数最省计算
  3. Qwen2 的 head_dim=64 → Br=32, Bc=64 刚好是 1-2 warp 覆盖
```

## 面试一句话

"128 KB SRAM 在 head_dim=64 时，12 KB 就能装下 Br=32, Bc=64 的 tile。剩下的 SRAM 全给 occupancy — 多 warp 并行隐藏内存延迟。Flash Attention 真正的瓶颈在 HBM 带宽，不在 SRAM 大小。"
