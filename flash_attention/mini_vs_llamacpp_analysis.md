# Flash Attention: mini 手写版 vs llama.cpp 生产版 逐段对照

## 总览

两个 kernel 执行**完全相同的算法**：
1. Q 常驻 SRAM
2. K/V 分块循环
3. online softmax
4. VKQ 归一化输出

差异在**工程维度**：多 head、GQA、warp 二维调度、mask、stride 布局。

---

## 1. SRAM 分区

### mini (flash_attn_mini_s1.cu:65-67)
```cpp
extern __shared__ half2 sram[];
half2 * Q_tile = sram;                          // 前半：32×32 = 1024 half2
half2 * KV_tile = sram + Br * head_dim_half;     // 后半：64×32 = 2048 half2
```

### llama.cpp (fattn-tile.cuh:854-858)
```cpp
__shared__ half2 Q_tmp[ncols * DKQ/2];
__shared__ half2 KV_tmp[nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV];
__shared__ half  KQ[ncols * nbatch_fa];  // ← 额外的 SRAM 缓冲区！
```

**差异**：
- llama.cpp 多了 `KQ[]` 缓冲区：QK 点积结果先全存入 SRAM，再统一做 softmax + V 累加。mini 版每个线程的 `KQ_local[64]` 在寄存器里。
- `KV_tmp` 多了 `cpy_ne` 和 `DVp-DV` padding：防止 bank conflict 和越界。

---

## 2. Q 加载

### mini (72-79)
```cpp
for (int idx = tid; idx < Br * head_dim_half; idx += Br) {
    int row = idx / head_dim_half;
    int col = idx % head_dim_half;
    Q_tile[idx] = Q[(q_start + row) * head_dim_half + col];
}
```

### llama.cpp (874-917)
```cpp
for (int jc0 = 0; jc0 < cpw; ++jc0) {         // cpw = 每 warp 负责几个 Q 列
    for (int i0 = 0; i0 < DKQp; i0 += ...) {  // 沿 head_dim 方向分段搬运
        float tmp_f[cpy_ne_D] = {0.0f};
        // 从 Q 全局内存读到寄存器 tmp_f
        // scale 直接乘进 Q（Q *= scale，省后续乘法）
        // float → half2 转换
        Q_tmp[jc*(DKQ/2) + i0/2 + ...] = tmp_h2;
    }
}
```

**差异**：
- mini：一行 Q 只搬 32 个 half2 = 1 列，简单直接
- llama.cpp：两重循环覆盖多个 head 列（ncols），**scale 乘进 Q 不是乘 dot**——省了每条 QK 点积的乘法指令
- llama.cpp：用 `threadIdx.y` 做 warp 级并行（二维线程块）

---

## 3. K 加载

### mini (96-101)
```cpp
for (int idx = tid; idx < kv_len * head_dim_half; idx += Br){
    int row = idx / head_dim_half;
    int col = idx % head_dim_half;
    KV_tile[idx] = K[(kv_start + row) * head_dim_half + col];
}
__syncthreads();
```

### llama.cpp (474-476, fattn-tile_iter_KQ)
```cpp
flash_attn_tile_load_tile<...>(
    K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2,
    KV_tmp, stride_K2, k_VKQ_sup);
__syncthreads();
```

**差异**：
- mini：直接算 row/col，`K[全局偏移]`
- llama.cpp：封装 `flash_attn_tile_load_tile`，用 stride（字节步长）计算地址
- stride 原因：ggml tensor 内存布局可能有 padding，`K[i+1]` 不一定紧挨 `K[i]`

---

## 4. QK 点积

### mini (106-120)
```cpp
for(int j = 0; j < kv_len; j++){
    float dot = 0;
    for(int d = 0; d < head_dim_half; d++){
        half2 qv = Q_tile[my_row * head_dim_half + d];
        half2 kv = KV_tile[j * head_dim_half + d];
        float2 qf = __half22float2(qv);
        float2 kf = __half22float2(kv);
        dot += qf.x * kf.x + qf.y * kf.y;
    }
    dot *= scale;
    if (kv_start + j > q_start + my_row) dot = -1e9f;
    KQ_local[j] = dot;
    if (dot > new_max) new_max = dot;
}
```

### llama.cpp (582-589 → 458-528, fattn_tile_iter_KQ)
```cpp
// DKQ 维度也分块（nbatch_K），不是一次算完 64 维
for (int k_KQ_0 = 0; k_KQ_0 < DKQ; k_KQ_0 += nbatch_K) {
    // 1. 加载 K 列片段到 KV_tmp
    // 2. __syncthreads()
    // 3. 寄存器加载 K_k[nbatch_fa][cpy_ne] 和 Q_k[cpw][cpy_ne]
    // 4. 三重循环：i_KQ × jc × k → KQ_acc += K_k[i][k] * Q_k[j][k]
}
// KQ_acc 存所有分数，再统一 apply mask + softmax
```

**差异**：
- mini：每个线程独立串行算 64 维点积 → 32 × 点积并行（线程级）
- llama.cpp：Q 和 K 的 head_dim 方向也分块（nbatch_K），块内用向量化 `ggml_cuda_mad` 做乘加
- mini：分数存在寄存器 `KQ_local[j]`（每线程 64 float）
- llama.cpp：分数存在 SRAM `KQ[ncols * nbatch_fa]`（所有线程共享）

### 为什么 llama.cpp 把 KQ 放 SRAM 而不是寄存器？
**因为** 多 head 共享同一组 K/V（GQA），KQ 矩阵要被所有 head 列共享 → 放 SRAM 让 warp 间可见。mini 版单 head 单线程，寄存器够用。

---

## 5. Mask

### mini (116-117)
```cpp
if (kv_start + j > q_start + my_row) dot = -1e9f;
```

### llama.cpp (611-613)
```cpp
KQ_acc[...] += (ncols2 > 1 || mask) ?
    slope * __half2float(mask[j*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;
```

**差异**：
- mini：硬编码因果 mask（`j > i`）
- llama.cpp：外部传入 mask 数组，支持 ALiBi（slope 因子），支持 padding mask、sliding window 等

---

## 6. Online Softmax

### mini (122-134)
```cpp
if(new_max > KQ_max){
    float old_scale = expf(KQ_max - new_max);
    for(int d = 0; d < HEAD_DIM; d++)
        VKQ[d] *= old_scale;
    KQ_sum *= old_scale;
    KQ_max = new_max;
}
for (int j = 0; j < kv_len; j++){
    float score = expf(KQ_local[j] - KQ_max);
    KQ_sum += score;
    KQ_local[j] = score;
}
```

### llama.cpp (648-673)
```cpp
const float KQ_max_scale = expf(KQ_max[jc] - KQ_max_new[jc]);
KQ_max[jc] = KQ_max_new[jc];

// 贬值 VKQ（每个 jc 列独立贬值）
for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
    VKQ[jc*...] *= KQ_max_scale_h2;   // half2 一次性贬值
}

// 计算 exp + 累加 KQ_sum
for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
    float val = expf(KQ_acc[...] - KQ_max[jc]);
    KQ_sum_add += val;
    tmp[...][jc1] = val;   // 存到 SRAM KQ 缓冲区
}
KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;
```

**差异**：
- mini：VKQ 贬值用 float 循环 64 次
- llama.cpp：VKQ 贬值用 half2 向量化（一步贬两个），且**每个 jc 列独立贬值**（多 head 时每列 max 不同）
- llama.cpp：exp 结果写入 SRAM `KQ[]`，供后续 VKQ 累加用（多 warp 共享）

---

## 7. V 加载 + VKQ 累加

### mini (137-153)
```cpp
// V 加载（覆盖 KV_tile）
for (int idx = tid; idx < kv_len * head_dim_half; idx += Br)
    KV_tile[idx] = V[(kv_start + row) * head_dim_half + col];
__syncthreads();

// VKQ 累加
for (int j = 0; j < kv_len; j++){
    for (int d = 0; d < head_dim_half; d++){
        half2 vv = KV_tile[j * head_dim_half + d];
        float2 vf = __half22float2(vv);
        VKQ[2*d]   += KQ_local[j] * vf.x;
        VKQ[2*d+1] += KQ_local[j] * vf.y;
    }
}
```

### llama.cpp (693-761)
```cpp
// V 也分块加载（nbatch_V），因为 V 矩阵可能一次放不下
for (int k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
    flash_attn_tile_load_tile<...>(V_h2 + ..., KV_tmp, ...);
    __syncthreads();

    for (int k1 = 0; k1 < nbatch_V; k1 += np) {
        // 从 KV_tmp 读到 V_k 寄存器
        // 从 KQ SRAM 读到 KQ_k 寄存器
        // VKQ += V_k * KQ_k   (half2 向量乘加)
        VKQ[jc*...] += V_k[i]*KQ_k[jc];
    }
    __syncthreads();
}
```

**差异**：
- mini：V 一次加载全部 kv_len 行，每个线程独立遍历 j × d
- llama.cpp：V 也分块（nbatch_V < nbatch_fa 时），V 和 KQ 以 half2 向量形式做乘加
- mini：KQ_local 在寄存器里
- llama.cpp：KQ 从 SRAM 读到寄存器再参与乘加

---

## 8. 归一化输出

### mini (155-158)
```cpp
for (int d = 0; d < HEAD_DIM; d++)
    O[(q_start + my_row) * HEAD_DIM + d] = VKQ[d] / KQ_sum;
```

### llama.cpp
归一化在 kernel 外（返回 `VKQ` 和 `KQ_sum` 给 host），由后续 kernel 做除法。

**差异**：
- mini：kernel 内直接除完写入 O
- llama.cpp：分离式设计——VKQ 累加器和 sum 都输出到 `dst_meta`，由单独 reduction kernel 做除法。原因：支持多 block 协作处理同一行 Q（需要跨 block 合并 VKQ 和 sum）

---

## 9. 线程模型对比

| | mini | llama.cpp |
|---|------|-----------|
| 线程维度 | 1D (`threadIdx.x`) | 2D (`threadIdx.x` + `threadIdx.y`) |
| block 大小 | Br=32 线程 | 64~256 线程 |
| 每线程职责 | 1 行 Q | cpw 列 Q（1~32 列） |
| warp 分工 | 无 | `threadIdx.y` 分 warp，`np` 控制并行度 |
| KQ 存储 | 寄存器 `KQ_local[64]` | SRAM `KQ[ncols * nbatch_fa]` |

---

## 10. 你的 mini 版没处理但生产版必须处理的

| 问题 | llama.cpp 方案 |
|------|---------------|
| 多 head（Qwen 有 14 head） | `ncols = ncols1 * ncols2`，每 block 处理多列 |
| GQA（Q head 多于 KV head） | `gqa_ratio = ne02/ne12`，多个 Q 列共享同组 K/V |
| 可变序列长度（生成时逐 token 增长） | `KV_max` 动态传入，`oob_check` 分支处理尾部 |
| ALiBi 位置编码 | `slope` 参数，mask 值乘以 slope |
| logit softcap | `tanhf` 压缩极值，防止 overflow |
| FP32 精度路径 | `#ifdef FAST_FP16_AVAILABLE` 双路径编译 |
| 量化 K/V（Q4_0, Q8_0 等） | `vec_dot_KQ_t` 函数指针，按量化类型分发 |
| Bank conflict | `KV_tmp` 加 `cpy_ne` padding |
| 跨 block 合并 | VKQ + sum 输出到 `dst_meta`，单独 reduction kernel |

---

## 结论

你的 mini 版是 llama.cpp fattn-tile 的**单 head、定长、纯 FP16、单 block** 精简版。核心算法（tile、online softmax、KV_tile 复用）完全一致。理解 mini 版后再读 llama.cpp，差异只有"工程泛化"——把 1 变 N。
