# Flash Attention 学习档案

## Step 1: 骨架 + 数据初始化

### 理解的问题

1. **constexpr 是什么？**
   - 编译期常量。值在编译时定死，运行期间不变。相当于类型安全的 #define。

2. **SEQ_LEN 和 HEAD_DIM 的含义？**
   - SEQ_LEN = 256，测试用的序列长度（用户输入动态，上限由显存决定）
   - HEAD_DIM = 64，Qwen2-0.5B 的 head 维度（每个 token 用 64 个数表示）

3. **half 是什么？为什么用？**
   - FP16，2 字节。GPU 原生格式，计算快 2-8 倍。推理不需要 23 位尾数。
   - `__float2half()`: float → half
   - `__half2float()`: half → float

4. **float 约 7 位有效数字是怎么来的？**
   - float = 23 位显式尾数 + 1 位隐含 = 24 位二进制精度
   - log10(2^24) ≈ 7.22 → 约 7 位十进制有效数字
   - 隐含位: IEEE 754 规范形式小数点前永远是 "1."，不存省一位

5. **cudaMalloc 是什么？**
   - GPU 版 new/malloc。在 GPU 全局内存 (HBM) 上分配空间
   - `cudaMalloc(&Q_d, n * sizeof(half))`: 第 2 个参数是**总字节数**，要自己乘数量

6. **cudaMemcpy 用法？**
   - `cudaMemcpy(目标, 源, 字节数, 方向)`
   - 第一个参数始终是目标，第二个始终是源
   - cudaMemcpyHostToDevice = CPU → GPU
   - cudaMemcpyDeviceToHost = GPU → CPU

7. **new half[n] 为什么不自动是 16384？**
   - new 只关心类型大小 × 数量，not 自动知道 n 等于多少
   - n = SEQ_LEN × HEAD_DIM = 256 × 64 = 16384 需要自己算

8. **GPU 内存层次命名**
   - `_h` = host (CPU 侧)
   - `_d` = device (GPU 侧)
   - cudaMalloc 分配的 = GPU global memory，不是堆（堆是 CPU 的概念）

9. **随机数的设计**
   - `rand()/RAND_MAX` → 0~1
   - `-0.5` → -0.5~0.5，中心在 0
   - `*0.1` → -0.05~0.05，小范围模拟真实权重分布
   - 真实 QKV 权重接近 0，避免点积爆炸 → softmax 溢出

### 犯过的错

1. **`srand(42)` 放在 for 循环里**
   - 每轮重置种子 → 每次 rand() 都返回相同序列的第一个数
   - 正确: srand 放 for 前面，只调一次

2. **`srand(42),` 逗号结尾 → 语法错误**
   - c++ 中必须用 `;` 结尾

3. **类型不匹配: float* 当 half* 传给 GPU**
   - Q_h 是 float (4 字节), Q_d 是 half (2 字节)
   - cudaMemcpy 如果抄 float 数据到 half 数组 → GPU 把 float 字节当 half 解析，数值全错
   - 修法: 加 half 临时数组当"格式桥" `__float2half()`

4. **`#include <cstdio.h>` 写错了**
   - C++ 标准库头文件不带 .h：`#include <cstdio>`

### 关键理解

- 在 CPU 上留一份 float 数组 (Q_h float) 当"标准答案"
- GPU 用 half (Q_d half)，搬运前通过临时数组转换
- CPU 参考实现跑完后，跟 GPU kernel 输出对比 → 误差应为 0

---

## Step 2: CPU 参考 Attention

### 理解的问题

1. **CPU attention 为什么用三重循环？**
   - i 循环 = 每个 token 是提问者
   - j 循环 = 每个 token 是被问者
   - d 循环 = 64 维逐位乘加
   - 因果 mask 用 j≤i 限制，不看未来

2. **score[j] / sum 的物理意义？**
   - 每个被问者的"注意力权重"，加和=1=100%

3. **为什么先算 CPU 版？**
   - GPU kernel 没有标准答案没法验证正确性
   - CPU 版虽然慢，但保证正确，是"考卷参考答案"

### 犯过的错

1. **因果 mask 优化时漏了自己**
   - `j < i` → 跳过 `j==i` → 从不看自己
   - 正确: `j <= i`

2. **V 加权循环到了未初始化的 scores**
   - `j < SEQ_LEN` 读了 scores[i+1..255] 的垃圾值
   - 正确: `j <= i`

---

## Step 3: 朴素 GPU Kernel

### 理解的问题

1. **half2 是什么，为什么用？**
   - half2 = 两个 half 打包，4 字节
   - GPU 一次读 4 字节 = 读两维
   - 64 维点积: float 版 64 次内存读，half2 版 32 次

2. **half2 不能直接 .x .y 访问？**
   - half2 是硬件存储格式，不是结构体
   - 要用 `__half22float2()` 转成 float2 才能拆 x,y

3. **naive kernel 为什么叫 naive？**
   - 不分块、不用 SRAM、全从 HBM 读
   - 最粗暴的 GPU 实现，只为验证"GPU 能算对"

4. **GPU kernel 内的变量存在哪里？**
   - 寄存器(最快, 0 延迟) — int, float, 小数组默认去
   - Local Memory(溢出时,HBM) — 寄存器不够自动溢
   - Shared Memory(程序员声明) — __shared__, 同 block 共享
   - GPU 无堆概念，只有这 3 种

5. **矩阵乘法的物理实现：不用转置**
   - Q[tid] · K[j] = 两个向量逐位乘加
   - 不是做完整矩阵乘法 Q×K^T
   - 是三重循环逐个算，每个 dot 是单独的点积

6. **<<<grid, block, sram>>> 三个参数的含义？**
   - grid = 几个 block 并行
   - block = 每个 block 多少线程
   - sram = extern __shared__ 的大小 (字节)

### 犯过的错

1. **kernel 函数名写错**
   - `__global__ void naive_attn_kernel(...);` 带分号声明
   - 忘了写函数体 `{ ... }`

2. **half2 指针转换忘写**
   - `Q_d` 是 `half*`，kernel 要 `const half2*`
   - 调用时强转 `(const half2 *)Q_d`

---

## Step 4: Flash Attention Tiled Kernel

### 理解的问题

1. **extern __shared__ 为什么用 extern？**
   - 声明和分配分离：声明在内核，大小在 <<<>>> 给
   - 大小运行时决定，改 tile 不改源码

2. **SRAM 如何手动分区？**
   - SRAM 是一整块连续内存，没有 malloc
   - 用指针算术切：Q_tile 前半，KV_tile 后半
   - Q_tile 常驻，KV_tile 每批覆盖

3. **Q_tile 协作加载：为什么要协作？**
   - 32 线程各搬每行的同一列 → 32 列 × 32 行 = 1024 元素
   - `idx += Br` = 跳一行到下一行同一列

4. **blockIdx.x, threadIdx.x 的来源？**
   - CUDA 内置变量，不用声明
   - <<<grid,block>>> 创建 block/线程时自动填充
   - 每个 block 有唯一 blockIdx，每个线程有唯一 threadIdx

5. **为什么 Br=32, Bc=64？**
   - Br=32: 一个 warp 刚好覆盖，不浪费线程
   - Bc=64: KV_tile=8KB，总 12KB < 48KB ✓
   - 循环 256/64=4 次，搬运合理
   - SRAM 留余量给多 block 交替隐藏延迟

6. **因果 mask 在 flash kernel 的写法**
   - `kv_start + j` = K 的全局行号，`q_start + my_row` = Q 的全局行号
   - K 全局 > Q 全局 → 未来 token → `dot = -1e9f`
   - 不能用 `continue` 跳过：softmax 分母需保留该位置，exp(-1e9f)≈0 刚好
   - 因为 K 数组在 HBM 里一次性全在，硬件不管语义，程序员自己加约束

7. **online softmax 贬值公式推导**
   - 分块前：全部用全局 max，`exp(dot - global_max)`
   - 分块后：先块用旧 max，发现新 max 更大时需修正
   - `exp(dot - new_max) = exp(dot - old_max) × exp(old_max - new_max)`
   - `scale_old = expf(KQ_max - new_max)` < 1 → 旧 VKQ 和 KQ_sum 等比贬值
   - 新旧分母统一到新 max 基准，保证最终归一化正确

8. **KV_tile 分时复用**
   - 同一块 SRAM，先装 K 算 QK 点积，再覆盖装 V 做加权累加
   - K 算完分数后不需要了 → 安全覆盖
   - K 加载和 V 加载代码一模一样，只改 `K[` → `V[`

9. **VKQ 归一化：为什么 kv 循环外一把除**
   - naive：`(score/sum) × V` → 每个 j 除一次
   - flash：`(Σ score×V) / sum` → sum 是常数可提出来，循环外一把除
   - 省 256 次除法，数学等价

10. **grid 和 sram_bytes 计算**
    - `grid = (SEQ_LEN + Br - 1) / Br` = ceil(256/32) = 8 个 block
    - `sram_bytes = (Br + Bc) × head_dim_half × sizeof(half)` = 96×32×2 = 6144 字节
    - SRAM 总需求 = Q_tile(2KB) + KV_tile(4KB) = 6KB < 48KB

11. **<<< >>> 和 ( ) 的区别**
    - `<<<grid, block, sram>>>` = 启动配置（告诉 GPU 怎么并行）
    - `(参数1, 参数2, ...)` = 普通函数参数（传数据）

### 犯过的错

1. **KV 加载忘了 __syncthreads()**

2. **K 访问忘了 kv_start 偏移**

3. **half2 直接用 .x .y (语法错误)**

4. **QK 点积忘了声明 dot 变量**

5. **`__stncthreads()` 拼写错误 → `__syncthreads()`**

6. **贬值时 KQ_sum 忘了乘 old_scale**
   - VKQ 贬值了但分母没贬 → 权重和不等于 1
   - 正确：VKQ[d] *= old_scale 同时 KQ_sum *= old_scale

7. **贬值循环上界用 head_dim_half 而非 HEAD_DIM**
   - VKQ 累加用 `VKQ[2*d]` `VKQ[2*d+1]` 填了 64 个 float
   - 贬值只跑 32 次 → 奇数位漏贬
   - 正确：`d < HEAD_DIM`(64)

8. **SRAM 大小算错**
   - 传了 `Br × HEAD_DIM × sizeof(half)` = 4KB，忘了 KV_tile 需要额外 4KB
   - 正确：`(Br + Bc) × (HEAD_DIM/2) × sizeof(half)` = 6KB

9. **main() 里 O_d2 用 new 分配又被 cudaMalloc 覆盖**
   - `float *O_d2 = new float[n]` 泄漏 CPU 内存
   - 正确：`float *O_d2; cudaMalloc(&O_d2, ...)`

10. **naive 比较在 flash 覆盖 O_gpu 后**
    - O_gpu 先存 naive 结果，后被 flash 结果覆盖
    - naive 比较必须在 flash 之前做

### 验证结果

板子编译运行：Naive 误差 0.000015，Flash 误差 0.000015。FP16 精度范围内一致。
