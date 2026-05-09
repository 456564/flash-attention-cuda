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
