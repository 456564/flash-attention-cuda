// 迷你 Flash Attention — 独立 CUDA kernel
// 功能：单头 attention，分块 QK + online softmax + 累加 V
// 不集成 llama.cpp，先验证算法正确性
//
// 编译 (Jetson Orin):
//   nvcc -arch=sm_87 -O3 flash_attn_mini.cu -o flash_attn_mini
// 运行:
//   ./flash_attn_mini

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============ 参数 ============
constexpr int SEQ_LEN    = 256;   // 序列长度
constexpr int HEAD_DIM   = 64;    // head 维度 (Qwen2-0.5B)
constexpr int Br         = 32;    // Q tile 大小 (一次处理 32 个 token)
constexpr int Bc         = 64;    // KV tile 大小 (一次处理 64 个位置的 KV)

// ============ Flash Attention Kernel ============
__global__ void flash_attn_kernel(
    const half2 * __restrict__ Q,     // (seq_len, head_dim/2)  half2 格式
    const half2 * __restrict__ K,
    const half2 * __restrict__ V,
    float       * __restrict__ O,     // (seq_len, head_dim)
    int seq_len,
    int head_dim_half,                // head_dim / 2
    float scale                       // 1/sqrt(head_dim)
) {
    // 每个 block 处理 Q 的一个 tile (Br 行)
    int q_start = blockIdx.x * Br;
    int tid = threadIdx.x;             // 0..Br-1, 每个线程处理 Q 的一行

    extern __shared__ half2 sram[];
    half2 * Q_tile  = sram;                           // Br × head_dim/2
    half2 * KV_tile = sram + Br * head_dim_half;      // Bc × head_dim/2
    // KQ scores 存 float (Br × Bc)
    float * KQ      = (float *)(sram + Br * head_dim_half + Bc * head_dim_half);

    // 1. 加载 Q tile 到 shared memory
    for (int i = tid; i < Br * head_dim_half; i += blockDim.x) {
        int row = i / head_dim_half;
        int col = i % head_dim_half;
        if (q_start + row < seq_len) {
            Q_tile[i] = Q[(q_start + row) * head_dim_half + col];
        }
    }
    __syncthreads();

    // 每个线程的累加器
    float VKQ[HEAD_DIM] = {0.0f};  // O 的一行
    float KQ_max = -1e9f;          // online softmax 的 max
    float KQ_sum = 0.0f;           // online softmax 的 sum

    int my_row = tid;
    if (my_row >= Br || q_start + my_row >= seq_len) return;

    // 2. 主循环: 分块处理 KV
    for (int kv_start = 0; kv_start < seq_len; kv_start += Bc) {
        int kv_end = min(kv_start + Bc, seq_len);
        int kv_len = kv_end - kv_start;

        // 加载 K tile 到 shared memory
        for (int i = tid; i < kv_len * head_dim_half; i += blockDim.x) {
            int row = i / head_dim_half;
            int col = i % head_dim_half;
            KV_tile[i] = K[(kv_start + row) * head_dim_half + col];
        }
        __syncthreads();

        // 2a. KQ = Q × K^T — 对当前 tile 算点积
        float KQ_local[Bc] = {0.0f};

        for (int j = 0; j < kv_len; j++) {
            float dot = 0.0f;
            // 用 half2 逐对做 FMA
            for (int d = 0; d < head_dim_half; d++) {
                half2 qv = Q_tile[my_row * head_dim_half + d];
                half2 kv = KV_tile[j * head_dim_half + d];
                float2 qf = __half22float2(qv);
                float2 kf = __half22float2(kv);
                dot += qf.x * kf.x + qf.y * kf.y;
            }
            KQ_local[j] = dot * scale;
        }

        // 2b. 因果 mask: 只看 kv_start+j <= q_start+my_row
        for (int j = 0; j < kv_len; j++) {
            if (kv_start + j > q_start + my_row) {
                KQ_local[j] = -1e9f; // mask 掉后面的
            }
        }

        // 2c. Online softmax + V 累加
        float new_max = KQ_max;
        for (int j = 0; j < kv_len; j++) {
            new_max = fmaxf(new_max, KQ_local[j]);
        }

        // 重新缩放旧累加器
        float rescale = expf(KQ_max - new_max);
        KQ_sum *= rescale;
        for (int d = 0; d < HEAD_DIM; d++) {
            VKQ[d] *= rescale;
        }
        KQ_max = new_max;

        // 累加新 tile
        for (int j = 0; j < kv_len; j++) {
            float score = expf(KQ_local[j] - KQ_max);
            KQ_sum += score;
        }

        __syncthreads();

        // 加载 V tile 到 shared memory (复用 KV_tile 位置)
        for (int i = tid; i < kv_len * head_dim_half; i += blockDim.x) {
            int row = i / head_dim_half;
            int col = i % head_dim_half;
            KV_tile[i] = V[(kv_start + row) * head_dim_half + col];
        }
        __syncthreads();

        // V × softmax 累加到 VKQ
        for (int j = 0; j < kv_len; j++) {
            float weight = expf(KQ_local[j] - KQ_max);
            for (int d = 0; d < head_dim_half; d++) {
                half2 vv = KV_tile[j * head_dim_half + d];
                float2 vf = __half22float2(vv);
                VKQ[2*d]     += weight * vf.x;
                VKQ[2*d + 1] += weight * vf.y;
            }
        }
        __syncthreads();
    }

    // 3. 归一化: VKQ / sum → 写回全局内存
    for (int d = 0; d < HEAD_DIM; d++) {
        VKQ[d] /= KQ_sum;
        O[(q_start + my_row) * HEAD_DIM + d] = VKQ[d];
    }
}

// ============ CPU 参考实现 ============
void cpu_attention(const float * Q, const float * K, const float * V,
                   float * O, int seq_len, int head_dim) {
    for (int i = 0; i < seq_len; i++) {
        // KQ = Q[i] × K^T
        float * scores = new float[seq_len];
        float max_score = -1e9f;
        for (int j = 0; j < seq_len; j++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += Q[i * head_dim + d] * K[j * head_dim + d];
            }
            dot /= sqrtf((float)head_dim);
            // 因果 mask
            if (j > i) dot = -1e9f;
            scores[j] = dot;
            max_score = fmaxf(max_score, dot);
        }

        // softmax
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[j] = expf(scores[j] - max_score);
            sum += scores[j];
        }

        // V 加权
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                acc += scores[j] * V[j * head_dim + d];
            }
            O[i * head_dim + d] = acc / sum;
        }
        delete[] scores;
    }
}

// ============ 工具函数 ============
void init_random(half * data, int n) {
    for (int i = 0; i < n; i++) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        data[i] = __float2half(val);
    }
}

float max_diff(const float * a, const float * b, int n) {
    float max_d = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

int main() {
    srand(42);

    int seq_len        = SEQ_LEN;
    int head_dim       = HEAD_DIM;
    int head_dim_half  = HEAD_DIM / 2;
    int n_elements     = seq_len * head_dim;

    // 分配主机内存
    half * Q_h = new half[n_elements];
    half * K_h = new half[n_elements];
    half * V_h = new half[n_elements];
    init_random(Q_h, n_elements);
    init_random(K_h, n_elements);
    init_random(V_h, n_elements);

    // CPU 参考计算 (转 float)
    float * Q_f = new float[n_elements];
    float * K_f = new float[n_elements];
    float * V_f = new float[n_elements];
    for (int i = 0; i < n_elements; i++) {
        Q_f[i] = __half2float(Q_h[i]);
        K_f[i] = __half2float(K_h[i]);
        V_f[i] = __half2float(V_h[i]);
    }

    float * O_cpu = new float[n_elements];
    cpu_attention(Q_f, K_f, V_f, O_cpu, seq_len, head_dim);

    // 分配设备内存
    half2 * Q_d, * K_d, * V_d;
    float * O_d;
    cudaMalloc(&Q_d, n_elements * sizeof(half));
    cudaMalloc(&K_d, n_elements * sizeof(half));
    cudaMalloc(&V_d, n_elements * sizeof(half));
    cudaMalloc(&O_d, n_elements * sizeof(float));

    cudaMemcpy(Q_d, Q_h, n_elements * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K_h, n_elements * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V_h, n_elements * sizeof(half), cudaMemcpyHostToDevice);

    // SRAM 大小: Q_tile + KV_tile + KQ (作为共享内存的一部分)
    int sram_bytes = Br * head_dim_half * sizeof(half2)   // Q_tile
                   + Bc * head_dim_half * sizeof(half2)   // KV_tile
                   + Br * Bc * sizeof(float);              // KQ scores

    printf("SRAM 占用: %d bytes (%.1f KB)\n", sram_bytes, sram_bytes / 1024.0f);

    // 启动 kernel
    int n_blocks = (seq_len + Br - 1) / Br;
    flash_attn_kernel<<<n_blocks, Br, sram_bytes>>>(
        (const half2 *)Q_d, (const half2 *)K_d, (const half2 *)V_d,
        O_d, seq_len, head_dim_half, 1.0f / sqrtf((float)head_dim)
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA 错误: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 取回结果
    float * O_gpu = new float[n_elements];
    cudaMemcpy(O_gpu, O_d, n_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // 对比
    float diff = max_diff(O_cpu, O_gpu, n_elements);
    printf("序列长度: %d, head维度: %d, Br: %d, Bc: %d\n", seq_len, head_dim, Br, Bc);
    printf("最大误差: %.6f\n", diff);
    if (diff < 1e-2) {
        printf("✓ 通过 (误差 < 1e-2)\n");
    } else if (diff < 1e-1) {
        printf("~ 可接受 (误差 < 1e-1)\n");
    } else {
        printf("✗ 失败 (误差过大)\n");
    }

    // 打印前几个值对比
    printf("\n前 5 个输出对比:\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d] CPU: %10.6f  GPU: %10.6f  diff: %10.6f\n",
               i, O_cpu[i], O_gpu[i], fabsf(O_cpu[i] - O_gpu[i]));
    }

    // 清理
    delete[] Q_h; delete[] K_h; delete[] V_h;
    delete[] Q_f; delete[] K_f; delete[] V_f;
    delete[] O_cpu; delete[] O_gpu;
    cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);

    return 0;
}
