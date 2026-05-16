#include <cstdio>          // printf
#include <cstdlib>         // srand, rand
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, cudaFree
#include <cuda_fp16.h>     // half, __float2half
#include <cmath>           // sqrt, exp, expf
#include <chrono>          // high_resolution_clock for timing

constexpr int SEQ_LEN  = 256;
constexpr int HEAD_DIM = 64;
constexpr int N_Q_HEADS = 14;
constexpr int N_KV_HEADS = 2;

constexpr int Br = 32;
constexpr int Bc = 64;

__global__ void naive_attn_kernel(
    const half2 * Q, // Q 的类型是 half2，每个元素包含两个半精度浮点数（FP16），占用 4 字节
    const half2 * K,
    const half2 * V,
    float * O,
    int seq_len,     // token的序列长度
    int head_dim_half, // 头部维度的一半，因为每个 half2 包含两个元素
    float scale      // 缩放因子，通常是 1/sqrt(head_dim)
){
    int i = threadIdx.x; // 获取当前线程号，在这里可以是第i个token
    if (i >= seq_len) return; // 防止超过总token

    float scores[256]; // 总权重数组，长度为序列长度，存储每个token与之前token的点积结果
    float max_score = -1e9f; // softmax的最大分数
    for (int j = 0; j <= i; j++){ // 计算QK点积，因为half2所以可以缩小一半的计算时间
        float dot = 0;
        for (int d = 0; d< head_dim_half; d++){
            half2 qv = Q[i * head_dim_half + d];
            half2 kv = K[j * head_dim_half + d];
            float2 qf = __half22float2(qv);
            float2 kf = __half22float2(kv);
            dot += qf.x * kf.x + qf.y * kf.y;
        }
        dot *= scale; // 缩放点积结果，通常是除以 sqrt(head_dim)，以防止数值过大导致softmax不稳定
        scores[j] = dot; // 将点积结果存储在 scores 数组中，长度为序列长度
        if (dot > max_score) max_score = dot; // 更新最大分数，用于后续的数值稳定的softmax计算
    }

    // softmax计算
    float sum = 0;
    for (int j = 0; j <= i; j++){ // 计算当前数和最大数的基于e的倍率差值，防止数值过大导致softmax不稳定
        scores[j] = expf(scores[j] - max_score);
        sum += scores[j];
    }

    for (int d = 0; d < head_dim_half; d++){ // 计算加权求和，得到输出O的第i个token的第d个维度的值
        float acc_x = 0, acc_y = 0;
        for (int j = 0; j <= i; j++){
            half2 vv = V[j * head_dim_half +d];
            float2 vf = __half22float2(vv);
            acc_x += scores[j] * vf.x;
            acc_y += scores[j] * vf.y;
        }
        O[i*64 + 2*d]    =  acc_x / sum; // 归一化，把分数权重转成0-1的概率
        O[i*64 + 2*d +1] =  acc_y / sum;
    }
}

__global__ void flash_attn_kernel( // flash attention
    const half2 * Q, const half2 * K, const half2 * V,
    float * O, int seq_len, int kv_max,int head_dim_half, float scale,
    int gqa_ratio
){
    __shared__ half2 sram[(Br + Bc) * (HEAD_DIM / 2)]; // SRAM: Q_tile(2KB) + KV_tile(4KB) = 6KB, 编译期固定
    half2 * Q_tile = sram;          // Q_tile占用前32*32个half2元素的共享内存，大小为32*32*4字节 = 4096字节
    half2 * KV_tile = sram + Br * head_dim_half;

    int q_start = blockIdx.x * Br;  // 每 block 处理 Br 行 Q
    int tid = threadIdx.x;

    int head_q = blockIdx.z;
    int head_kv = head_q / gqa_ratio;
    int offset = SEQ_LEN * head_dim_half;
    Q += head_q * offset;
    K += head_kv * offset;
    V += head_kv * offset;

    // 协作加载 Q_tile：32 个线程并行，每个搬几个元素
    for (int idx = tid; idx < Br * head_dim_half; idx += Br) {
        int row = idx / head_dim_half;
        int col = idx % head_dim_half;
        if (q_start + row < seq_len)
            Q_tile[idx] = Q[(q_start + row) * head_dim_half + col];
    }
    __syncthreads(); // 确保所有线程都完成了Q_tile的加载

    int my_row = tid; // 每个线程负责计算Q_tile中的一行与K/V的点积
    if (my_row >= Br || q_start + my_row >= seq_len) return; // 防止越界

    float VKQ[64] = {0};
    float KQ_max = -1e9f;
    float KQ_sum = 0.0f;
    float KQ_local[64];
    float new_max = -1e9f;

    // 循环K/V块
    for(int kv_start = 0; kv_start < kv_max; kv_start += Bc){
        int kv_end = kv_start + Bc;
        if (kv_end > kv_max) kv_end = kv_max;
        int kv_len = kv_end - kv_start;

        // ---- A. 加载 K 块到 KV_tile ----
        for (int idx = tid; idx < kv_len * head_dim_half; idx += Br){
            int row = idx / head_dim_half;
            int col = idx % head_dim_half;
            KV_tile[idx] = K[(kv_start + row) * head_dim_half + col];
        }

        __syncthreads(); // 确保所有线程都完成了K块的加载

        // ---- B. 计算 QK 点积 ----
        for(int j = 0; j < kv_len; j++){
            float dot = 0;
            for(int d = 0; d < head_dim_half; d++){
                half2 qv = Q_tile[my_row * head_dim_half + d];
                half2 kv = KV_tile[j * head_dim_half + d];
                float2 qf = __half22float2(qv);
                float2 kf = __half22float2(kv);
                dot += qf.x * kf.x + qf.y * kf.y; // 计算QK的点积，累加到dot变量中
            }
            dot *= scale; // 缩放点积结果，通常是除以 sqrt(head_dim)，以防止数值过大导致softmax不稳定
            if (kv_start + j > q_start + my_row) //  防止偷看到后面的token的信息，保证自回归的因果性
                dot = -1e9f;
            KQ_local[j] = dot; // 将点积结果存储在KQ_local数组中，长度为当前K块的长度
            if (dot > new_max) new_max = dot; // 更新当前块的最大分数，用于数值稳定的softmax计算
        }

        // ---- C. online softmax ----
        if(new_max > KQ_max){ // old_max用old_scale贬值,用new_max覆盖old_max,用new_max更新old_scale
            float old_scale = expf(KQ_max - new_max);
            for(int d = 0; d < HEAD_DIM; d++)
                VKQ[d] *= old_scale; // 把之前块的加权和乘以old_scale，贬值之前块的分数权重
            KQ_sum *= old_scale;
            KQ_max = new_max;
        }
        for (int j = 0; j < kv_len; j++){
            float score = expf(KQ_local[j] - KQ_max);
            KQ_sum += score; // 累加分数权重的和，用于后续的归一化
            KQ_local[j] = score;
        }
        __syncthreads();

        // ---- D. 计算加权求和 ----
        for (int idx = tid; idx < kv_len * head_dim_half; idx += Br){
            int row = idx / head_dim_half;
            int col = idx % head_dim_half;
            KV_tile[idx] = V[(kv_start + row) * head_dim_half + col];
        }
        __syncthreads();

        for (int j = 0; j < kv_len; j++){
            for (int d = 0; d < head_dim_half; d++){
                half2 vv = KV_tile[j * head_dim_half + d];
                float2 vf = __half22float2(vv);
                VKQ[2*d] += KQ_local[j] * vf.x;
                VKQ[2*d+1] += KQ_local[j] * vf.y;
            }
        }
        __syncthreads();
    }
    // ---- F. 归一化，得到最终的输出O ----
    for (int d = 0; d < HEAD_DIM; d++){
        O[(head_q * SEQ_LEN + q_start + my_row) * HEAD_DIM + d] = VKQ[d] / KQ_sum; // 归一化，把分数权重转成0-1的概率，得到最终的输出O
    }


}

int main() {
    int n_q  = N_Q_HEADS  * SEQ_LEN * HEAD_DIM;  // 14×16384=229376
    int n_kv = N_KV_HEADS * SEQ_LEN * HEAD_DIM;  //  2×16384=32768

    half * Q_d, * K_d, * V_d; // Q、K、V的设备内存指针（GPU的全局内存）
    float * O_d;              // 输出O的设备内存指针（GPU的全局内存）
    float * Q_h, * K_h, * V_h, * O_h; // Q、K、V、O的主机内存指针（CPU的堆内存）

    Q_h = new float[n_q];        // Q的本地内存分配（CPU的堆内存）
    K_h = new float[n_kv];        // K的本地内存分配（CPU的堆内存）
    V_h = new float[n_kv];        // V的本地内存分配（CPU的堆内存）
    O_h = new float[n_q];        // O的本地内存分配（CPU的堆内存）

    srand(42); // 设置随机数种子，确保每次运行生成相同的随机数序列
    // Q 单独：14 head
    for (int h = 0; h < N_Q_HEADS; h++){
        for (int i = 0; i < SEQ_LEN * HEAD_DIM; i++){
            Q_h[h * SEQ_LEN * HEAD_DIM + i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }
    // K/V 单独：2 head
    for (int h = 0; h < N_KV_HEADS; h++){
        for (int i = 0; i < SEQ_LEN * HEAD_DIM; i++){
            K_h[h * SEQ_LEN * HEAD_DIM + i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            V_h[h * SEQ_LEN * HEAD_DIM + i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }

    // CPU 参考：14 head attention，GQA（Q 14头 × K/V 2头，gqa_ratio=7）
    int gqa_ratio = N_Q_HEADS / N_KV_HEADS;
    for (int h = 0; h < N_Q_HEADS; h++){
        int head_kv = h / gqa_ratio;  // Q head h 对应 KV head
        for (int i = 0; i < SEQ_LEN; i++){
            float scores[SEQ_LEN];
            float max_score = -1e9f;
            for (int j = 0; j <= i; j++){
                float dot = 0;
                for (int d = 0; d < HEAD_DIM; d++){
                    dot += Q_h[(h * SEQ_LEN + i)        * HEAD_DIM + d]
                        * K_h[(head_kv * SEQ_LEN + j)   * HEAD_DIM + d];
                }
                dot /= sqrt(HEAD_DIM);
                scores[j] = dot;
                if (dot > max_score) max_score = dot;
            }
            float sum = 0;
            for (int j = 0; j <= i; j++){
                scores[j] = exp(scores[j] - max_score);
                sum += scores[j];
            }
            for (int d = 0; d < HEAD_DIM; d++){
                float acc = 0;
                for (int j = 0; j <= i; j++){
                    acc += scores[j] / sum * V_h[(head_kv * SEQ_LEN + j) * HEAD_DIM + d];
                }
                O_h[(h * SEQ_LEN + i) * HEAD_DIM + d] = acc;
            }
        }
    }
    printf("CPU 参考完成: O_h[0]=%f\n", O_h[0]);    

    // 在 GPU 全局内存上为 Q/K/V/O 分配空间
    cudaMalloc(&Q_d, n_q * sizeof(half));   // 16384 个 half × 2 字节 = 32768 字节
    cudaMalloc(&K_d, n_kv * sizeof(half));
    cudaMalloc(&V_d, n_kv * sizeof(half));
    cudaMalloc(&O_d, n_q * sizeof(float));  // 输出用 float，4 字节/元素

    // 将 Q/K/V 从主机内存复制到设备内存
    half * Q_half = new half[n_q]; // 临时数组，用于存储转换后的半精度值
    half * K_half = new half[n_kv];
    half * V_half = new half[n_kv];
    for (int i = 0; i < n_q; i++)
        Q_half[i] = __float2half(Q_h[i]);
    for (int i = 0; i < n_kv; i++){
        K_half[i] = __float2half(K_h[i]);
        V_half[i] = __float2half(V_h[i]);
    }
    cudaMemcpy(Q_d, Q_half, n_q * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K_half, n_kv * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V_half, n_kv * sizeof(half), cudaMemcpyHostToDevice);

    // ---- 计时：naive kernel (warmup=5, iters=50) ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int warmup = 5;
    const int iters  = 50;

    for (int k = 0; k < warmup; k++)
        naive_attn_kernel<<<1, 256>>>(
            (const half2 *)Q_d, (const half2 *)K_d, (const half2 *)V_d,
            O_d, SEQ_LEN, HEAD_DIM/2, 1.0f/sqrtf(HEAD_DIM));
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    for (int k = 0; k < iters; k++)
        naive_attn_kernel<<<1, 256>>>(
            (const half2 *)Q_d, (const half2 *)K_d, (const half2 *)V_d,
            O_d, SEQ_LEN, HEAD_DIM/2, 1.0f/sqrtf(HEAD_DIM));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start, stop);

    float * O_gpu = new float[n_q];
    cudaMemcpy(O_gpu, O_d, n_q * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err_naive = 0;
    for (int i = 0; i < SEQ_LEN * HEAD_DIM; i++){
        float err = fabsf(O_gpu[i] - O_h[i]);
        if (err > max_err_naive) max_err_naive = err;
    }
    printf("Naive GPU vs CPU 最大误差: %f\n", max_err_naive);

    // ---- 计时：flash kernel (多 head：dim3 grid, gqa_ratio) ----
    float * O_d2;
    cudaMalloc(&O_d2, n_q * sizeof(float));
    dim3 grid_3d((SEQ_LEN + Br - 1) / Br, 1, N_Q_HEADS);  // 8×1×14=112 blocks
    gqa_ratio = N_Q_HEADS / N_KV_HEADS;               // 7
    const int flash_iters = 10000;

    for (int k = 0; k < warmup; k++)
        flash_attn_kernel<<<grid_3d, Br>>>(
            (const half2 *)Q_d, (const half2 *)K_d, (const half2 *)V_d,
            O_d2, SEQ_LEN, SEQ_LEN, HEAD_DIM/2, 1.0f/sqrtf(HEAD_DIM), gqa_ratio);
    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < flash_iters; k++)
        flash_attn_kernel<<<grid_3d, Br>>>(
            (const half2 *)Q_d, (const half2 *)K_d, (const half2 *)V_d,
            O_d2, SEQ_LEN, SEQ_LEN, HEAD_DIM/2, 1.0f/sqrtf(HEAD_DIM), gqa_ratio);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    float ms_flash = std::chrono::duration<float, std::milli>(t2 - t1).count();

    cudaMemcpy(O_gpu, O_d2, n_q * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err_flash = 0;
    for (int i = 0; i < n_q; i++){
        float err = fabsf(O_gpu[i] - O_h[i]);
        if (err > max_err_flash) max_err_flash = err;
    }
    printf("Flash GPU vs CPU 最大误差: %f\n", max_err_flash);

    printf("\n========== 性能对比 (SEQ=%d, D=%d, warmup=%d) ==========\n",
        SEQ_LEN, HEAD_DIM, warmup);
    printf("Naive:  %.4f ms/run  (total %.2f ms / %d runs)\n", ms_naive / iters, ms_naive, iters);
    printf("Flash:  %.4f ms/run  (total %.2f ms / %d runs)\n", ms_flash / flash_iters, ms_flash, flash_iters);
    printf("加速比: %.2fx\n", (ms_naive / iters) / (ms_flash / flash_iters));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] O_gpu;

    cudaFree(Q_d);cudaFree(K_d);cudaFree(V_d);cudaFree(O_d);cudaFree(O_d2);
    delete[] Q_h;delete[] K_h;delete[] V_h;delete[] O_h;
    delete[] Q_half;delete[] K_half;delete[] V_half;

    printf("\n第 3 周完成\n");
    return 0;
}
