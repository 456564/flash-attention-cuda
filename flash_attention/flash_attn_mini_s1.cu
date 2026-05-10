#include <cstdio>          // printf
#include <cstdlib>         // srand, rand
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, cudaFree
#include <cuda_fp16.h>     // half, __float2half
#include <cmath>           // sqrt, exp, expf

constexpr int SEQ_LEN  = 256;
constexpr int HEAD_DIM = 64;

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
    float * O, int seq_len, int head_dim_half, float scale
){
    extern __shared__ half2 sram[]; // 先声明half2类型的共享内存，后续会根据需要划分为Q_tile和KV_tile
    half2 * Q_tile = sram;          // Q_tile占用前32*32个half2元素的共享内存，大小为32*32*4字节 = 4096字节
    half2 * KV_tile = sram + Br * head_dim_half;

    int q_start = blockIdx.x * Br;  // 每 block 处理 Br 行 Q
    int tid = threadIdx.x;

    // 协作加载 Q_tile：32 个线程并行，每个搬几个元素
    for (int idx = tid; idx < Br * head_dim_half; idx += Br) {
        int row = idx / head_dim_half;
        int col = idx % head_dim_half;
        if (q_start + row < seq_len)
            Q_tile[idx] = Q[(q_start + row) * head_dim_half + col];
    }
    __syncthreads(); // 确保所有线程都完成了Q_tile的加载

}

int main() {
    int n = SEQ_LEN * HEAD_DIM; // 计算总元素数量（序列长度乘以头部维度）

    half * Q_d, * K_d, * V_d; // Q、K、V的设备内存指针（GPU的全局内存）
    float * O_d;              // 输出O的设备内存指针（GPU的全局内存）
    float * Q_h, * K_h, * V_h, * O_h; // Q、K、V、O的主机内存指针（CPU的堆内存）

    Q_h = new float[n];        // Q的本地内存分配（CPU的堆内存）
    K_h = new float[n];        // K的本地内存分配（CPU的堆内存）
    V_h = new float[n];        // V的本地内存分配（CPU的堆内存）
    O_h = new float[n];        // O的本地内存分配（CPU的堆内存）

    srand(42); // 设置随机数种子，确保每次运行生成相同的随机数序列
    for (int i = 0; i < n; i++){
        float r = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;     // 生成[-0.05, 0.05]范围内的随机数（单精度浮点数，FP32，约7位有效数字）
        float r2 = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;    
        float r3 = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        Q_h[i] = (r);                               // 将随机数赋值给 Q_h 数组的第 i 个元素 (FP32，4字节/元素)
        K_h[i] = (r2);                              
        V_h[i] = (r3);                              
    }

    // 在CPU计算attention,作为标准答案
    for (int i = 0; i < SEQ_LEN; i++){
        float scores[SEQ_LEN];
        float max_score = -1e9f;
        for (int j = 0; j <= i; j++){
            float dot = 0;
            for (int d = 0; d < HEAD_DIM; d++){
                dot += Q_h[i * HEAD_DIM +d] * K_h[j * HEAD_DIM +d];
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
                acc += scores[j] / sum * V_h[j * HEAD_DIM + d];
            }
            O_h[i * HEAD_DIM + d] = acc;
        }
    }
    printf("CPU 参考完成: O_h[0]=%f, O_h[100]=%f\n", O_h[0], O_h[100]);

    // 在 GPU 全局内存上为 Q/K/V/O 分配空间
    cudaMalloc(&Q_d, n * sizeof(half));   // 16384 个 half × 2 字节 = 32768 字节
    cudaMalloc(&K_d, n * sizeof(half));
    cudaMalloc(&V_d, n * sizeof(half));
    cudaMalloc(&O_d, n * sizeof(float));  // 输出用 float，4 字节/元素

    // 将 Q/K/V 从主机内存复制到设备内存
    half * Q_half = new half[n]; // 临时数组，用于存储转换后的半精度值
    half * K_half = new half[n];
    half * V_half = new half[n];
    for (int i = 0; i < n; i++) {
        Q_half[i] = __float2half(Q_h[i]); // 将单精度浮点数转换为半精度并存储在临时数组中
        K_half[i] = __float2half(K_h[i]);
        V_half[i] = __float2half(V_h[i]);
    }
    cudaMemcpy(Q_d, Q_half, n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K_half, n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V_half, n * sizeof(half), cudaMemcpyHostToDevice);

    naive_attn_kernel<<<1, 256>>>(
        (const half2 *)Q_d, (const half2 *)K_d, (const half2 *)V_d,
        O_d, SEQ_LEN, HEAD_DIM/2, 1.0f/sqrtf(HEAD_DIM)
    );
    cudaDeviceSynchronize();

    float * O_gpu = new float[n]; // 用于从设备内存复制回主机内存的数组，存储 GPU 计算的结果
    cudaMemcpy(O_gpu, O_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0;
    for (int i = 0; i < n; i++){
        float err = fabsf(O_gpu[i] - O_h[i]);
        if (err > max_err) max_err = err;
    }
    printf("GPU vs CPU最大误差: %f\n", max_err);
    delete[] O_gpu;

    // 清理设备内存和主机内存
    cudaFree(Q_d);cudaFree(K_d);cudaFree(V_d);cudaFree(O_d);
    delete[] Q_h;delete[] K_h;delete[] V_h;delete[] O_h;
    delete[] Q_half;delete[] K_half;delete[] V_half;

    printf("step 3 完成：完成CPU的标准计算\n");
    

    return 0;
}