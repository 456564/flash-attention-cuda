#include <cstdio>          // printf
#include <cstdlib>         // srand, rand
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, cudaFree
#include <cuda_fp16.h>     // half, __float2half

constexpr int SEQ_LEN  = 256;
constexpr int HEAD_DIM = 64;

int main() {
    int n = SEQ_LEN * HEAD_DIM; // 计算总元素数量（序列长度乘以头部维度）

    half * Q_d, * K_d, * V_d; // Q、K、V的设备内存指针（GPU的全局内存）
    float * O_d;              // 输出O的设备内存指针（GPU的全局内存）
    half * Q_h, * K_h, * V_h; // Q、K、V的主机内存指针（CPU的堆内存）

    Q_h = new half[n];        // Q的本地内存分配（CPU的堆内存）
    K_h = new half[n];        // K的本地内存分配（CPU的堆内存）
    V_h = new half[n];        // V的本地内存分配（CPU的堆内存）
    for (int i = 0; i < n; i++){
        float r = (srand(42), (float)rand() / RAND_MAX - 0.5f) * 0.1f;     // 生成[-0.05, 0.05]范围内的随机数（单精度浮点数，FP32，约7位有效数字）
        float r2 = (srand(42), (float)rand() / RAND_MAX - 0.5f) * 0.1f;    
        float r3 = (srand(42), (float)rand() / RAND_MAX - 0.5f) * 0.1f;
        Q_h[i] = __float2half(r);                               // 将全精度浮点数转换为半精度浮点数，并存储在Q_h数组中，以便后续使用（FP16，约3位有效数字）
        K_h[i] = __float2half(r2);                              // 将全精度浮点数转换为半精度浮点数，并存储在K_h数组中，以便后续使用（FP16，约3位有效数字）
        V_h[i] = __float2half(r3);                              // 将全精度浮点数转换为半精度浮点数，并存储在V_h数组中，以便后续使用（FP16，约3位有效数字）
    }

    // 在 GPU 全局内存上为 Q/K/V/O 分配空间
    cudaMalloc(&Q_d, n * sizeof(half));   // 16384 个 half × 2 字节 = 32768 字节
    cudaMalloc(&K_d, n * sizeof(half));
    cudaMalloc(&V_d, n * sizeof(half));
    cudaMalloc(&O_d, n * sizeof(float));  // 输出用 float，4 字节/元素

    // 将 Q/K/V 从主机内存复制到设备内存
    cudaMemcpy(Q_d, Q_h, n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K_h, n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V_h, n * sizeof(half), cudaMemcpyHostToDevice);

    // 清理设备内存和主机内存
    cudaFree(Q_d);cudaFree(K_d);cudaFree(V_d);cudaFree(O_d);
    delete[] Q_h;delete[] K_h;delete[] V_h;

    printf("step 1 完成：Q/K/V 已成功分配和初始化，并复制到 GPU 内存。\n");

    return 0;
}