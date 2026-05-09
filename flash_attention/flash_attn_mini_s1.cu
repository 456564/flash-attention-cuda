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

    // 清理设备内存和主机内存
    cudaFree(Q_d);cudaFree(K_d);cudaFree(V_d);cudaFree(O_d);
    delete[] Q_h;delete[] K_h;delete[] V_h;delete[] O_h;
    delete[] Q_half;delete[] K_half;delete[] V_half;

    printf("step 1 完成：Q/K/V 已成功分配和初始化，并复制到 GPU 内存。\n");

    return 0;
}