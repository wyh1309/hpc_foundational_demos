/**
 * @file vector_add.cu
 * @brief Day4: Basic host <-> device memcpy + simplest kernel
 * Feature: GPU event timing + CPU chrono timing + speedup ratio
 */
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <random>
#include <cstdio>
#include <cmath>

#define CHECK_CUDA_FATAL(err) \
do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "[CUDA FATAL] %s:%d -> %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_KERNEL() \
do { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "[KERNEL LAUNCH ERROR] %s:%d -> %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// GPU Kernel
__global__ void vector_add_kernel(
    float* d_a,
    float* d_b,
    float* d_c,
    int n
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}

// CPU 串行参考计算
void cpu_verify(float* h_a, float* h_b, float* h_ref, int n)
{
    for (int i = 0; i < n; i++)
    {
        h_ref[i] = h_a[i] + h_b[i];
    }
}

int main()
{
    const int N = 1 << 20;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // ===== Step1: Host 内存分配 & 初始化 =====
    float *h_a, *h_b, *h_c_gpu, *h_ref_cpu;
    h_a       = (float *)malloc(N * sizeof(float));
    h_b       = (float *)malloc(N * sizeof(float));
    h_c_gpu   = (float *)malloc(N * sizeof(float));
    h_ref_cpu = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        h_a[i] = dist(gen);
        h_b[i] = dist(gen);
    }

    // ===== Step2: Device 显存分配 =====
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_b, N * sizeof(float)));
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_c, N * sizeof(float)));

    // ===== Step3: Host -> Device 拷贝 =====
    CHECK_CUDA_FATAL(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_FATAL(cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice));

    // ===== Step4: GPU Kernel 计时（cudaEvent） =====
    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA_FATAL(cudaEventCreate(&start_gpu));
    CHECK_CUDA_FATAL(cudaEventCreate(&stop_gpu));

    dim3 block_dim(256);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x);

    CHECK_CUDA_FATAL(cudaEventRecord(start_gpu));
    vector_add_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    CHECK_KERNEL();
    CHECK_CUDA_FATAL(cudaEventRecord(stop_gpu));
    CHECK_CUDA_FATAL(cudaEventSynchronize(stop_gpu));

    float gpu_kernel_ms = 0.0f;
    CHECK_CUDA_FATAL(cudaEventElapsedTime(&gpu_kernel_ms, start_gpu, stop_gpu));

    // ===== Step5: 取回GPU结果到CPU =====
    CHECK_CUDA_FATAL(cudaMemcpy(h_c_gpu, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));

    // ===== Step6: CPU 串行计时（chrono 高精度） =====
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_verify(h_a, h_b, h_ref_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;

    // ===== 误差校验 =====
    bool pass = true;
    const float eps = 1e-5f;
    for (int i = 0; i < N; i++)
    {
        if (fabs(h_c_gpu[i] - h_ref_cpu[i]) > eps)
        {
            pass = false;
            std::cerr << "Mismatch at index " << i
                      << " GPU:" << h_c_gpu[i]
                      << " CPU:" << h_ref_cpu[i] << "\n";
            break;
        }
    }

    // ===== 输出计时与加速比 =====
    std::cout << "====================================\n";
    std::cout << "GPU kernel time:    " << gpu_kernel_ms << " ms\n";
    std::cout << "CPU serial time:    " << cpu_ms << " ms\n";
    std::cout << "Speedup (CPU/GPU):  " << cpu_ms / gpu_kernel_ms << " x\n";

    // 计算显存带宽
    double total_bytes = 3.0 * N * sizeof(float); // read a, read b, write c
    double gb = total_bytes / (1024.0*1024.0*1024.0);
    double bw_gbs = gb / (gpu_kernel_ms / 1000.0);
    std::cout << "Effective bandwidth: " << bw_gbs << " GB/s\n";
    std::cout << "====================================\n";

    if (pass)
        std::cout << "Verification passed!\n";
    else
        std::cout << "Verification failed!\n";

    // ===== 释放资源 =====
    CHECK_CUDA_FATAL(cudaEventDestroy(start_gpu));
    CHECK_CUDA_FATAL(cudaEventDestroy(stop_gpu));

    CHECK_CUDA_FATAL(cudaFree(d_a));
    CHECK_CUDA_FATAL(cudaFree(d_b));
    CHECK_CUDA_FATAL(cudaFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_ref_cpu);

    return 0;
}
