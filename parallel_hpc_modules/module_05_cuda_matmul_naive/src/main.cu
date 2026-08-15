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
__global__ void matmul_naive_kernel(
    float* d_a,
    float* d_b,
    float* d_c,
    int M,int N,int P
)
{   
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    int jdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < M && jdx < P)
    {   
        float tmp = 0;
        for(int k = 0;k < N;k++){
            tmp += d_a[idx*N+k]*d_b[k*P+jdx];
        }
        d_c[idx*P+jdx] = tmp;
    }
}

// CPU serial reference calculation
void cpu_verify(float* h_a, float* h_b, float* h_ref, int M,int N,int P)
{
    for(int i=0;i<M;i++){
        for(int j=0;j<P;j++){
            for(int k=0;k<N;k++){
                h_ref[i*P+j] += h_a[i*N+k]*h_b[k*P+j];
            }
        }
    }
}


int main()
{
    int M = 13;
    int N = 17;
    int P = 19;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // ===== Step1: Host memory allocation & initialization =====
    float *h_a, *h_b, *h_c_gpu, *h_ref_cpu;
    h_a       = (float *)malloc(M * N * sizeof(float));
    h_b       = (float *)malloc(N * P * sizeof(float));
    h_c_gpu   = (float *)malloc(M * P * sizeof(float));
    h_ref_cpu = (float *)malloc(M * P * sizeof(float));

    for (int i = 0; i < M*N; i++)
    {
        h_a[i] = dist(gen);
    }
    for (int i = 0; i < N*P; i++)
    {
        h_b[i] = dist(gen);
    }

    // ===== Step2: Device memory allocation =====
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_a, M * N * sizeof(float)));
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_b, N * P * sizeof(float)));
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_c, M * P * sizeof(float)));



    // ===== Step3: GPU kernel timing (cudaEvent) =====
    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA_FATAL(cudaEventCreate(&start_gpu));
    CHECK_CUDA_FATAL(cudaEventCreate(&stop_gpu));   


    const dim3 block_dim(16,16);
    const dim3 grid_dim((M + block_dim.x - 1) / block_dim.x,(P + block_dim.y - 1) / block_dim.y);

    CHECK_CUDA_FATAL(cudaEventRecord(start_gpu));
     // ===== Host -> Device copy =====
    CHECK_CUDA_FATAL(cudaMemcpy(d_a, h_a, M*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_FATAL(cudaMemcpy(d_b, h_b, N*P*sizeof(float), cudaMemcpyHostToDevice));

    matmul_naive_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M,N,P);
    CHECK_KERNEL();

    // ===== Copy the GPU result back to the CPU =====
    CHECK_CUDA_FATAL(cudaMemcpy(h_c_gpu, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));

    
    CHECK_CUDA_FATAL(cudaEventRecord(stop_gpu));
    CHECK_CUDA_FATAL(cudaEventSynchronize(stop_gpu));

    float gpu_ms = 0;
    CHECK_CUDA_FATAL(cudaEventElapsedTime(&gpu_ms, start_gpu, stop_gpu));


    // ===== Step4: CPU serial timing (high-resolution chrono) =====
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_verify(h_a, h_b, h_ref_cpu, M,N,P);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;

    // ===== Error verification =====
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

    // ===== Output timing and speedup =====
    std::cout << "====================================\n";
    std::cout << "GPU time:    " << gpu_ms << " ms\n";
    std::cout << "CPU serial time:    " << cpu_ms << " ms\n";
    std::cout << "Speedup (CPU/GPU):  " << cpu_ms / gpu_ms << " x\n";


    if (pass)
        std::cout << "Verification passed!\n";
    else
        std::cout << "Verification failed!\n";

    // ===== Release resources =====
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
