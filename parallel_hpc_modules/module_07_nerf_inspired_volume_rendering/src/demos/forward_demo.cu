#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <random>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include "cuda_forward.cuh"
#include "cpu_forward.hpp"

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

bool cuda_forward(int num_rays, int num_samples, float ds, cudaStream_t stream)
{
    const std::size_t total_samples = static_cast<std::size_t>(num_rays) *
                                      static_cast<std::size_t>(num_samples);
    const std::size_t sample_bytes = total_samples * sizeof(float);
    const std::size_t ray_bytes = static_cast<std::size_t>(num_rays) * sizeof(float);

    std::mt19937 gen(12345);  // Keep the demo input reproducible.
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // ===== Step1: Host memory allocation & initialization =====
    float *h_sigma, *h_source, *h_I0, *h_output_gpu, *h_ref_cpu;
    h_sigma = static_cast<float *>(malloc(sample_bytes));
    h_source = static_cast<float *>(malloc(sample_bytes));
    h_I0 = static_cast<float *>(malloc(ray_bytes));
    h_output_gpu = static_cast<float *>(malloc(ray_bytes));
    h_ref_cpu = static_cast<float *>(malloc(ray_bytes));
    if (!h_sigma || !h_source || !h_I0 || !h_output_gpu || !h_ref_cpu) {
        std::cerr << "Host allocation failed\n";
        return false;
    }

    for (std::size_t i = 0; i < total_samples; ++i) {
        h_sigma[i] = dist(gen);
        h_source[i] = dist(gen);
    }
    for (int i = 0; i < num_rays; ++i) {
        h_I0[i] = dist(gen);
    }

    // ===== Step2: CPU reference timing =====
    const auto cpu_start = std::chrono::high_resolution_clock::now();
    solve_transport_cpu_multi(h_sigma, h_source, h_I0, h_ref_cpu,
                              num_rays, num_samples, ds);
    const auto cpu_end = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(
                              cpu_end - cpu_start).count();

    // ===== Step3: Device memory allocation =====
    float *d_sigma = nullptr, *d_source = nullptr, *d_I0 = nullptr, *d_output = nullptr;
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_sigma, sample_bytes));
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_source, sample_bytes));
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_I0, ray_bytes));
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_output, ray_bytes));

    // Separate events make H2D, kernel, D2H, and end-to-end times visible.
    cudaEvent_t e2e_start, e2e_stop, h2d_start, h2d_stop;
    cudaEvent_t kernel_start, kernel_stop, d2h_start, d2h_stop;
    CHECK_CUDA_FATAL(cudaEventCreate(&e2e_start));
    CHECK_CUDA_FATAL(cudaEventCreate(&e2e_stop));
    CHECK_CUDA_FATAL(cudaEventCreate(&h2d_start));
    CHECK_CUDA_FATAL(cudaEventCreate(&h2d_stop));
    CHECK_CUDA_FATAL(cudaEventCreate(&kernel_start));
    CHECK_CUDA_FATAL(cudaEventCreate(&kernel_stop));
    CHECK_CUDA_FATAL(cudaEventCreate(&d2h_start));
    CHECK_CUDA_FATAL(cudaEventCreate(&d2h_stop));

    // ===== Step4: Warm-up (excluded from all reported timings) =====
    CHECK_CUDA_FATAL(cudaMemcpy(d_sigma, h_sigma, sample_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_FATAL(cudaMemcpy(d_source, h_source, sample_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_FATAL(cudaMemcpy(d_I0, h_I0, ray_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_FATAL(launch_volume_transport_cuda(
        d_sigma, d_source, d_output, num_rays, num_samples, ds, d_I0, stream));
    CHECK_KERNEL();
    CHECK_CUDA_FATAL(cudaStreamSynchronize(stream));

    // ===== Step5: Timed H2D -> kernel -> D2H pipeline =====
    CHECK_CUDA_FATAL(cudaEventRecord(e2e_start, stream));
    CHECK_CUDA_FATAL(cudaEventRecord(h2d_start, stream));
    CHECK_CUDA_FATAL(cudaMemcpy(d_sigma, h_sigma, sample_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_FATAL(cudaMemcpy(d_source, h_source, sample_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_FATAL(cudaMemcpy(d_I0, h_I0, ray_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_FATAL(cudaEventRecord(h2d_stop, stream));

    CHECK_CUDA_FATAL(cudaEventRecord(kernel_start, stream));
    CHECK_CUDA_FATAL(launch_volume_transport_cuda(
        d_sigma, d_source, d_output, num_rays, num_samples, ds, d_I0, stream));
    CHECK_KERNEL();
    CHECK_CUDA_FATAL(cudaEventRecord(kernel_stop, stream));

    CHECK_CUDA_FATAL(cudaEventRecord(d2h_start, stream));
    CHECK_CUDA_FATAL(cudaMemcpy(h_output_gpu, d_output, ray_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_FATAL(cudaEventRecord(d2h_stop, stream));
    CHECK_CUDA_FATAL(cudaEventRecord(e2e_stop, stream));
    CHECK_CUDA_FATAL(cudaEventSynchronize(e2e_stop));

    float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f, e2e_ms = 0.0f;
    CHECK_CUDA_FATAL(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));
    CHECK_CUDA_FATAL(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    CHECK_CUDA_FATAL(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));
    CHECK_CUDA_FATAL(cudaEventElapsedTime(&e2e_ms, e2e_start, e2e_stop));

    // ===== Step6: Error verification =====
    double squared_error = 0.0;
    float max_abs_error = 0.0f;
    for (int i = 0; i < num_rays; ++i) {
        const float error = std::fabs(h_output_gpu[i] - h_ref_cpu[i]);
        squared_error += static_cast<double>(error) * error;
        max_abs_error = std::max(max_abs_error, error);
    }
    const double rmse = std::sqrt(squared_error / num_rays);
    const bool pass = max_abs_error <= 1.0e-3f;
    const std::size_t device_bytes = 2 * sample_bytes + 2 * ray_bytes;

    // ===== Step7: Output timing, throughput, memory, and speedup =====
    const double h2d_gbps = h2d_ms > 0.0f
                                ? (2.0 * sample_bytes + ray_bytes) / 1.0e9 / (h2d_ms / 1000.0)
                                : 0.0;
    const double d2h_gbps = d2h_ms > 0.0f
                                ? ray_bytes / 1.0e9 / (d2h_ms / 1000.0)
                                : 0.0;
    const double samples_per_second = kernel_ms > 0.0f
                                ? total_samples / (kernel_ms / 1000.0)
                                : 0.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "rays=" << num_rays << ", samples=" << num_samples << ", ds=" << ds << "\n";
    std::cout << "warmup: 1 run (excluded)\n";
    std::cout << "H2D: " << h2d_ms << " ms (" << h2d_gbps << " GB/s)\n";
    std::cout << "kernel: " << kernel_ms << " ms (" << samples_per_second / 1.0e6
              << " Msamples/s)\n";
    std::cout << "D2H: " << d2h_ms << " ms (" << d2h_gbps << " GB/s)\n";
    std::cout << "end-to-end: " << e2e_ms << " ms\n";
    std::cout << "CPU serial: " << cpu_ms << " ms\n";
    std::cout << "Speedup (CPU / end-to-end): " << cpu_ms / e2e_ms << " x\n";
    std::cout << "Device memory: " << device_bytes / (1024.0 * 1024.0) << " MiB\n";
    std::cout << "RMSE: " << rmse << ", max abs error: " << max_abs_error << "\n";
    std::cout << (pass ? "Verification passed!\n" : "Verification failed!\n");

    CHECK_CUDA_FATAL(cudaEventDestroy(e2e_start));
    CHECK_CUDA_FATAL(cudaEventDestroy(e2e_stop));
    CHECK_CUDA_FATAL(cudaEventDestroy(h2d_start));
    CHECK_CUDA_FATAL(cudaEventDestroy(h2d_stop));
    CHECK_CUDA_FATAL(cudaEventDestroy(kernel_start));
    CHECK_CUDA_FATAL(cudaEventDestroy(kernel_stop));
    CHECK_CUDA_FATAL(cudaEventDestroy(d2h_start));
    CHECK_CUDA_FATAL(cudaEventDestroy(d2h_stop));
    CHECK_CUDA_FATAL(cudaFree(d_I0));
    CHECK_CUDA_FATAL(cudaFree(d_output));
    CHECK_CUDA_FATAL(cudaFree(d_sigma));
    CHECK_CUDA_FATAL(cudaFree(d_source));
    free(h_I0);
    free(h_output_gpu);
    free(h_ref_cpu);
    free(h_sigma);
    free(h_source);
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char** argv)
{
    int num_rays = argc > 1 ? std::atoi(argv[1]) : 4096;
    int num_samples = argc > 2 ? std::atoi(argv[2]) : 256;
    if (num_rays <= 0 || num_samples <= 0 || argc > 3) {
        std::cerr << "usage: forward_demo [num_rays] [num_samples]\n";
        return EXIT_FAILURE;
    }

    CHECK_CUDA_FATAL(cudaFree(0));  // Exclude CUDA context initialization from timings.
    return cuda_forward(num_rays, num_samples, 0.01f, 0)
               ? EXIT_SUCCESS : EXIT_FAILURE;
}
