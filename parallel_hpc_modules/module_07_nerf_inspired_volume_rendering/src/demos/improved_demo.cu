#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

#include <cuda_runtime.h>

#include "cpu_forward.hpp"
#include "cuda_forward.cuh"

#define CHECK_CUDA(err) do { \
    const cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "[CUDA] %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(e)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (false)

bool run_improved_demo(int num_rays, int num_samples, float ds) {
    const std::size_t samples = static_cast<std::size_t>(num_rays) *
                                static_cast<std::size_t>(num_samples);
    const std::size_t sample_bytes = samples * sizeof(float);
    const std::size_t ray_bytes = static_cast<std::size_t>(num_rays) * sizeof(float);

    // Page-locked host buffers allow the CUDA driver to transfer data directly.
    float *h_sigma = nullptr, *h_source = nullptr, *h_i0 = nullptr;
    float *h_gpu = nullptr, *h_cpu = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_sigma, sample_bytes));
    CHECK_CUDA(cudaMallocHost(&h_source, sample_bytes));
    CHECK_CUDA(cudaMallocHost(&h_i0, ray_bytes));
    CHECK_CUDA(cudaMallocHost(&h_gpu, ray_bytes));
    CHECK_CUDA(cudaMallocHost(&h_cpu, ray_bytes));

    std::mt19937 generator(12345);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (std::size_t i = 0; i < samples; ++i) {
        h_sigma[i] = distribution(generator);
        h_source[i] = distribution(generator);
    }
    for (int i = 0; i < num_rays; ++i) h_i0[i] = distribution(generator);

    const auto cpu_start = std::chrono::high_resolution_clock::now();
    solve_transport_cpu_multi(h_sigma, h_source, h_i0, h_cpu,
                              num_rays, num_samples, ds);
    const auto cpu_stop = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(
        cpu_stop - cpu_start).count();

    float *d_sigma = nullptr, *d_source = nullptr, *d_i0 = nullptr, *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_sigma, sample_bytes));
    CHECK_CUDA(cudaMalloc(&d_source, sample_bytes));
    CHECK_CUDA(cudaMalloc(&d_i0, ray_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, ray_bytes));

    cudaEvent_t e2e_start, e2e_stop, h2d_start, h2d_stop;
    cudaEvent_t kernel_start, kernel_stop, d2h_start, d2h_stop;
    CHECK_CUDA(cudaEventCreate(&e2e_start));
    CHECK_CUDA(cudaEventCreate(&e2e_stop));
    CHECK_CUDA(cudaEventCreate(&h2d_start));
    CHECK_CUDA(cudaEventCreate(&h2d_stop));
    CHECK_CUDA(cudaEventCreate(&kernel_start));
    CHECK_CUDA(cudaEventCreate(&kernel_stop));
    CHECK_CUDA(cudaEventCreate(&d2h_start));
    CHECK_CUDA(cudaEventCreate(&d2h_stop));

    // Warmup is excluded from reported timings.
    CHECK_CUDA(cudaMemcpy(d_sigma, h_sigma, sample_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_source, h_source, sample_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_i0, h_i0, ray_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(launch_volume_transport_cuda(d_sigma, d_source, d_output,
                                             num_rays, num_samples, ds, d_i0));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(e2e_start));
    CHECK_CUDA(cudaEventRecord(h2d_start));
    CHECK_CUDA(cudaMemcpy(d_sigma, h_sigma, sample_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_source, h_source, sample_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_i0, h_i0, ray_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(h2d_stop));
    CHECK_CUDA(cudaEventRecord(kernel_start));
    CHECK_CUDA(launch_volume_transport_cuda(d_sigma, d_source, d_output,
                                             num_rays, num_samples, ds, d_i0));
    CHECK_CUDA(cudaEventRecord(kernel_stop));
    CHECK_CUDA(cudaEventRecord(d2h_start));
    CHECK_CUDA(cudaMemcpy(h_gpu, d_output, ray_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(d2h_stop));
    CHECK_CUDA(cudaEventRecord(e2e_stop));
    CHECK_CUDA(cudaEventSynchronize(e2e_stop));

    float h2d_ms, kernel_ms, d2h_ms, e2e_ms;
    CHECK_CUDA(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    CHECK_CUDA(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));
    CHECK_CUDA(cudaEventElapsedTime(&e2e_ms, e2e_start, e2e_stop));

    double squared_error = 0.0;
    float max_error = 0.0f;
    for (int i = 0; i < num_rays; ++i) {
        const float error = std::fabs(h_gpu[i] - h_cpu[i]);
        squared_error += static_cast<double>(error) * error;
        max_error = std::max(max_error, error);
    }
    const double rmse = std::sqrt(squared_error / num_rays);
    const double input_bytes = 2.0 * sample_bytes + ray_bytes;
    const double h2d_gbps = h2d_ms > 0.0
        ? input_bytes / 1.0e9 / (h2d_ms / 1000.0) : 0.0;
    const double throughput = kernel_ms > 0.0
        ? samples / (kernel_ms / 1000.0) / 1.0e6 : 0.0;

    std::cout << std::fixed << std::setprecision(6)
              << "rays=" << num_rays << ", samples=" << num_samples
              << ", ds=" << ds << "\n"
              << "host_memory: pinned (cudaMallocHost)\n"
              << "warmup: 1 run (excluded)\n"
              << "H2D: " << h2d_ms << " ms (" << h2d_gbps << " GB/s)\n"
              << "kernel: " << kernel_ms << " ms (" << throughput << " Msamples/s)\n"
              << "D2H: " << d2h_ms << " ms\n"
              << "end-to-end: " << e2e_ms << " ms\n"
              << "CPU serial: " << cpu_ms << " ms\n"
              << "Speedup (CPU / end-to-end): " << cpu_ms / e2e_ms << " x\n"
              << "RMSE: " << rmse << ", max abs error: " << max_error << "\n"
              << (max_error <= 1.0e-3f ? "Verification passed!\n"
                                       : "Verification failed!\n");

    CHECK_CUDA(cudaEventDestroy(e2e_start));
    CHECK_CUDA(cudaEventDestroy(e2e_stop));
    CHECK_CUDA(cudaEventDestroy(h2d_start));
    CHECK_CUDA(cudaEventDestroy(h2d_stop));
    CHECK_CUDA(cudaEventDestroy(kernel_start));
    CHECK_CUDA(cudaEventDestroy(kernel_stop));
    CHECK_CUDA(cudaEventDestroy(d2h_start));
    CHECK_CUDA(cudaEventDestroy(d2h_stop));
    CHECK_CUDA(cudaFree(d_sigma));
    CHECK_CUDA(cudaFree(d_source));
    CHECK_CUDA(cudaFree(d_i0));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFreeHost(h_sigma));
    CHECK_CUDA(cudaFreeHost(h_source));
    CHECK_CUDA(cudaFreeHost(h_i0));
    CHECK_CUDA(cudaFreeHost(h_gpu));
    CHECK_CUDA(cudaFreeHost(h_cpu));
    return max_error <= 1.0e-3f;
}

int main(int argc, char** argv) {
    const int num_rays = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int num_samples = argc > 2 ? std::atoi(argv[2]) : 256;
    if (num_rays <= 0 || num_samples <= 0 || argc > 3) {
        std::cerr << "usage: improved_demo [num_rays] [num_samples]\n";
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaFree(0));
    return run_improved_demo(num_rays, num_samples, 0.01f)
               ? EXIT_SUCCESS : EXIT_FAILURE;
}
