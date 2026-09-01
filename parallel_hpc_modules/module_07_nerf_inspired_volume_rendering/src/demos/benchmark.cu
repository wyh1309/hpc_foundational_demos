#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "cpu_forward.hpp"
#include "cuda_forward.cuh"

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                      \
        if (error != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << ": " << cudaGetErrorString(error) << '\n';            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                       \
    } while (false)

namespace {

struct Timings {
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double end_to_end_ms = 0.0;
};

double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
}

double cpu_time_ms(const std::vector<float>& sigma,
                   const std::vector<float>& source,
                   const std::vector<float>& initial_intensity,
                   std::vector<float>& output,
                   int num_rays, int num_samples, float ds) {
    const auto start = std::chrono::steady_clock::now();
    solve_transport_cpu_multi(sigma.data(), source.data(),
                              initial_intensity.data(), output.data(),
                              num_rays, num_samples, ds);
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

Timings benchmark_case(int num_rays, int num_samples,
                       int warmup_iterations, int measurement_iterations,
                       float ds, std::mt19937& generator) {
    const std::size_t sample_count = static_cast<std::size_t>(num_rays) *
                                     static_cast<std::size_t>(num_samples);
    const std::size_t sample_bytes = sample_count * sizeof(float);
    const std::size_t ray_bytes = static_cast<std::size_t>(num_rays) * sizeof(float);

    std::vector<float> h_sigma(sample_count);
    std::vector<float> h_source(sample_count);
    std::vector<float> h_initial_intensity(num_rays);
    std::vector<float> h_output(num_rays);
    std::vector<float> h_cpu_output(num_rays);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (std::size_t i = 0; i < sample_count; ++i) {
        h_sigma[i] = distribution(generator);
        h_source[i] = distribution(generator);
    }
    for (float& value : h_initial_intensity) {
        value = distribution(generator);
    }

    const double cpu_ms = cpu_time_ms(
        h_sigma, h_source, h_initial_intensity, h_cpu_output,
        num_rays, num_samples, ds);

    float *d_sigma = nullptr, *d_source = nullptr;
    float *d_initial_intensity = nullptr, *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_sigma, sample_bytes));
    CHECK_CUDA(cudaMalloc(&d_source, sample_bytes));
    CHECK_CUDA(cudaMalloc(&d_initial_intensity, ray_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, ray_bytes));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Copy once before warm-up. Warm-up launches use the same input and output buffers.
    CHECK_CUDA(cudaMemcpy(d_sigma, h_sigma.data(), sample_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_source, h_source.data(), sample_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_initial_intensity, h_initial_intensity.data(),
                         ray_bytes, cudaMemcpyHostToDevice));
    for (int i = 0; i < warmup_iterations; ++i) {
        CHECK_CUDA(launch_volume_transport_cuda(
            d_sigma, d_source, d_output, num_rays, num_samples,
            ds, d_initial_intensity));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    Timings total;
    for (int iteration = 0; iteration < measurement_iterations; ++iteration) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(d_sigma, h_sigma.data(), sample_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_source, h_source.data(), sample_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_initial_intensity, h_initial_intensity.data(),
                              ray_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        total.h2d_ms += elapsed_ms(start, stop);

        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(launch_volume_transport_cuda(
            d_sigma, d_source, d_output, num_rays, num_samples,
            ds, d_initial_intensity));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        total.kernel_ms += elapsed_ms(start, stop);

        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, ray_bytes,
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        total.d2h_ms += elapsed_ms(start, stop);

        // End-to-end includes a fresh H2D, kernel launch, and D2H transfer.
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(d_sigma, h_sigma.data(), sample_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_source, h_source.data(), sample_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_initial_intensity, h_initial_intensity.data(),
                              ray_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(launch_volume_transport_cuda(
            d_sigma, d_source, d_output, num_rays, num_samples,
            ds, d_initial_intensity));
        CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, ray_bytes,
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        total.end_to_end_ms += elapsed_ms(start, stop);
    }

    total.h2d_ms /= measurement_iterations;
    total.kernel_ms /= measurement_iterations;
    total.d2h_ms /= measurement_iterations;
    total.end_to_end_ms /= measurement_iterations;

    const double max_error = [&]() {
        double result = 0.0;
        for (int i = 0; i < num_rays; ++i) {
            result = std::max(result, static_cast<double>(
                std::fabs(h_output[i] - h_cpu_output[i])));
        }
        return result;
    }();

    const std::size_t device_bytes = 2 * sample_bytes + 2 * ray_bytes;
    const double input_bytes = 2.0 * sample_bytes + ray_bytes;
    const double output_bytes = static_cast<double>(ray_bytes);
    const double samples_per_second = total.kernel_ms > 0.0
        ? sample_count / (total.kernel_ms / 1000.0) : 0.0;
    const double h2d_gbps = total.h2d_ms > 0.0
        ? input_bytes / 1.0e9 / (total.h2d_ms / 1000.0) : 0.0;
    const double d2h_gbps = total.d2h_ms > 0.0
        ? output_bytes / 1.0e9 / (total.d2h_ms / 1000.0) : 0.0;

    std::cout << num_rays << ',' << num_samples << ','
              << warmup_iterations << ',' << measurement_iterations << ','
              << total.h2d_ms << ',' << total.kernel_ms << ','
              << total.d2h_ms << ',' << total.end_to_end_ms << ','
              << cpu_ms << ','
              << (total.end_to_end_ms > 0.0 ? cpu_ms / total.end_to_end_ms : 0.0) << ','
              << samples_per_second / 1.0e6 << ','
              << h2d_gbps << ',' << d2h_gbps << ','
              << device_bytes / (1024.0 * 1024.0) << ',' << max_error << '\n';

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_initial_intensity));
    CHECK_CUDA(cudaFree(d_source));
    CHECK_CUDA(cudaFree(d_sigma));
    return total;
}

}  // namespace

int main(int argc, char** argv) {
    const int warmup_iterations = argc > 1 ? std::atoi(argv[1]) : 5;
    const int measurement_iterations = argc > 2 ? std::atoi(argv[2]) : 20;
    if (warmup_iterations < 0 || measurement_iterations <= 0 || argc > 3) {
        std::cerr << "usage: benchmark [warmup_iterations] [measurement_iterations]\n";
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaFree(0));  // Exclude context initialization from measurements.
    std::mt19937 generator(12345);
    constexpr float ds = 0.01f;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "num_rays,num_samples,warmup,iterations,h2d_ms,kernel_ms,d2h_ms,"
                 "end_to_end_ms,cpu_ms,speedup_kernel_samples_millions,"
                 "h2d_gbps,d2h_gbps,device_memory_mib,max_abs_error\n";

    // Fixed samples, increasing rays and fixed rays, increasing samples.
    const std::vector<std::pair<int, int>> cases = {
        {1024, 64}, {4096, 64}, {16384, 64},
        {1024, 256}, {4096, 256}, {16384, 256}
    };
    for (const auto& problem : cases) {
        benchmark_case(problem.first, problem.second,
                       warmup_iterations, measurement_iterations,
                       ds, generator);
    }
    return EXIT_SUCCESS;
}
