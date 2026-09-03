#include "cuda_forward.cuh"
#include "cpu_forward.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

bool check_cuda(cudaError_t error, const char* operation)
{
    if (error == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed: " << cudaGetErrorString(error) << '\n';
    return false;
}

bool close_enough(float value, float reference, float tolerance)
{
    return std::fabs(value - reference) <= tolerance;
}

}  // namespace

int main()
{
    constexpr int num_rays = 3;
    constexpr int num_samples = 7;
    constexpr float ds = 0.05f;
    constexpr float tolerance = 1.0e-5f;

    // This case exercises the zero-absorption branch. The analytic result is
    // I(L) = I0 + ds * sum(source), independently for each ray.
    std::vector<float> sigma(num_rays * num_samples, 0.0f);
    std::vector<float> source(num_rays * num_samples);
    std::vector<float> initial{0.25f, 1.5f, -0.5f};
    std::vector<float> cpu_output(num_rays, 0.0f);
    std::vector<float> cuda_output(num_rays, 0.0f);
    std::vector<float> analytic_output(num_rays, 0.0f);

    for (int ray = 0; ray < num_rays; ++ray) {
        for (int sample = 0; sample < num_samples; ++sample) {
            const float value = 0.2f * static_cast<float>(sample + 1)
                              + 0.1f * static_cast<float>(ray);
            source[ray * num_samples + sample] = value;
            analytic_output[ray] += value * ds;
        }
        analytic_output[ray] += initial[ray];
    }

    solve_transport_cpu_multi(
        sigma.data(), source.data(), initial.data(), cpu_output.data(),
        num_rays, num_samples, ds);

    float* d_sigma = nullptr;
    float* d_source = nullptr;
    float* d_initial = nullptr;
    float* d_output = nullptr;
    const std::size_t sample_bytes = sigma.size() * sizeof(float);
    const std::size_t ray_bytes = initial.size() * sizeof(float);

    bool passed = true;
    passed &= check_cuda(cudaMalloc(&d_sigma, sample_bytes), "cudaMalloc(d_sigma)");
    passed &= check_cuda(cudaMalloc(&d_source, sample_bytes), "cudaMalloc(d_source)");
    passed &= check_cuda(cudaMalloc(&d_initial, ray_bytes), "cudaMalloc(d_initial)");
    passed &= check_cuda(cudaMalloc(&d_output, ray_bytes), "cudaMalloc(d_output)");
    if (!passed) {
        cudaFree(d_output);
        cudaFree(d_initial);
        cudaFree(d_source);
        cudaFree(d_sigma);
        return EXIT_FAILURE;
    }

    passed &= check_cuda(cudaMemcpy(
        d_sigma, sigma.data(), sample_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(sigma)");
    passed &= check_cuda(cudaMemcpy(
        d_source, source.data(), sample_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(source)");
    passed &= check_cuda(cudaMemcpy(
        d_initial, initial.data(), ray_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(initial)");
    passed &= check_cuda(launch_volume_transport_cuda(
        d_sigma, d_source, d_output, num_rays, num_samples, ds, d_initial),
        "launch_volume_transport_cuda");
    passed &= check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    passed &= check_cuda(cudaMemcpy(
        cuda_output.data(), d_output, ray_bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy(output)");

    for (int ray = 0; ray < num_rays; ++ray) {
        const bool cpu_ok = close_enough(cpu_output[ray], analytic_output[ray], tolerance);
        const bool cuda_ok = close_enough(cuda_output[ray], analytic_output[ray], tolerance);
        const bool match = close_enough(cuda_output[ray], cpu_output[ray], tolerance);
        if (!cpu_ok || !cuda_ok || !match) {
            std::cerr << "ray " << ray
                      << ": cpu=" << cpu_output[ray]
                      << ", cuda=" << cuda_output[ray]
                      << ", analytic=" << analytic_output[ray] << '\n';
            passed = false;
        }
    }

    cudaFree(d_output);
    cudaFree(d_initial);
    cudaFree(d_source);
    cudaFree(d_sigma);

    if (passed) {
        std::cout << "CUDA zero-absorption test passed\n";
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
