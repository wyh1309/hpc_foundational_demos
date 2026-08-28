#include "cuda_forward.cuh"

__global__ void volume_transport_kernel(
    const float* sigma,
    const float* source,
    float* output,
    int num_rays,
    int num_samples,
    float ds,
    const float* initial_intensity
)
{
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray < num_rays) {
        const int offset = ray * num_samples;
        float I = initial_intensity[ray];
        for (int sample = 0; sample < num_samples; ++sample) {
            const float sigma_i = sigma[offset + sample];
            const float tau = sigma_i * ds;
            const float q_i = source[offset + sample];
            if (fabsf(tau) < 1e-5f) {
                I += q_i * ds - I * tau;
            } else {
                const float one_minus_T = -expm1f(-tau);
                I = I * (1.0f - one_minus_T)
                    + (q_i / sigma_i) * one_minus_T;
            }
        }
        output[ray] = I;
    }
}

cudaError_t launch_volume_transport_cuda(
    const float* sigma,
    const float* source,
    float* output,
    int num_rays,
    int num_samples,
    float ds,
    const float* initial_intensity,
    cudaStream_t stream)
{
    if (num_rays < 0 || num_samples < 0 || ds < 0.0f) {
        return cudaErrorInvalidValue;
    }
    if (num_rays == 0) {
        return cudaSuccess;
    }

    constexpr int threads_per_block = 256;
    const int blocks = (num_rays + threads_per_block - 1) / threads_per_block;
    volume_transport_kernel<<<blocks, threads_per_block, 0, stream>>>(
        sigma, source, output, num_rays, num_samples, ds, initial_intensity);
    return cudaGetLastError();
}
