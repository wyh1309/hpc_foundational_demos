#pragma once

#include <cuda_runtime.h>

// sigma and source are row-major [ray][sample]. One CUDA thread processes one
// ray; samples along that ray remain sequential because of the transport
// recurrence.
__global__ void volume_transport_kernel(
    const float* sigma,
    const float* source,
    float* output,
    int num_rays,
    int num_samples,
    float ds,
    const float* initial_intensity);

// Launches volume_transport_kernel with one thread per ray.
cudaError_t launch_volume_transport_cuda(
    const float* sigma,
    const float* source,
    float* output,
    int num_rays,
    int num_samples,
    float ds,
    const float* initial_intensity,
    cudaStream_t stream = 0);
