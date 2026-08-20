#pragma once

// One CUDA thread should process one ray. The sample loop belongs inside this
// kernel and should implement the discrete absorption-emission update.
__global__ void volume_transport_kernel(
    const float* sigma,
    const float* source,
    float* output,
    int num_rays,
    int num_samples,
    float ds,
    float initial_intensity);
