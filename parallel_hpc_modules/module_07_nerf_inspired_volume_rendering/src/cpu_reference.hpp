#pragma once

float solve_transport_cpu(
    const float* sigma,
    const float* source,
    int num_samples,
    float ds,
    float initial_intensity);
