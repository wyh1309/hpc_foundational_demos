#pragma once

float solve_transport_cpu(
    const float* sigma,
    const float* q,
    int num_samples,
    float ds,
    float I0);

// sigma and source are row-major [ray][sample]. Each ray has its own
// coefficients and initial intensity.
void solve_transport_cpu_multi(
    const float* sigma,
    const float* source,
    const float* initial_intensity,
    float* output,
    int num_rays,
    int num_samples,
    float ds);
