#include "cpu_forward.hpp"
#include <cmath>

namespace {

float solve_one_ray(
    const float* sigma,
    const float* q,
    int num_samples,
    float ds,
    float I0)
{
    float I = I0;
    for (int i = 0; i < num_samples; ++i) {
        const float tau = sigma[i] * ds;
        if (std::fabs(tau) < 1e-5f) {
            I += q[i] * ds - I * tau;
        } else {
            const float one_minus_T = -std::expm1(-tau);
            I = I * (1.0f - one_minus_T) + (q[i] / sigma[i]) * one_minus_T;
        }
    }
    return I;
}

}  // namespace

float solve_transport_cpu(
    const float* sigma,
    const float* q,
    int num_samples,
    float ds,
    float I0)
{
    return solve_one_ray(sigma, q, num_samples, ds, I0);
}

void solve_transport_cpu_multi(
    const float* sigma,
    const float* source,
    const float* initial_intensity,
    float* output,
    int num_rays,
    int num_samples,
    float ds)
{
    for (int ray = 0; ray < num_rays; ++ray) {
        const int offset = ray * num_samples;
        output[ray] = solve_one_ray(
            sigma + offset,
            source + offset,
            num_samples,
            ds,
            initial_intensity[ray]);
    }
}
