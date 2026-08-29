#include "cpu_forward.hpp"
#include "transport_step.hpp"
#include <cmath>

namespace {

float solve_one_ray(
    const float* sigma,
    const float* source,
    int num_samples,
    float ds,
    float I0)
{
    float I = I0;
    for (int i = 0; i < num_samples; ++i) {
        I = transport_detail::transport_step(
            I, sigma[i], source[i], ds);
    }
    return I;
}

}  // namespace

float solve_transport_cpu(
    const float* sigma,
    const float* source,
    int num_samples,
    float ds,
    float I0)
{
    return solve_one_ray(sigma, source, num_samples, ds, I0);
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
