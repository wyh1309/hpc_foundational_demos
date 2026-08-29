#pragma once

#include <cmath>

namespace transport_detail {

#ifdef __CUDACC__
#define TRANSPORT_HOST_DEVICE __host__ __device__
#else
#define TRANSPORT_HOST_DEVICE
#endif

inline TRANSPORT_HOST_DEVICE
float transport_step(float I, float sigma, float q, float ds)
{
    const float tau = sigma * ds;

    if (fabsf(tau) < 1e-5f) {
        return I + q * ds - I * tau;
    }

    const float one_minus_T = -expm1f(-tau);
    return I * (1.0f - one_minus_T) + (q / sigma) * one_minus_T;
}

#undef TRANSPORT_HOST_DEVICE

}  // namespace transport_detail
