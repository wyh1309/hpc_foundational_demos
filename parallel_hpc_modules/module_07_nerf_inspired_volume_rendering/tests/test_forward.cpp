#include "cpu_forward.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

float solve_constant_reference(
    float sigma, float q, int num_samples, float ds, float I0)
{
    const float length = num_samples * ds;
    if (std::fabs(sigma) < 1e-5f) {
        return I0 + q * length;
    }

    const float transmission = std::exp(-sigma * length);
    return I0 * transmission + (q / sigma) * (1.0f - transmission);
}

bool check_close(const char* name, float value, float reference, float tolerance)
{
    const float error = std::fabs(value - reference);
    const bool passed = error <= tolerance;
    std::cout << name << (passed ? " pass" : " fail")
              << ": value=" << value
              << ", reference=" << reference
              << ", abs_error=" << error << '\n';
    return passed;
}

// q(s) = source_ratio * sigma(s) makes the variable-coefficient solution
// exact: I(L) = source_ratio + (I0 - source_ratio) exp(-integral sigma ds).
float variable_exact_reference(float I0, float source_ratio, float integral_sigma)
{
    return source_ratio
        + (I0 - source_ratio) * std::exp(-integral_sigma);
}

float solve_variable_case(int num_samples)
{
    const float length = 1.0f;
    const float I0 = 0.3f;
    const float source_ratio = 2.0f;
    const float ds = length / num_samples;

    std::vector<float> sigma(num_samples);
    std::vector<float> q(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        const float s_mid = (i + 0.5f) * ds;
        const float sigma_mid = 0.5f + 0.8f * s_mid + 0.3f * s_mid * s_mid;
        sigma[i] = sigma_mid;
        q[i] = source_ratio * sigma_mid;
    }

    return solve_transport_cpu(sigma.data(), q.data(), num_samples, ds, I0);
}

}  // namespace

bool test_multi_ray()
{
    constexpr int rays = 3;
    constexpr int samples = 4;
    const float sigma[rays * samples] = {
        0.4f, 0.7f, 1.1f, 0.8f,
        0.0f, 0.3f, 0.9f, 0.2f,
        2.0f, 1.5f, 0.5f, 1.0f};
    const float source[rays * samples] = {
        0.1f, 0.4f, 0.9f, 0.2f,
        1.0f, 0.8f, 0.2f, 1.3f,
        0.4f, 0.1f, 0.8f, 0.3f};
    const float initial[rays] = {0.1f, 4.0f, 0.0f};
    float output[rays] = {};

    solve_transport_cpu_multi(
        sigma, source, initial, output, rays, samples, 0.5f);

    bool passed = true;
    for (int ray = 0; ray < rays; ++ray) {
        const int offset = ray * samples;
        const float expected = solve_transport_cpu(
            sigma + offset, source + offset, samples, 0.5f, initial[ray]);
        const std::string name = "multi-ray " + std::to_string(ray);
        passed &= check_close(name.c_str(),
                             output[ray], expected, 1e-6f);
    }
    return passed;
}

int main()
{
    bool all_passed = true;

    all_passed &= test_multi_ray();

    // Constant-coefficient regression tests.
    {
        const int num_samples = 1024;
        const float ds = 1e-3f;
        std::vector<float> sigma(num_samples, 1.0f);
        std::vector<float> q(num_samples, 2.0f);

        all_passed &= check_close(
            "constant coefficients",
            solve_transport_cpu(sigma.data(), q.data(), num_samples, ds, 0.1f),
            solve_constant_reference(1.0f, 2.0f, num_samples, ds, 0.1f),
            1e-3f);

        std::fill(sigma.begin(), sigma.end(), 0.0f);
        all_passed &= check_close(
            "zero absorption",
            solve_transport_cpu(sigma.data(), q.data(), num_samples, ds, 6.0f),
            solve_constant_reference(0.0f, 2.0f, num_samples, ds, 6.0f),
            1e-3f);
    }

    // Variable coefficients: sigma(s) = 0.5 + 0.8s + 0.3s^2 and q(s) = 2sigma(s).
    {
        const float I0 = 0.3f;
        const float source_ratio = 2.0f;
        const float integral_sigma = 0.5f + 0.4f + 0.1f;
        const float numerical = solve_variable_case(128);
        const float reference =
            variable_exact_reference(I0, source_ratio, integral_sigma);
        all_passed &= check_close(
            "variable coefficients", numerical, reference, 2e-5f);
    }

    // The midpoint samples should converge as ds decreases.
    {
        const float I0 = 0.3f;
        const float source_ratio = 2.0f;
        const float integral_sigma = 1.0f;
        const float reference =
            variable_exact_reference(I0, source_ratio, integral_sigma);
        const std::vector<int> sample_counts{16, 32, 64, 128};
        std::vector<float> errors;

        std::cout << std::setprecision(8) << "step convergence:\n";
        for (int num_samples : sample_counts) {
            const float value = solve_variable_case(num_samples);
            const float error = std::fabs(value - reference);
            errors.push_back(error);
            std::cout << "  N=" << num_samples
                      << ", ds=" << 1.0f / num_samples
                      << ", abs_error=" << error << '\n';
        }

        // Require monotone convergence and at least a modest reduction at
        // each refinement; this avoids over-constraining floating-point order.
        for (std::size_t i = 1; i < errors.size(); ++i) {
            if (!(errors[i] < errors[i - 1] &&
                  errors[i] <= 0.8f * errors[i - 1])) {
                all_passed = false;
                std::cout << "step convergence fail at refinement " << i << '\n';
            }
        }
        if (all_passed) {
            std::cout << "step convergence pass\n";
        }
    }

    return all_passed ? 0 : 1;
}
