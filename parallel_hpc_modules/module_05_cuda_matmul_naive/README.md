# Naive CUDA Matrix Multiplication

## 1. Overview

This module implements a baseline CUDA matrix multiplication benchmark:

`C(M, P) = A(M, N) * B(N, P)`

The kernel is intentionally naive. Each CUDA thread computes one output element
of `C` by reading `A` and `B` directly from global memory. The launch uses a
`16 x 16` thread block and does not use shared-memory tiling.

This module is useful as the first GPU dense-matrix multiplication reference
before introducing tiled shared-memory kernels, register blocking, memory
coalescing improvements, or cuBLAS comparisons.

## 2. Implementation Details

- Programming model: CUDA
- Language: CUDA C++ / C++17
- Build system: CMake
- Matrix type: `float`
- Kernel: one thread computes one `C[row, col]`
- Block size: `dim3(16, 16)`
- Grid size: `ceil(P / 16)` by `ceil(M / 16)`
- Memory path: host memory to device global memory to host memory
- Tiling: none
- CPU reference: serial matrix multiplication
- GPU timing: `cudaEventRecord` and `cudaEventElapsedTime`

The row and column mapping is:

- `row = blockIdx.y * blockDim.y + threadIdx.y`
- `col = blockIdx.x * blockDim.x + threadIdx.x`
- `C[row * P + col] = sum(A[row * N + k] * B[k * P + col])`

Boundary checks are required because dimensions such as `M=13`, `N=17`, and
`P=19` are not aligned to the `16 x 16` block shape.

## 3. Features

- Naive global-memory CUDA matrix multiplication
- End-to-end benchmark flow covering H2D copy, kernel execution, and D2H copy
- Explicit CUDA runtime error checking
- Kernel launch error checking
- CPU serial reference validation
- CUDA event based GPU timing
- Small unaligned-dimension test case for boundary and grid validation
- Large matrix benchmark for throughput estimation

## 4. Observed Benchmark & Debug Analysis

Benchmark log 1: small unaligned case, `M=13`, `N=17`, `P=19`

> GPU time: 2.55 ms  
> CPU serial time: 0.003 ms  
> Speedup: 0.001x  
> Verification: PASS

This is a tiny workload. The CPU timing is mostly measurement noise because the
serial work is too small for a reliable host-side timer measurement. The GPU
time is dominated by fixed overhead such as launch latency, runtime scheduling,
and synchronization. A speedup below `1x` is expected for this case.

Benchmark log 2: large matrix, `M=1024`, `N=1024`, `P=1024`

> GPU time: 7.37 ms  
> CPU serial time: 3819 ms  
> Speedup: 518x  
> Observed bug: printed decimal values looked identical, but verification failed.

Root cause: the CPU and GPU do not necessarily accumulate floating-point sums
in the same order. The arithmetic is mathematically equivalent, but `float`
rounding can differ by a few ULPs. Exact `==` or `!=` comparison is therefore
wrong for floating-point validation. Standard `printf("%f")` prints only six
decimal digits and can hide binary-level differences that still break exact
comparison.

Correct validation should use a combined absolute and relative tolerance. The
following function computes the serial CPU reference and validates the GPU
matrix with `eps_abs = 1e-4f` and `eps_rel = 1e-4f`. Mismatches are printed
with `%.8f` values and scientific-notation diff output.

```cpp
bool cpu_verify(const float* h_a,
                const float* h_b,
                const float* h_c_gpu,
                int M,
                int N,
                int P)
{
    const float eps_abs = 1e-4f;
    const float eps_rel = 1e-4f;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float ref = 0.0f;

            for (int k = 0; k < N; ++k) {
                ref += h_a[i * N + k] * h_b[k * P + j];
            }

            const float got = h_c_gpu[i * P + j];
            const float diff = fabsf(got - ref);
            const float threshold = eps_abs + eps_rel * fabsf(ref);

            if (diff > threshold) {
                printf("Mismatch at C[%d,%d]: GPU=%.8f CPU=%.8f diff=%.8e threshold=%.8e\n",
                       i, j, got, ref, diff, threshold);
                return false;
            }
        }
    }

    return true;
}
```

For the `1024 x 1024 x 1024` benchmark, the output matrix has shape
`1024 x 1024`. The total work is:

`2 * M * N * P = 2 * 1024 * 1024 * 1024 = 2,147,483,648 FLOP`

With a measured GPU time of approximately `7.37 ms`, the naive global-memory
kernel reaches roughly `286 GFLOPS`. This is a reasonable baseline result for
an untiled implementation that repeatedly reads matrix operands from global
memory.

## 5. Key Fixes & Lessons Learned

- Grid dimension mapping must follow the output matrix shape. `grid.x` covers
  columns `P`, and `grid.y` covers rows `M`.
- The D2H copy size must be `M * P * sizeof(float)`, because `C` has shape
  `M x P`. Using `N` in this copy is incorrect unless `N == P`.
- Host reference buffers allocated by `malloc` contain uninitialized garbage.
  They must be initialized before accumulation, or each output element must be
  assigned from a local zero-initialized accumulator.
- Unaligned dimensions such as `M=13`, `N=17`, and `P=19` are necessary tests.
  They expose boundary-check, grid-shape, and copy-size bugs that square
  aligned matrices can hide.
- Float equality is a validation trap. CPU and GPU accumulation can differ at
  the binary rounding level even when printed decimal values look identical.
- `memset` works for zero-initializing `float` arrays only because all-zero
  bytes represent `0.0f`. Never use `memset` to initialize a `float` array to
  any non-zero numeric value.
- Small matrix GPU benchmarks are dominated by launch and synchronization
  overhead. They are correctness tests, not throughput tests.

## 6. Build & Run

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./matmul_naive
```

If the executable name follows the current CMake target, run:
`./src/cuda_matmul_naive`

## 7. Important Notes

- Use CUDA events for GPU timing. Do not use `std::chrono` to measure kernel
  execution directly, because kernel launches are asynchronous with respect to
  the host.
- Report H2D copy, kernel time, and D2H copy separately when the goal is
  performance diagnosis.
- End-to-end time and kernel-only time answer different questions. Keep them
  separate in benchmark output.
- The naive kernel is a baseline, not an optimized GEMM implementation.
- Higher performance requires reducing global-memory traffic with shared-memory
  tiling or using a production library such as cuBLAS.
