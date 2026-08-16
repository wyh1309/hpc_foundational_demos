# Tiled CUDA Matrix Multiplication with Shared Memory

## 1. Overview

This module optimizes dense matrix multiplication with CUDA shared-memory
tiling and compares it with the naive global-memory kernel from
`module_05_cuda_matmul_naive`.

The program computes:

`C(M, P) = A(M, N) * B(N, P)`

Each thread produces one element of `C`. In the tiled kernel, a thread block
cooperatively loads tiles of `A` and `B` into shared memory. The block then
reuses these values for the dot-product calculation, reducing repeated global
memory reads.

## 2. Implementation Details

- Programming model: CUDA
- Language: CUDA C++17
- Build system: CMake
- Matrix element type: `float`
- Matrix dimensions: `M = 1024`, `N = 1024`, `P = 1024`
- Tiled block size: `32 x 32` threads
- Tile width: `32`
- Naive comparison block size: `32 x 32` threads
- Grid dimensions: `ceil(P / block.x)` by `ceil(M / block.y)`
- CPU reference: serial matrix multiplication
- GPU timing: CUDA events
- Numerical validation: absolute error tolerance `1e-3`

The tiled kernel allocates two block-private shared-memory arrays:

```cpp
__shared__ float As[TILE_WIDTH][TILE_WIDTH];
__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
```

For every tile along the reduction dimension `N`, threads load one element of
`A` and one element of `B`, synchronize, compute the partial dot product from
shared memory, and synchronize again before loading the next tile. Out-of-range
elements are filled with zero, so the same kernel also supports dimensions that
are not multiples of `TILE_WIDTH`.

## 3. Kernel Comparison

### Naive kernel

The baseline kernel maps one thread to one output element and reads every
operand directly from device global memory:

```text
C[row, col] = sum(A[row, k] * B[k, col])
```

For each output element, the same input values are repeatedly fetched by
different threads. This makes global-memory traffic the main optimization
target.

### Tiled shared-memory kernel

The tiled kernel divides the matrices into `32 x 32` tiles. Values loaded once
by a block are reused by all threads in that block while accumulating the
output element. The two `__syncthreads()` calls protect tile loading and prevent
threads from overwriting shared-memory data before all computations finish.

Both kernels use boundary checks for rows, columns, and tile loads. This keeps
the output shape correct for general `M x N` multiplied by `N x P` matrices.

## 4. Build & Run

From the repository root:

```bash
cmake -S parallel_hpc_modules/module_06_cuda_matmul_tiled \
      -B parallel_hpc_modules/module_06_cuda_matmul_tiled/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build parallel_hpc_modules/module_06_cuda_matmul_tiled/build --parallel
./parallel_hpc_modules/module_06_cuda_matmul_tiled/build/src/cuda_matmul_tiled
```

The build requires CMake 3.20 or newer, a CUDA toolkit, and an NVIDIA GPU
supported by the installed CUDA compiler. A fresh single-configuration build
defaults to `Release` with the project's optimized `-O3 -DNDEBUG` settings.

## 5. Benchmark Results

Recorded on an NVIDIA A30 with random `float` input matrices. Both GPU kernels
passed numerical validation.

```text
=============================================
Matrix Size: M=1024 N=1024 P=1024
Tile Width:  32
---------------------------------------------
CPU Serial Time:            3843.89 ms
---------------------------------------------
[GPU Kernel-Only Time (pure compute)]
Naive GPU Kernel:           9.65645 ms | Valid: YES
Tiled Shared GPU Kernel:    1.61488 ms | Valid: YES
---------------------------------------------
[GPU End-to-End Time (kernel + DtoH copy)]
Naive GPU E2E:              14.3385 ms
Tiled Shared GPU E2E:       6.0457 ms
---------------------------------------------
Speedup CPU / Naive(Kernel):    398.065 x
Speedup CPU / Tiled(Kernel):    2380.3 x
Speedup Tiled / Naive(Kernel):  5.97967 x
---------------------------------------------
Speedup CPU / Naive(E2E):       268.081 x
Speedup CPU / Tiled(E2E):       635.806 x
=============================================
```

| Metric | Naive kernel | Tiled shared-memory kernel |
|--------|-------------:|---------------------------:|
| Kernel-only time | 9.65645 ms | 1.61488 ms |
| Kernel-only speedup over CPU | 398.065x | 2380.3x |
| End-to-end time | 14.3385 ms | 6.0457 ms |
| End-to-end speedup over CPU | 268.081x | 635.806x |

The tiled kernel is approximately `5.98x` faster than the naive kernel when
measuring kernel execution only. The improvement is smaller for the reported
end-to-end path because the DtoH transfer and synchronization overhead become a
larger part of the total time.

For this benchmark, the matrix multiplication performs
`2 * M * N * P = 2,147,483,648` floating-point operations. The corresponding
kernel-only throughput is approximately `222.4 GFLOPS` for the naive kernel
and `1.33 TFLOPS` for the tiled kernel.

## 6. Timing and Validation Notes

- CUDA events measure device-side elapsed time and are appropriate for kernel
  timing because CUDA launches are asynchronous with respect to the CPU.
- The kernel-only interval measures pure kernel execution.
- The E2E interval in this implementation measures kernel execution plus the
  device-to-host copy. Host-to-device input copies occur before the interval
  starts, so this is not a complete HtoD + kernel + DtoH application time.
- CPU and GPU accumulation order can differ, so floating-point results should
  be checked with a tolerance rather than exact equality.
- Small, unaligned dimensions such as `M=997`, `N=997`, and `P=997` are useful
  additional tests for tile boundary handling.

## 7. Key Lessons

- Shared-memory tiling improves data reuse and substantially reduces repeated
  global-memory accesses.
- `grid.x` covers output columns `P`, while `grid.y` covers output rows `M`.
- The output allocation and DtoH copy size must be `M * P * sizeof(float)`.
- Synchronization is required both after loading a tile and before reusing the
  shared-memory buffers for the next tile.
- Kernel-only speedup and end-to-end speedup describe different performance
  questions and should be reported separately.
