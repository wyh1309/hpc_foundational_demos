# CUDA Basic Programming: Device Query and Vector Addition

This module introduces the basic CUDA programming workflow: querying GPU
properties, allocating host and device memory, transferring data between CPU
and GPU, launching a kernel, measuring execution time, and verifying the
result against a CPU reference implementation.

## Project Overview

The module contains two independent CUDA examples:

- `device_info`: queries the available CUDA devices and prints their name,
  number of SMs, warp size, maximum threads per block, and global memory size.
- `vector_add`: computes element-wise vector addition on the GPU:

  ```text
  C[i] = A[i] + B[i]
  ```

The vector addition example also compares GPU kernel time with the serial CPU
implementation and reports the effective GPU global-memory bandwidth.

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Programming model | CUDA |
| Language | CUDA C++ |
| Build system | CMake |
| C++ standard | C++17 |
| CUDA standard | CUDA C++11 |
| Vector length | `N = 1 << 20` (1,048,576 `float` elements) |
| Block size | 256 threads |
| Grid size | `(N + 255) / 256` blocks |
| Verification | CPU serial reference with `1e-5` tolerance |

## Hardware Environment

The experiment was run on an NVIDIA A30 GPU. The `device_info` executable
reported the following device properties:

| Property | Value |
|----------|-------|
| GPU model | NVIDIA A30 |
| Streaming multiprocessors | 56 SMs |
| Warp size | 32 threads |
| Maximum threads per block | 1024 |
| Global memory | 25,338,052,608 bytes (approximately 23.6 GiB) |

Recorded device query output:

```text
==== GPU Device 0 ====
==== GPU Name NVIDIA A30 ====
==== SM Number 56 ====
==== Wrap Size 32 ====
==== Max Block Size 1024 ====
==== Global Memory Size 25338052608 ====
```

## Build

From the repository root:

```bash
cmake -S parallel_hpc_modules/module_04_cuda_basic \
      -B parallel_hpc_modules/module_04_cuda_basic/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build parallel_hpc_modules/module_04_cuda_basic/build --parallel
```

The build requires a CUDA-capable system, an installed CUDA toolkit, and a
CMake version that supports CUDA as a project language.

## Run

```bash
./parallel_hpc_modules/module_04_cuda_basic/build/src/device_info/device_info
./parallel_hpc_modules/module_04_cuda_basic/build/src/vector_add/vector_add
```

Depending on the CMake generator and target layout, the executable may also be
available directly under the build directory. Locate it with:

```bash
find parallel_hpc_modules/module_04_cuda_basic/build -type f -executable
```

## `device_info.cu`: CUDA Device Query

The program uses `cudaGetDeviceCount` and `cudaGetDeviceProperties` to inspect
every visible CUDA device. The output includes:

- GPU model name
- Number of streaming multiprocessors (SMs)
- Warp size
- Maximum threads per block
- Total global memory

This is a useful first diagnostic step before running CUDA kernels, because the
reported hardware limits influence block configuration and occupancy.

## `vector_add.cu`: Simple Vector Addition Kernel

The vector addition workflow is:

1. Allocate and initialize host arrays `h_a` and `h_b`.
2. Allocate device arrays `d_a`, `d_b`, and `d_c` with `cudaMalloc`.
3. Copy the input arrays from host to device.
4. Launch `vector_add_kernel` with 256 threads per block.
5. Copy the result from device to host.
6. Compute a CPU serial reference result.
7. Compare GPU and CPU results and release all resources.

The kernel maps one CUDA thread to one vector element. The `idx < n` boundary
check allows the same launch pattern to handle vector lengths that are not
exactly divisible by the block size.

CUDA runtime errors and kernel launch errors are checked explicitly with the
`CHECK_CUDA_FATAL` and `CHECK_KERNEL` macros.

## Timing Observation and Pitfall

The CUDA event timer starts immediately before the kernel launch and stops
after the kernel completes. Therefore, `GPU kernel time` includes only device
kernel execution. It excludes:

- host-to-device copies (`cudaMemcpyHostToDevice`)
- device-to-host copy (`cudaMemcpyDeviceToHost`)
- memory allocation and initialization

The printed `Speedup (CPU/GPU)` compares the CPU serial addition time with GPU
kernel-only time. It is consequently an optimistic compute-only comparison,
not an end-to-end application speedup. Vector addition performs very little
arithmetic per byte moved, so PCIe transfer overhead can be significant when
the input data originates on the CPU.

> Lesson learned: GPU kernel timing and end-to-end timing answer different
> performance questions. A fair real-world comparison must time the complete
> H2D-copy, kernel, and D2H-copy pipeline. The later CUDA matrix multiplication
> modules provide a more compute-intensive comparison.

The reported effective bandwidth is also kernel-only. It is calculated from
three global-memory operations: reading `A`, reading `B`, and writing `C`.

## A30 Benchmark Results

The following measurements were recorded on the NVIDIA A30, but with two
different vector sizes. Both runs passed numerical verification.

| Vector size | Elements | GPU kernel time | CPU serial time | Reported speedup | Effective bandwidth |
|-------------|---------:|----------------:|----------------:|-----------------:|--------------------:|
| `N = 1 << 20` | 1,048,576 | 13.9931 ms | 8.802 ms | 0.629025x | 0.837467 GB/s |
| `N = 1 << 24` | 16,777,216 | 0.47712 ms | 134.826 ms | 282.583x | 392.983 GB/s |

Recorded output for `N = 1 << 20`:

```text
GPU kernel time:    13.9931 ms
CPU serial time:    8.802 ms
Speedup (CPU/GPU):  0.629025 x
Effective bandwidth: 0.837467 GB/s
Verification passed!
```

Recorded output for `N = 1 << 24`:

```text
GPU kernel time:    0.47712 ms
CPU serial time:    134.826 ms
Speedup (CPU/GPU):  282.583 x
Effective bandwidth: 392.983 GB/s
Verification passed!
```

### Scaling Analysis

Increasing `N` from `1 << 20` to `1 << 24` increases the number of elements,
and therefore the vector data volume, by 16x. The CPU time increases from
8.802 ms to 134.826 ms, approximately 15.3x, which is close to the expected
linear scaling of serial vector addition.

The GPU results should not be interpreted as a 29x improvement caused only by
the larger input. The first `N = 1 << 20` measurement is a very small workload
and its 13.9931 ms kernel time is dominated by fixed overhead such as CUDA
context initialization or first-use kernel loading. In contrast, the larger
`N = 1 << 24` workload amortizes this fixed cost and exposes the A30's highly
parallel memory-throughput behavior. This is also reflected by the effective
bandwidth increasing from 0.837467 GB/s to 392.983 GB/s.

The `N = 1 << 24` result therefore provides a more representative
kernel-throughput measurement, but it is still kernel-only. It does not
include the host-to-device and device-to-host transfers, so the reported
282.583x speedup is not an end-to-end application speedup. A rigorous scaling
study should compile or parameterize both problem sizes, warm up the GPU, run
multiple repetitions, and report separate kernel-only and H2D-kernel-D2H
timings.

## Skills Demonstrated

- CUDA device property inspection
- CUDA kernel launch configuration
- Host/device memory allocation and data transfer
- CUDA event timing and CPU `chrono` timing
- CUDA runtime and kernel error checking
- GPU result verification against a CPU ground truth
- Interpreting memory-bound GPU benchmark results
