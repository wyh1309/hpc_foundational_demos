# OpenMP Matrix Multiplication Optimization (CPU Cache Tiling)

This project implements and benchmarks three classic dense matrix multiplication versions to demonstrate HPC CPU optimization techniques, including serial baseline, naive OpenMP parallelism, and cache tiling optimization.

---

## Project Overview

- **Serial naive IJK matrix multiplication** (baseline)
- **OpenMP multi-threaded parallel matrix multiplication**
- **OpenMP + cache blocking (tiling) optimized matrix multiplication**

All codes follow high-performance computing optimization logic:

- Multi-core parallelism + memory locality optimization

---

## Experimental Environment

| Parameter | Value |
|-----------|-------|
| CPU Cores | 4 |
| Compiler | GCC + OpenMP |
| Optimization Flag | `-O3 -fopenmp` |
| Matrix Size | 1024 × 1024 |
| Tile Block Size | 32 |

---

## Performance Results

```text
Serial time:      4.54388 s
OMP naive time:   1.24872 s
OMP tiled time:   1.03977 s
Speedup (naive):  3.63884
Speedup (tiled):  4.37009
```

---

## Result Analysis

### 1. Naive OpenMP Parallel Speedup

- The outermost loop is parallelized with `schedule(static)`, which perfectly fits the uniform workload of dense matrix multiplication.
- 4-core parallel speedup reaches **3.64**
- Parallel efficiency ≈ **91%**
- Very low thread scheduling & fork-join overhead

### 2. Cache Tiling Optimization Gain

- The naive IJK algorithm causes severe cache miss due to irregular memory access of matrix B.
- Tiling optimization splits the entire matrix into small blocks that fit in CPU L1/L2 cache, significantly improving data locality.
- Under the same 4 threads, tiling further reduces running time by **16.7%**
- Final overall speedup reaches **4.37**

### 3. Why Speedup Exceeds CPU Cores (4)

This is a typical HPC optimization phenomenon:

- **Amdahl's law limit (4×)** only restricts pure multi-thread parallelism
- Tiling optimizes the **serial memory access efficiency**
- The overall acceleration includes both **multi-core parallelism** and **single-core cache optimization**, so breaking the core-number limit is reasonable

---

## Compile & Run Command

```bash
g++ matmul.cpp -O3 -fopenmp -o matmul
./matmul
```

---

## Core Technical Points

| Technique | Description |
|-----------|-------------|
| **OpenMP static scheduling** | Best choice for equal-load computing tasks such as dense matrix multiplication. |
| **Private variable implicit rules** | Loop variables defined in `for()` are automatically private in C++11 and later. |
| **Cache Tiling / Blocking** | Core CPU HPC optimization: reduce DRAM access, maximize cache hit rate. |

### Code Style Iteration

- **Early prototype code** uses `using namespace std` for rapid implementation
- **Formal optimized code** uses fully qualified `std::` prefix
- Standard industrial C++ style to avoid naming conflicts with MPI / CUDA / BLAS libraries

---

## Skills Demonstrated

- C++ high-performance programming
- OpenMP multi-thread parallel computing
- CPU cache locality optimization
- Performance benchmark and speedup analysis

## Experience
I initially implemented cache tiling without boundary checks, which caused out-of-bounds memory access when matrix size was not divisible by block size. I fixed this issue with std::min(ii + BLOCK, N) to ensure general-purpose compatibility with arbitrary matrix dimensions.