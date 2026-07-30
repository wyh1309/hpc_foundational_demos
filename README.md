# HPC Parallel Optimization & NeRF Acceleration Project
## Project Overview
This repository records a complete technical path: early NeRF 3D reconstruction research (2024 MIT xPRO) → systematic learning of parallel computing & customized GPU acceleration for neural volume rendering (2026).

Most parallel learning materials only contain basic MPI/OpenMP/CUDA demos without real industry-oriented application scenarios. This project combines fundamental high-performance computing implementations with optimized NeRF rendering kernels, forming a closed-loop from theoretical bottleneck discovery to engineering acceleration.

## Two Core Parts
### 1. 2024 MIT xPRO NeRF Research
Folder: `2024_mit_nerf_research/`
- Official MIT xPRO TechXcelerate completion certificate (verifiable official certificate with CEU credits)
- Full comparative experiment report: evaluate image/video input pipelines for NeRF 3D avatar reconstruction
- Core findings: Vanilla NeRF suffers severe rendering latency and heavy ray sampling computation overhead, which inspires the subsequent HPC optimization practice.

### 2. Modular HPC Parallel Computing Implementation
Folder: `hpc_parallel_modules/`
10 independent, self-contained practice modules covering multi-threading, distributed computing and GPU optimization:
- module_01 ~ module_06: CPU baseline, OpenMP multi-threading, MPI communication, naive & tiled CUDA matrix multiplication (memory optimization practice)
- module_07: Custom CUDA volume rendering kernel optimized for NeRF ray sampling
- module_08 ~ module_10: Performance profiling template, unified project refactor & final performance benchmark

## Environment & Build
All modules adopt cross-platform CMake build system, support C++ / OpenMP / MPI / CUDA compilation separately.