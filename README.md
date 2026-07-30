# hpc_foundation_demos
A collection of foundational high‑performance computing practice projects.
Covers CPU baseline, OpenMP multi‑thread parallelism, MPI distributed computing, and CUDA GPU acceleration.
A lightweight NeRF volume‑rendering kernel prototype is included as a practical acceleration case.

## Modules
- module_01: Naive CPU matrix multiplication with timer
- module_02: Matrix multiplication with OpenMP
- module_03: Basic MPI reduction demo
- module_04: CUDA vector addition
- module_05: Naive CUDA matrix multiplication (global memory)
- module_06: Tiled matrix multiplication with shared memory optimization
- module_07: Simplified NeRF rendering CUDA kernel
- module_08: Performance profiling & unified CMake template
- module_09: Project structure refactoring
- module_10: Final cleanup & benchmark summary

## Dependencies
- GCC / G++
- OpenMP
- MPI (MPICH / OpenMPI)
- CUDA Toolkit
- CMake >= 3.18