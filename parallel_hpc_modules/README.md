# Modular HPC Parallel Computing Practice
## Introduction
Based on the performance bottlenecks summarized from the 2024 NeRF research, this part systematically implements mainstream parallel acceleration technologies, and finally realizes a customized NeRF rendering CUDA kernel as the landing application.

## Module List
- module_01_matmul_cpu: Baseline CPU matrix multiplication with timing function
- module_02_matmul_omp: OpenMP multi-thread parallel matrix multiplication
- module_03_mpi_basic: MPI distributed parallel summation demo
- module_04_cuda_vector_add: CUDA basic vector addition
- module_05_cuda_matmul_naive: Global memory naive CUDA matrix multiplication
- module_06_cuda_matmul_tiled: Shared memory tiled matrix multiplication optimization
- module_07_nerf_render_kernel: Self-implemented CUDA volume rendering kernel for NeRF ray sampling acceleration
- module_08_profiling_cmake: General CMake template with performance profiling script
- module_09_project_refactor: Unified code specification & project structure sorting
- module_10_final_cleanup: Performance test logs & overall speedup comparison

## Compile Rule
Each module contains independent CMakeLists.txt, can be compiled and run separately.