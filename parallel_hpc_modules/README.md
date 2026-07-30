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

## Build Configuration
Each module contains an independent `CMakeLists.txt` and can be configured, built,
and run separately. On WSL/Ubuntu with a single-config generator such as Ninja or
Unix Makefiles, a fresh build directory defaults to `Release` with `-O3 -DNDEBUG`
for reproducible HPC performance measurements. Existing Release-specific flags are
preserved; only the optimization level and `NDEBUG` definition are normalized. The
configuration keeps C++17 and all existing target, dependency, and link definitions
unchanged.

```bash
cmake -S module_01_matmul_cpu -B module_01_matmul_cpu/build
cmake --build module_01_matmul_cpu/build --parallel
```

To debug a module, explicitly select `Debug` in a separate build directory:

```bash
cmake -S module_01_matmul_cpu -B module_01_matmul_cpu/build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build module_01_matmul_cpu/build-debug --parallel
```

For VSCode CMake Tools, an unset build type inherits the CMake default `Release`.
The build type selected in the extension is treated as explicit input, so select
`Debug` and reconfigure when debugging is needed. Existing build directories retain
their cached build type; use a new build directory or pass
`-DCMAKE_BUILD_TYPE=Release` to convert an earlier Debug configuration.
