cmake_minimum_required(VERSION 3.18)
project(Module04CudaBasic LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(src/device_info)
add_subdirectory(src/vector_add)
