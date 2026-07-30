cmake_minimum_required(VERSION 3.20)

project(module_05_cuda_matmul_naive LANGUAGES CXX CUDA)

# Default to an optimized build for single-config generators (Ninja/Makefiles).
# An explicit -DCMAKE_BUILD_TYPE=Debug (or another type) always takes precedence.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# WSL/Ubuntu HPC benchmark configuration: retain custom flags while enforcing O3.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
    string(REGEX REPLACE "(^|[ \t])-O[0-3gs]($|[ \t])" "\\1-O3\\2"
        CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    if(NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "(^|[ \t])-O3($|[ \t])")
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
    endif()
    if(NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "(^|[ \t])-DNDEBUG($|[ \t])")
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -DNDEBUG")
    endif()
endif()
if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA|Clang")
    string(REGEX REPLACE "(^|[ \t])-O[0-3gs]($|[ \t])" "\\1-O3\\2"
        CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
    if(NOT CMAKE_CUDA_FLAGS_RELEASE MATCHES "(^|[ \t])-O3($|[ \t])")
        string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -O3")
    endif()
    if(NOT CMAKE_CUDA_FLAGS_RELEASE MATCHES "(^|[ \t])-DNDEBUG($|[ \t])")
        string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -DNDEBUG")
    endif()
endif()

add_subdirectory(src)
