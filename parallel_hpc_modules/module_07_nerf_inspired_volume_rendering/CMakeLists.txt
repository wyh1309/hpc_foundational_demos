# CMakeLists.txt for Constant-Parameter Inverse Problem Solver

cmake_minimum_required(VERSION 3.20)

project(module_07_nerf_inspired_volume_rendering LANGUAGES CXX CUDA)

# Build type configuration
message(STATUS "CMAKE_BUILD_TYPE before default: '${CMAKE_BUILD_TYPE}'")
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()
message(STATUS "CMAKE_BUILD_TYPE after default: '${CMAKE_BUILD_TYPE}'")

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include HPC project options
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/HpcProjectOptions.cmake)

# Enforce O3 optimization for Release builds
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

# Global include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add source subdirectory
add_subdirectory(src)

# Testing (optional)
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
