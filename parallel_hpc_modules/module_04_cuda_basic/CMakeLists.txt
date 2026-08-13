cmake_minimum_required(VERSION 3.18)
project(Module04CudaBasic LANGUAGES CXX CUDA)

# Default to an optimized build for single-config generators (Ninja/Makefiles).
# An explicit -DCMAKE_BUILD_TYPE=Debug (or another type) always takes precedence.
message(STATUS "CMAKE_BUILD_TYPE before default: '${CMAKE_BUILD_TYPE}'")
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()
message(STATUS "CMAKE_BUILD_TYPE after default: '${CMAKE_BUILD_TYPE}'")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/HpcProjectOptions.cmake)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
    string(REGEX REPLACE "(^|[ \t])-O[0-3gs]($|[ \t])" "\\1-O3\\2"
        CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    if(NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "(^|[ \t])-O3($|[ \t])")
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
    endif()
endif()
if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA|Clang")
    string(REGEX REPLACE "(^|[ \t])-O[0-3gs]($|[ \t])" "\\1-O3\\2"
        CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
    if(NOT CMAKE_CUDA_FLAGS_RELEASE MATCHES "(^|[ \t])-O3($|[ \t])")
        string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -O3")
    endif()
endif()

add_subdirectory(src/device_info)
add_subdirectory(src/vector_add)
