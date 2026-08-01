cmake_minimum_required(VERSION 3.20)

project(module_02_matmul_omp LANGUAGES CXX)

# Default build type to optimized Release for single-config generators (Make/Ninja).
# User specified -DCMAKE_BUILD_TYPE=Debug always overrides this setting.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# HPC optimization configuration for Ubuntu/GCC/Clang/Intel compilers.
# Force -O3 optimization level for Release builds, preserve other existing flags.
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

# Locate OpenMP parallel computing library
find_package(OpenMP REQUIRED)

# Helper function: attach OpenMP linkage to specified target
function(target_link_omp target_name)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
    endif()
endfunction()

add_subdirectory(src)