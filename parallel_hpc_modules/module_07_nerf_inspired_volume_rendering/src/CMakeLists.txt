# Source-level CMakeLists for modular build

# ============================================================================
# Utility library (data generation, validation, plotting)
# ============================================================================
add_library(transport_utils STATIC
    utils/data_generator.cpp
    utils/validation.cpp
    utils/plotting.cpp
)
target_include_directories(transport_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ============================================================================
# Forward solver library (CPU + CUDA)
# ============================================================================
add_library(forward_solver STATIC
    forward/cpu_forward.cpp
    forward/cuda_forward.cu
)
target_include_directories(forward_solver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(forward_solver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Inverse solver library (constant parameters only)
# ============================================================================
add_library(inverse_solver STATIC
    inverse/grid_search.cpp
    inverse/newton_method.cpp
)
target_include_directories(inverse_solver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(inverse_solver PUBLIC
    forward_solver
    transport_utils
)

# ============================================================================
# Demo 1: Forward solver validation
# ============================================================================
add_executable(forward_demo
    demos/forward_demo.cu
)
target_link_libraries(forward_demo PRIVATE
    forward_solver
    transport_utils
)
set_target_properties(forward_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================================
# Demo 2: Inverse problem solver
# ============================================================================
add_executable(inverse_demo
    demos/inverse_demo.cu
)
target_link_libraries(inverse_demo PRIVATE
    inverse_solver
    forward_solver
    transport_utils
)
set_target_properties(inverse_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================================
# Demo 3: Performance benchmark
# ============================================================================
add_executable(benchmark
    demos/benchmark.cu
)
target_link_libraries(benchmark PRIVATE
    forward_solver
    transport_utils
)
set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================================
# Legacy executable (keep for compatibility)
# ============================================================================
add_executable(pde_volume_render
    main.cu
    cpu_reference.cpp
)
target_include_directories(pde_volume_render PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(pde_volume_render PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================================================
# Installation
# ============================================================================
install(TARGETS forward_demo inverse_demo benchmark pde_volume_render
    RUNTIME DESTINATION bin
)
