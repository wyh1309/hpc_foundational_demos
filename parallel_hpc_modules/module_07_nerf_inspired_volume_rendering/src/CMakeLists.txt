# Source-level CMakeLists for modular build

# ============================================================================
# Forward solver library (CPU + CUDA)
# ============================================================================
set(FORWARD_SOURCES forward/cpu_forward.cpp)
if(BUILD_CUDA_FORWARD)
    list(APPEND FORWARD_SOURCES forward/cuda_forward.cu)
endif()

add_library(forward_solver STATIC ${FORWARD_SOURCES})
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
)

if(BUILD_DEMOS)
    function(add_demo_if_ready target source library)
        file(SIZE "${CMAKE_CURRENT_SOURCE_DIR}/${source}" source_size)
        if(source_size GREATER 0)
            add_executable(${target} ${source})
            target_link_libraries(${target} PRIVATE ${library})
            set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        else()
            message(STATUS "Skipping empty demo source: ${source}")
        endif()
    endfunction()

    add_demo_if_ready(forward_demo demos/forward_demo.cu forward_solver)
    add_demo_if_ready(inverse_demo demos/inverse_demo.cu inverse_solver)
    add_demo_if_ready(benchmark demos/benchmark.cu forward_solver)

    foreach(demo_target forward_demo inverse_demo benchmark)
        if(TARGET ${demo_target})
            install(TARGETS ${demo_target} RUNTIME DESTINATION bin)
        endif()
    endforeach()
endif()
