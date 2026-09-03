# Test targets are added only after their source files contain an implementation.
# This keeps BUILD_TESTS=ON usable while the exercise files are still scaffolds.

function(add_test_if_ready target source)
    file(SIZE "${CMAKE_CURRENT_SOURCE_DIR}/${source}" source_size)
    if(source_size GREATER 0)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE forward_solver inverse_solver)
        target_include_directories(${target} PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/forward
        )
        add_test(NAME ${target} COMMAND ${target})
    else()
        message(STATUS "Skipping empty test source: ${source}")
    endif()
endfunction()

add_test_if_ready(test_forward test_forward.cpp)
add_test_if_ready(test_inverse test_inverse.cpp)
add_test_if_ready(test_convergence test_convergence.cpp)

if(BUILD_CUDA_FORWARD)
    add_executable(test_cuda_forward test_cuda_forward.cu)
    target_link_libraries(test_cuda_forward PRIVATE forward_solver)
    target_include_directories(test_cuda_forward PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/forward
    )
    set_target_properties(test_cuda_forward PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME test_cuda_forward COMMAND test_cuda_forward)
endif()
