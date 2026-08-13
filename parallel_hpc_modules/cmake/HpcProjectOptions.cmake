# Common compiler diagnostics for the project's own C++ and CUDA targets.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang|Intel")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>"
    )
endif()

# nvcc compiles CUDA sources with a host C++ compiler. Forward the warning
# options so both device translation and host code are checked consistently.
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
    )
elseif(CMAKE_CUDA_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
    )
endif()
