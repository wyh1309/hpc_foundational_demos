set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HPC_ENABLE_PROFILING "Enable profiler-friendly compiler options" OFF)

if(HPC_ENABLE_PROFILING)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-g>)
endif()
