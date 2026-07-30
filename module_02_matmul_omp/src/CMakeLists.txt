add_executable(matmul_omp main.cpp)
target_link_libraries(matmul_omp PRIVATE OpenMP::OpenMP_CXX)
