add_executable(mpi_block_matmul main.cpp)
target_link_libraries(mpi_block_matmul PRIVATE MPI::MPI_CXX)
