add_executable(mpi_basic main.cpp)
target_link_libraries(mpi_basic PRIVATE MPI::MPI_CXX)
