#include <mpi.h>
#include <iostream>
#include <chrono>
#include <climits>

int main(int argc, char** argv) {
    // Step1: Initialize MPI environment
    MPI_Init(&argc, &argv);

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int local_val = world_rank * 10;
    

    auto start = std::chrono::high_resolution_clock::now();

    // Step2: Collective communication: Reduce all local value to rank 0
    int global_max = 0;
    MPI_Reduce(
        &local_val,        // send buffer: local data of current rank
        &global_max,       // recv buffer: only valid on root rank
        1,                  // element count
        MPI_INT,            // data type
        MPI_MAX,            // reduction operation
        0,                  // root process rank
        MPI_COMM_WORLD
    );

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // Step3: Root process print final result
    if (world_rank == 0) {
        std::cout << "Total process count: " << world_size << std::endl;
        std::cout << "Global max result: " << global_max << std::endl;
        std::cout << "MPI reduce time cost: " << elapsed << " s" << std::endl;
    }

    // Step4: Finalize MPI
    MPI_Finalize();
    return 0;
}
