#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mpi.h>
#include <algorithm>

const int GLOBAL_N = 1024;
int local_rows;

void init_matrix(std::vector<double>& mat, int rows, int cols,
                 std::mt19937& gen,
                 std::uniform_real_distribution<double>& dist)
{
    mat.resize(rows * cols);
    for (int k = 0; k < rows * cols; k++) {
        mat[k] = dist(gen);
    }
}

void serial_block_matmul(
    const std::vector<double>& A_local,
    const std::vector<double>& B_global,
    std::vector<double>& C_local,
    int l_rows, int n
)
{
    for (int i = 0; i < l_rows; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                C_local[i * n + j] += A_local[i * n + k] * B_global[k * n + j];
            }
        }
    }
}

void print_part_matrix(const std::vector<double>& mat, int rows, int cols, int print_rows)
{
    int slice_len = print_rows * cols;
    auto start = mat.begin();
    auto end = mat.begin() + slice_len;

    std::cout << "matrix slice: [ ";
    std::for_each(start, end, [](double val) {
        std::cout << val << " ";
    });
    std::cout << "]" << std::endl;
}

int main(int argc, char** argv)
{
    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (GLOBAL_N % world_size != 0)
    {
        if (rank == 0)
        {
            std::cerr << "Matrix size cannot be divided evenly by number of processes" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    local_rows = GLOBAL_N / world_size;

    std::vector<double> A_global, B_global, C_global;
    std::mt19937 seed1, seed2;
    std::uniform_real_distribution<double> dist;
    std::chrono::steady_clock::time_point start_time;

    if (rank == 0)
    {
        A_global.resize(GLOBAL_N * GLOBAL_N);
        B_global.resize(GLOBAL_N * GLOBAL_N);
        C_global.resize(GLOBAL_N * GLOBAL_N, 0.0);

        init_matrix(A_global, GLOBAL_N, GLOBAL_N, seed1, dist);
        init_matrix(B_global, GLOBAL_N, GLOBAL_N, seed2, dist);

        start_time = std::chrono::steady_clock::now();
    }


    std::vector<double> B_recv(GLOBAL_N * GLOBAL_N); // Buffer for all processes to receive B matrix
    std::vector<double> A_local(local_rows * GLOBAL_N);
    std::vector<double> C_local(local_rows * GLOBAL_N, 0.0);

    MPI_Scatter(A_global.data(), local_rows * GLOBAL_N,
        MPI_DOUBLE, A_local.data(),
        local_rows * GLOBAL_N, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // Rank 0 uses original B_global; other ranks receive data into B_recv
    MPI_Bcast(rank == 0 ? B_global.data() : B_recv.data(),
        GLOBAL_N * GLOBAL_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Computation: rank0 uses original B_global, other ranks use received B_recv
    serial_block_matmul(A_local, rank == 0 ? B_global : B_recv, C_local, local_rows, GLOBAL_N);

    MPI_Gather(C_local.data(), local_rows * GLOBAL_N,
        MPI_DOUBLE, C_global.data(),
        local_rows * GLOBAL_N, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        auto end_time = std::chrono::steady_clock::now();
        double cost = std::chrono::duration<double>(end_time - start_time).count();

        std::cout << "time: " << cost << " s" << std::endl;
        print_part_matrix(C_global, GLOBAL_N, GLOBAL_N, 10);
    }

    MPI_Finalize();
    return 0;
}