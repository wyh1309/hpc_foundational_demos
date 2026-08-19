#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mpi.h>
#include <algorithm>
#include <string>
#include <cstdlib>

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

void print_part_matrix(const std::vector<double>& mat,int cols, int print_rows)
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
    if (argc < 3)
    {
        std::cout << "Usage:\n";
        std::cout << "  Strong scaling: ./mpi_matmul_scaling strong GLOBAL_N\n";
        std::cout << "  Strong scaling with baseline: ./mpi_matmul_scaling strong GLOBAL_N BASELINE_TIME\n";
        std::cout << "  Weak scaling: ./mpi_matmul_scaling weak LOCAL_ROWS_PER_PROC\n";
        std::cout << "Examples:\n";
        std::cout << "  mpirun -np 4 ./mpi_matmul_scaling strong 1024\n";
        std::cout << "  mpirun -np 4 ./mpi_matmul_scaling weak 256\n";
        return 1;
    }

    std::string mode = argv[1];
    int param = std::atoi(argv[2]);
    // Scaling runs are separate MPI jobs, so the single-process baseline must
    // be passed explicitly to jobs with more than one process.
    double baseline_time = -1.0;
    if (argc >= 4) {
        baseline_time = std::atof(argv[3]);
    }

    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int GLOBAL_N{0};
    int local_rows{0};

    if(mode == "strong"){
        // Strong scaling: keep the global matrix size fixed.
        GLOBAL_N = param;
        if (GLOBAL_N % world_size != 0)
        {
            if (rank == 0)
            {
                std::cerr << "[Strong scaling] GLOBAL_N must divisible by world_size\n";
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        local_rows = GLOBAL_N / world_size;
    }else if(mode == "weak"){
        // Weak scaling: keep the local row count per process fixed and
        // increase the global matrix size with the number of processes.
        int local_rows_per_proc = param;
        local_rows = local_rows_per_proc;
        GLOBAL_N = local_rows_per_proc * world_size;
    }else{
        if(rank==0){
            std::cerr<<"mode must be strong or weak\n";
        }
        MPI_Abort(MPI_COMM_WORLD,1);
    }


    std::vector<double> A_global, B_global, C_global;
    std::mt19937 seed1, seed2;
    std::uniform_real_distribution<double> dist;

    if (rank == 0)
    {
        A_global.resize(GLOBAL_N * GLOBAL_N);
        B_global.resize(GLOBAL_N * GLOBAL_N);
        C_global.resize(GLOBAL_N * GLOBAL_N, 0.0);

        init_matrix(A_global, GLOBAL_N, GLOBAL_N, seed1, dist);
        init_matrix(B_global, GLOBAL_N, GLOBAL_N, seed2, dist);
    }


    std::vector<double> B_recv(GLOBAL_N * GLOBAL_N);
    std::vector<double> A_local(local_rows * GLOBAL_N);
    std::vector<double> C_local(local_rows * GLOBAL_N, 0.0);

    // Synchronize all processes before starting the timer.
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    MPI_Scatter(A_global.data(), local_rows * GLOBAL_N,
        MPI_DOUBLE, A_local.data(),
        local_rows * GLOBAL_N, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    MPI_Bcast(rank == 0 ? B_global.data() : B_recv.data(),
        GLOBAL_N * GLOBAL_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    serial_block_matmul(A_local, rank == 0 ? B_global : B_recv, C_local, local_rows, GLOBAL_N);

    MPI_Gather(C_local.data(), local_rows * GLOBAL_N,
        MPI_DOUBLE, C_global.data(),
        local_rows * GLOBAL_N, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double wall_time = t_end - t_start;


    // Rank 0 outputs scaling statistics in CSV format.
    if(rank == 0)
    {
        double speedup = 0.0;
        double efficiency = 0.0;

        // Store the single-process baseline time.
        if(world_size == 1){
            baseline_time = wall_time;
        }

        if(mode == "strong")
        {
            if(baseline_time > 1e-12 && wall_time > 1e-12){
                speedup    = baseline_time / wall_time;
                efficiency = speedup / world_size;
            }else{
                speedup = NAN; efficiency = NAN;
            }
        }
        else if(mode == "weak")
        {
            // Speedup is not commonly defined for weak scaling; output wall time only.
            speedup = NAN;
            efficiency = NAN;
        }

        // Print the CSV header only once (for np=1).
        if(world_size == 1){
            std::cout<<"mode,world_size,global_N,local_rows,wall_time[s],speedup,efficiency\n";
        }
        std::cout<<mode<<","
                 <<world_size<<","
                 <<GLOBAL_N<<","
                 <<local_rows<<","
                 <<wall_time<<","
                 <<speedup<<","
                 <<efficiency<<"\n";
    }

    MPI_Finalize();
    return 0;
}
