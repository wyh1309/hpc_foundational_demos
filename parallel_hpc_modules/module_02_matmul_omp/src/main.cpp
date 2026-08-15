#include <iostream>
#include <chrono>
#include <algorithm>
#include <omp.h>

// Matrix dimension
#define N 1024
// Tile block size for cache optimization
#define BLOCK 32

// Serial matrix multiplication
void matmul_serial(double *A, double *B, double *C)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double tmp = 0.0;
            for (int k = 0; k < N; k++)
            {
                tmp += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = tmp;
        }
    }
}

// OpenMP parallel naive matrix multiplication
void matmul_omp(double *A, double *B, double *C)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double tmp = 0.0;
            for (int k = 0; k < N; k++)
            {
                tmp += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = tmp;
        }
    }
}

// OpenMP parallel matrix multiplication with tiling optimization
void matmul_omp_tiled(double *A, double *B, double *C)
{
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < N; ii += BLOCK)
    {
        for (int jj = 0; jj < N; jj += BLOCK)
        {
            for (int kk = 0; kk < N; kk += BLOCK)
            {
                for (int i = ii; i < std::min(ii + BLOCK, N); i++)
                {
                    for (int j = jj; j < std::min(jj + BLOCK, N); j++)
                    {
                        double tmp = 0.0;
                        for (int k = kk; k < std::min(kk + BLOCK, N); k++)
                        {
                            tmp += A[i*N + k] * B[k*N + j];
                        }
                        C[i*N + j] += tmp;
                    }
                }
            }
        }
    }
}

int main()
{
    // Allocate memory for matrix A, B, C
    double *A = new double[N*N];
    double *B = new double[N*N];
    double *C = new double[N*N];

    // Initialize matrix elements
    for(int i=0; i<N*N; i++) A[i] = 1.0;
    for(int i=0; i<N*N; i++) B[i] = 2.0;

    // ========== Serial benchmark ==========
    for(int i=0; i<N*N; i++) C[i] = 0.0;
    auto t1 = std::chrono::high_resolution_clock::now();
    matmul_serial(A, B, C);
    auto t2 = std::chrono::high_resolution_clock::now();
    double time_serial = std::chrono::duration<double>(t2 - t1).count();

    // ========== OpenMP naive parallel benchmark ==========
    omp_set_num_threads(4); // Set number of OpenMP threads
    for(int i=0; i<N*N; i++) C[i] = 0.0;
    auto t3 = std::chrono::high_resolution_clock::now();
    matmul_omp(A, B, C);
    auto t4 = std::chrono::high_resolution_clock::now();
    double time_omp = std::chrono::duration<double>(t4 - t3).count();

    // ========== OpenMP tiled parallel benchmark ==========
    for(int i=0; i<N*N; i++) C[i] = 0.0;
    auto t5 = std::chrono::high_resolution_clock::now();
    matmul_omp_tiled(A, B, C);
    auto t6 = std::chrono::high_resolution_clock::now();
    double time_omp_tiled = std::chrono::duration<double>(t6 - t5).count();

    // Print elapsed time and speedup metrics
    std::cout << "Serial time:      " << time_serial << " s\n";
    std::cout << "OMP naive time:   " << time_omp << " s\n";
    std::cout << "OMP tiled time:   " << time_omp_tiled << " s\n";
    std::cout << "Speedup (naive):  " << time_serial / time_omp << "\n";
    std::cout << "Speedup (tiled):  " << time_serial / time_omp_tiled << "\n";

    // Free dynamically allocated memory
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}