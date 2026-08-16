#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <random>
#include <cstdio>
#include <cmath>
#include <cstring>

// Macro for CUDA runtime error checking, exit program on failure
#define CHECK_CUDA_FATAL(err) \
do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "[CUDA FATAL] %s:%d -> %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Macro for kernel launch error checking
#define CHECK_KERNEL() \
do { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "[KERNEL LAUNCH ERROR] %s:%d -> %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Hyperparameter: tile block size for shared memory optimization
#define TILE_WIDTH 32


/**
 * Tiled Matrix Multiplication Kernel with Shared Memory
 * Calculates matrix multiplication: C = A * B
 * Matrix A shape: M rows × N columns
 * Matrix B shape: N rows × P columns
 * Output Matrix C shape: M rows × P columns
 */
__global__ void matmul_tiled_kernel(
    float* d_a,
    float* d_b,
    float* d_c,
    int M, int N, int P
)
{
    // Step 1: Get local thread index inside current block
    int tx = threadIdx.x;
    int ty = threadIdx.y;


    // Step 2: Calculate global row & column index for output matrix C
   
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // Step 3: Allocate block-private shared memory buffer for A tile and B tile
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];


    // Step 4: Register variable to accumulate dot product result
    float accumulate = 0.0f;


    // Step 5: Calculate total tile count along reduction dimension N
    int total_tiles = (N+TILE_WIDTH-1)/TILE_WIDTH;


    // Outer loop: iterate every tile segment along N dimension
    for (int tile_idx = 0; tile_idx < total_tiles; tile_idx++)
    {

        // ---------------- Load A tile from global memory to shared memory As ----------------
        int a_global_col = tile_idx*TILE_WIDTH+tx;

        // Boundary judgment: only load valid elements, fill zero for out-of-bound position
        if (row<M && a_global_col<N)  // TO-DO: Boundary check for matrix A
        {
            int a_global_addr = row*N+a_global_col;
            As[ty][tx] = d_a[a_global_addr];
        }
        else
        {
            As[ty][tx] = 0.0f;
        }



        // ---------------- Load B tile from global memory to shared memory Bs ----------------
        int b_global_row = tile_idx*TILE_WIDTH+ty;

        if (b_global_row<N && col<P)  // TO-DO: Boundary check for matrix B
        {
            int b_global_addr = b_global_row*P+col;
            Bs[ty][tx] = d_b[b_global_addr];
        }
        else
        {
            Bs[ty][tx] = 0.0f;
        }



        // Step 6: Block thread synchronization
        // Wait all threads finish loading current tile to shared memory
        __syncthreads();



        // Step 7: Compute dot product inside shared memory tile
        
        for (int k=0;k<TILE_WIDTH;k++)
        {
            accumulate += As[ty][k]*Bs[k][tx];
        }



        // Step 8: Synchronize again before loading next tile block
        __syncthreads();

    } // End of tile outer loop


    // Step 9: Write final accumulated value back to global output matrix C
    if (row < M && col < P)
    {
        int c_global_addr = row*P+col;
        d_c[c_global_addr] = accumulate;
    }

} // End of tiled kernel



/**
 * Naive Global Memory Matrix Multiplication Kernel (baseline comparison)
 * Original unoptimized version without shared memory
 */
__global__ void matmul_naive_kernel(
    float* d_a,
    float* d_b,
    float* d_c,
    int M,int N,int P
)
{
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    int jdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < M && jdx < P)
    {   
        float tmp = 0;
        for(int k = 0;k < N;k++){
            tmp += d_a[idx*N+k]*d_b[k*P+jdx];
        }
        d_c[idx*P+jdx] = tmp;
    }
}


/**
 * CPU Serial Reference Implementation
 * Generate ground truth result for error verification
 */
void cpu_verify(float* h_a, float* h_b, float* h_ref, int M,int N,int P)
{
    for(int i=0;i<M;i++){
        for(int j=0;j<P;j++){
            for(int k=0;k<N;k++){
                h_ref[i*P+j] += h_a[i*N+k]*h_b[k*P+j];
            }
        }
    }
}



int main()
{
    // Modify matrix dimensions here
    // int M = 997;
    // int N = 997;
    // int P = 997;
    int M = 1024;
    int N = 1024;
    int P = 1024;

    // Random float value generator for matrix initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);


    // ===== Step 1: Allocate & Initialize Host Memory =====
    float *h_a, *h_b, *h_c_naive, *h_c_tiled, *h_ref_cpu;
    h_a         = (float *)malloc(M * N * sizeof(float));
    h_b         = (float *)malloc(N * P * sizeof(float));
    h_c_naive   = (float *)malloc(M * P * sizeof(float));
    h_c_tiled   = (float *)malloc(M * P * sizeof(float));
    h_ref_cpu   = (float*)malloc(M * P * sizeof(float));

    memset(h_ref_cpu, 0, M*P*sizeof(float));

    // Fill matrix A with random floats
    for (int i = 0; i < M*N; i++)
    {
        h_a[i] = dist(gen);
    }
    // Fill matrix B with random floats
    for (int i = 0; i < N*P; i++)
    {
        h_b[i] = dist(gen);
    }


    // ===== Step 2: Allocate Device Memory on GPU =====
    float *d_a = nullptr, *d_b = nullptr, *d_c_naive = nullptr, *d_c_tiled = nullptr;
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_a, M * N * sizeof(float)));
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_b, N * P * sizeof(float)));
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_c_naive, M * P * sizeof(float)));
    CHECK_CUDA_FATAL(cudaMalloc((void**)&d_c_tiled, M * P * sizeof(float)));

    // Copy input matrices from host to device global memory
    CHECK_CUDA_FATAL(cudaMemcpy(d_a, h_a, M*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_FATAL(cudaMemcpy(d_b, h_b, N*P*sizeof(float), cudaMemcpyHostToDevice));


    // ===== Step 3: Create CUDA Event Timers for GPU Runtime Measurement =====
    cudaEvent_t start_naive_k, stop_naive_k;
    cudaEvent_t start_tiled_k, stop_tiled_k;
    cudaEvent_t start_naive_e2e, stop_naive_e2e;
    cudaEvent_t start_tiled_e2e, stop_tiled_e2e;

    CHECK_CUDA_FATAL(cudaEventCreate(&start_naive_k));
    CHECK_CUDA_FATAL(cudaEventCreate(&stop_naive_k));
    CHECK_CUDA_FATAL(cudaEventCreate(&start_tiled_k));
    CHECK_CUDA_FATAL(cudaEventCreate(&stop_tiled_k));
    CHECK_CUDA_FATAL(cudaEventCreate(&start_naive_e2e));
    CHECK_CUDA_FATAL(cudaEventCreate(&stop_naive_e2e));
    CHECK_CUDA_FATAL(cudaEventCreate(&start_tiled_e2e));
    CHECK_CUDA_FATAL(cudaEventCreate(&stop_tiled_e2e));


    // Kernel execution configuration for naive matmul
    const dim3 block_naive(32,32);
    const dim3 grid_naive(
        (P + block_naive.x - 1) / block_naive.x,
        (M + block_naive.y - 1) / block_naive.y
    );

    // Kernel execution configuration for tiled shared memory matmul
    const dim3 block_tiled(TILE_WIDTH, TILE_WIDTH);
    const dim3 grid_tiled(
        (P + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );

    float naive_kernel_ms = 0.0f;
    float naive_e2e_ms = 0.0f;
    float tiled_kernel_ms = 0.0f;
    float tiled_e2e_ms = 0.0f;


    // ---------------------- Naive GPU: End‑to‑end (H2D already done, kernel + D2H) ----------------------
    CHECK_CUDA_FATAL(cudaEventRecord(start_naive_e2e));

    CHECK_CUDA_FATAL(cudaEventRecord(start_naive_k));
    matmul_naive_kernel<<<grid_naive, block_naive>>>(d_a, d_b, d_c_naive, M,N,P);
    CHECK_KERNEL();
    CHECK_CUDA_FATAL(cudaDeviceSynchronize());
    CHECK_CUDA_FATAL(cudaEventRecord(stop_naive_k));
    CHECK_CUDA_FATAL(cudaEventSynchronize(stop_naive_k));
    CHECK_CUDA_FATAL(cudaEventElapsedTime(&naive_kernel_ms, start_naive_k, stop_naive_k));

    CHECK_CUDA_FATAL(cudaMemcpy(h_c_naive, d_c_naive, M*P*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_FATAL(cudaEventRecord(stop_naive_e2e));
    CHECK_CUDA_FATAL(cudaEventSynchronize(stop_naive_e2e));
    CHECK_CUDA_FATAL(cudaEventElapsedTime(&naive_e2e_ms, start_naive_e2e, stop_naive_e2e));


    // ---------------------- Tiled GPU: End‑to‑end (kernel + D2H) ----------------------
    CHECK_CUDA_FATAL(cudaEventRecord(start_tiled_e2e));

    CHECK_CUDA_FATAL(cudaEventRecord(start_tiled_k));
    matmul_tiled_kernel<<<grid_tiled, block_tiled>>>(d_a, d_b, d_c_tiled, M,N,P);
    CHECK_KERNEL();
    CHECK_CUDA_FATAL(cudaDeviceSynchronize());
    CHECK_CUDA_FATAL(cudaEventRecord(stop_tiled_k));
    CHECK_CUDA_FATAL(cudaEventSynchronize(stop_tiled_k));
    CHECK_CUDA_FATAL(cudaEventElapsedTime(&tiled_kernel_ms, start_tiled_k, stop_tiled_k));

    CHECK_CUDA_FATAL(cudaMemcpy(h_c_tiled, d_c_tiled, M*P*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_FATAL(cudaEventRecord(stop_tiled_e2e));
    CHECK_CUDA_FATAL(cudaEventSynchronize(stop_tiled_e2e));
    CHECK_CUDA_FATAL(cudaEventElapsedTime(&tiled_e2e_ms, start_tiled_e2e, stop_tiled_e2e));


    // ===== Step 4: Run CPU Serial Calculation & Timing =====
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_verify(h_a, h_b, h_ref_cpu, M,N,P);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;



    // ===== Step 5: Result Error Verification =====
    bool naive_pass = true;
    bool tiled_pass = true;
    const float eps = 1e-3f;

    // Check naive kernel output
    for (int i = 0; i < M*P; i++)
    {
        if (fabs(h_c_naive[i] - h_ref_cpu[i]) > eps)
        {
            naive_pass = false;
            std::cerr << "[Naive Kernel] Mismatch index " << i
                      << " GPU:" << h_c_naive[i]
                      << " CPU:" << h_ref_cpu[i] << "\n";
            break;
        }
    }

    // Check tiled kernel output
    for (int i = 0; i < M*P; i++)
    {
        if (fabs(h_c_tiled[i] - h_ref_cpu[i]) > eps)
        {
            tiled_pass = false;
            std::cerr << "[Tiled Kernel] Mismatch index " << i
                      << " GPU:" << h_c_tiled[i]
                      << " CPU:" << h_ref_cpu[i] << "\n";
            break;
        }
    }



    // ===== Step 6: Print Performance Report =====
    std::cout << "=============================================\n";
    std::cout << "Matrix Size: M=" << M << " N=" << N << " P=" << P << "\n";
    std::cout << "Tile Width:  " << TILE_WIDTH << "\n";
    std::cout << "---------------------------------------------\n";
    std::cout << "CPU Serial Time:            " << cpu_ms << " ms\n";
    std::cout << "---------------------------------------------\n";
    std::cout << "[GPU Kernel‑Only Time (pure compute)]\n";
    std::cout << "Naive GPU Kernel:           " << naive_kernel_ms << " ms | Valid: " << (naive_pass ? "YES" : "NO") << "\n";
    std::cout << "Tiled Shared GPU Kernel:    " << tiled_kernel_ms << " ms | Valid: " << (tiled_pass ? "YES" : "NO") << "\n";
    std::cout << "---------------------------------------------\n";
    std::cout << "[GPU End‑to‑End Time (kernel + DtoH copy)]\n";
    std::cout << "Naive GPU E2E:              " << naive_e2e_ms << " ms\n";
    std::cout << "Tiled Shared GPU E2E:       " << tiled_e2e_ms << " ms\n";
    std::cout << "---------------------------------------------\n";
    std::cout << "Speedup CPU / Naive(Kernel):    " << cpu_ms / naive_kernel_ms << " x\n";
    std::cout << "Speedup CPU / Tiled(Kernel):    " << cpu_ms / tiled_kernel_ms << " x\n";
    std::cout << "Speedup Tiled / Naive(Kernel):  " << naive_kernel_ms / tiled_kernel_ms << " x\n";
    std::cout << "---------------------------------------------\n";
    std::cout << "Speedup CPU / Naive(E2E):       " << cpu_ms / naive_e2e_ms << " x\n";
    std::cout << "Speedup CPU / Tiled(E2E):       " << cpu_ms / tiled_e2e_ms << " x\n";
    std::cout << "=============================================\n";



    // ===== Step 7: Release All Allocated Resources =====
    // Destroy CUDA timer events
    CHECK_CUDA_FATAL(cudaEventDestroy(start_naive_k));
    CHECK_CUDA_FATAL(cudaEventDestroy(stop_naive_k));
    CHECK_CUDA_FATAL(cudaEventDestroy(start_tiled_k));
    CHECK_CUDA_FATAL(cudaEventDestroy(stop_tiled_k));
    CHECK_CUDA_FATAL(cudaEventDestroy(start_naive_e2e));
    CHECK_CUDA_FATAL(cudaEventDestroy(stop_naive_e2e));
    CHECK_CUDA_FATAL(cudaEventDestroy(start_tiled_e2e));
    CHECK_CUDA_FATAL(cudaEventDestroy(stop_tiled_e2e));

    // Free GPU device memory
    CHECK_CUDA_FATAL(cudaFree(d_a));
    CHECK_CUDA_FATAL(cudaFree(d_b));
    CHECK_CUDA_FATAL(cudaFree(d_c_naive));
    CHECK_CUDA_FATAL(cudaFree(d_c_tiled));

    // Free CPU host memory
    free(h_a);
    free(h_b);
    free(h_c_naive);
    free(h_c_tiled);
    free(h_ref_cpu);

    return 0;
}