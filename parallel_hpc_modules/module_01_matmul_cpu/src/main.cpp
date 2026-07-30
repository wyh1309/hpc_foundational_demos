#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

// Matrix multiplication naive version i-j-k loop order
void matmul_ijk(double *A, double *B, double *C, int N){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            for(int k=0;k<N;k++){
                C[i*N+j] += A[i*N+k]*B[k*N+j];
            }
        }
    }
}


// Matrix multiplication optimized version i-k-j loop order
void matmul_ikj(double *A, double *B, double *C, int N){
    for(int i=0;i<N;i++){
        for(int k=0;k<N;k++){
            for(int j=0;j<N;j++){
                C[i*N+j] += A[i*N+k]*B[k*N+j];
            }
        }
    }
}

int main(){
    const int N = 1024; 
    
    // 1. Allocate matrix A,B,C on heap
    double *A = new double[N*N];
    double *B = new double[N*N];
    double *C = new double[N*N];

    // 2. Initialize value of matrix A and B
    for(int i=0; i<N*N; i++) A[i] = 1.0;
    for(int i=0; i<N*N; i++) B[i] = 2.0;
    

    
    // 3. Benchmark ijk implementation, reset matrix C to zero first
    for(int i=0; i<N*N; i++) C[i] = 0.0;
    auto start1 = steady_clock::now();

    matmul_ijk(A,B,C,N);

    auto end1 = steady_clock::now();
    auto dur1 = duration_cast<microseconds>(end1 - start1);
    double t1 = dur1.count() / 1e6;
    cout << "ijk runtime: " << t1 << " s\n";
    
    // 4. Benchmark ikj implementation, reset matrix C to zero again
    for(int i=0; i<N*N; i++) C[i] = 0.0;
    auto start2 = steady_clock::now();

    matmul_ikj(A,B,C,N);

    auto end2 = steady_clock::now();
    auto dur2 = duration_cast<microseconds>(end2 - start2);
    double t2 = dur2.count() / 1e6;
    cout << "ikj runtime: " << t2 << " s\n";
    
    // 5. Free dynamically allocated memory
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}