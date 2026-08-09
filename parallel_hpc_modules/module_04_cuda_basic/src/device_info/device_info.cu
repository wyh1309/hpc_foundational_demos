/**
 * @file device_info.cu
 * @brief Day4: Query CUDA GPU hardware info
 */
#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int device_count = 0;

    // TODO: 1. Get the number of available GPUs
    cudaGetDeviceCount(&device_count);
    

    for (int dev = 0; dev < device_count; dev++)
    {
        cudaDeviceProp prop;
        // TODO: 2. Get the properties of the corresponding device
        cudaGetDeviceProperties(&prop, dev);

        // TODO: 3. Print key parameters: SM count, warp size, maximum block size, and memory size
        std::cout << "==== GPU Device " << dev << " ====" << std::endl;
        std::cout << "==== GPU Name " << prop.name << " ====" << std::endl;
        std::cout << "==== SM Number " << prop.multiProcessorCount << " ====" << std::endl;
        std::cout << "==== Wrap Size " << prop.warpSize << " ====" << std::endl;
        std::cout << "==== Max Block Size " << prop.maxThreadsPerBlock << " ====" << std::endl;
        std::cout << "==== Global Memory Size " << prop.totalGlobalMem << " ====" << std::endl;
        // print prop.name, prop.multiProcessorCount, prop.warpSize ...
    }

    return 0;
}
