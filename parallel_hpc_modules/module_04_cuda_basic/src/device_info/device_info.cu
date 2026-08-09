/**
 * @file device_info.cu
 * @brief Day4: Query CUDA GPU hardware info
 */
#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int device_count = 0;

    // TODO: 1. 获取可用GPU数量
    cudaGetDeviceCount(&device_count);
    

    for (int dev = 0; dev < device_count; dev++)
    {
        cudaDeviceProp prop;
        // TODO: 2. 获取对应设备属性
        cudaGetDeviceProperties(&prop, dev);

        // TODO: 3. 打印关键参数：SM数量、warp大小、最大block尺寸、显存大小
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
