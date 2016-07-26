#ifndef GPUCOMMON_HPP
#define GPUCOMMON_HPP

#include "itf/trackers/buffers.h"
#include <vector>
#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>
#define gpu_zalloc(ptr, num, size) cudaMalloc(&ptr,size*num);cudaMemset(ptr,0,size*num);
#define  NUMTHREAD 1024
#define  MAXSTREAM 64 //Kepler Arch
__host__ __device__ __forceinline__ int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}


#endif // GPUCOMMON_HPP

