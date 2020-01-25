#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "OriginalAlgorithm.h"

__global__ void MyTimeline(const uint   outer,
                const uint   numX,
                const uint   numY,
                const uint   numT,
                const REAL   s0,
                const REAL   t, 
                const REAL   alpha, 
                const REAL   nu,
    ) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x;
    //Memory dedication
    //globstastic[i].Initialize(numX, numY, numT);
    initGrid(s0,alpha,nu,t, numX, numY, numT, globstastic[i]);
    initOperator(globstastic[i].myX,globstastic[i].myDxx);
    initOperator(globstastic[i].myY,globstastic[i].myDyy);
    REAL strike = 0.001*i;
    setPayoff(strike, globstastic[i]);
}

#endif