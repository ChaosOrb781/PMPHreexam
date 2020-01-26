#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "OriginalAlgorithm.h"

using namespace thrust;

__global__ void InitGridTest(const uint   outer,
                const uint   numX,
                const uint   numY,
                const uint   numT,
                const REAL   s0,
                const REAL   t, 
                const REAL   alpha, 
                const REAL   nu,
    ) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x;

    for(unsigned i=0;i<numT;++i)
        myTimeline[gidx] = t*i/(numT-1);

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    uint myXindex = static_cast<unsigned>(s0/dx) % numX;

    for(unsigned i=0;i<numX;++i)
        myX[i] = i*dx - myXindex*dx + s0;

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    uint myYindex = static_cast<unsigned>(numY/2.0);

    for(unsigned i=0;i<numY;++i)
        myY[i] = i*dy - myYindex*dy + logAlpha;
}

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