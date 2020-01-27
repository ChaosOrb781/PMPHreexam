#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "OriginalAlgorithm.h"

using namespace thrust;
template <class T>
using hvec = host_vector<T>;
template <class T>
using dvec = device_vector<T>;

///numT iterations
__global__ void InitMyTimeline(
        const uint   numT,
        const REAL   t,
        REAL* myTimeline
    ) {
    /*
    for(unsigned i=0;i<numT;++i)
        myTimeline[i] = t*i/(numT-1);
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numT)
        myTimeline[gidx] = t*gidx/(numT-1);
}

///numX iterations
__global__ void InitMyX(
        const uint numX,
        const uint myXindex,
        const REAL s0,
        const REAL dx,
        REAL* myX
    ) {
    /*
    for(unsigned i=0;i<numX;++i)
        myX[i] = i*dx - myXindex*dx + s0;
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numX)
        myX[gidx] = gidx*dx - myXindex*dx + s0;
}

///numY iterations
__global__ void InitMyY(
        const uint numY,
        const uint myYindex,
        const REAL logAlpha,
        const REAL dy,
        REAL* myY
    ) {
    /*
    for(unsigned i=0;i<numY;++i)
        myY[i] = i*dy - myYindex*dy + logAlpha;
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numY)
        myY[gidx] = gidx*dy - myYindex*dy + logAlpha;
}

///numZ iterations
__global__ void InitMyDzz(
        const uint numZ,
        REAL* myZ,
        REAL* Dzz
    ) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numZ * 4) {
        uint row = gidx / numZ;
        uint col = gidx / 4;
        REAL dl, du;
        dl = (col == 0) ? 0.0 : myZ[col] - myZ[col - 1];
        du = (col == numZ - 1) ? 0.0 : myZ[col] - myZ[col - 1];;
        Dzz[gidx] = col > 0 && col < numZ-1 ?
                    (row == 0 ? 2.0/dl/(dl+du) :
                    (row == 1 ? -2.0*(1.0/dl + 1.0/du)/(dl+du) :
                    (row == 2 ? 2.0/du/(dl+du) :
                    0.0)))
                    : 0.0;
    }
}

__global__ void InitMyResult(
        const uint outer,
        const uint numX,
        const uint numY,
        REAL* myX,
        REAL* myResult
    ) {
    /*
    for(uint gidx = 0; gidx < outer * numX * numY; gidx++) {
        int o = gidx / (numX * numY);
        int plane_remain = gidx % (numX * numY);
        int i = plane_remain / numY;
        //int j = plane_remain % numY
        myResult[gidx] = std::max(myX[i]-0.001*(REAL)o, (REAL)0.0);
    }
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < outer * numX * numY) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        //int j = plane_remain % numY
        myResult[gidx] = std::max(myX[i]-0.001*(REAL)o, (REAL)0.0);
    }
}

__global__ void InitParams(
        const uint numT,
        const uint numX,
        const uint numY,
        const REAL alpha,
        const REAL beta,
        const REAL nu,
        REAL* myX,
        REAL* myY,
        REAL* myTimeline,
        REAL* myVarX,
        REAL* myVarY
    ) {
    /*
    for(uint gidx = 0; gidx < numT * numX * numY; gidx++) {
        int t = gidx / (numX * numY);
        int plane_remain = gidx % (numX * numY);
        int i = plane_remain / numY;
        int j = plane_remain % numY;
        myVarX[gidx] = exp(2.0*(  beta*log(myX[i])   
                                    + myY[j]             
                                    - 0.5*nu*nu*myTimeline[t] )
                                );
        myVarY[gidx] = exp(2.0*(  alpha*log(myX[i])   
                                    + myY[j]             
                                    - 0.5*nu*nu*myTimeline[t] )
                                ); // nu*nu
    }
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numT * numX * numY) {
        int t = gidx / (numX * numY);
        int plane_remain = gidx % (numX * numY);
        int i = plane_remain / numY;
        int j = plane_remain % numY;
        myVarX[gidx] = exp(2.0*(  beta*log(myX[i])   
                                    + myY[j]             
                                    - 0.5*nu*nu*myTimeline[t] )
                                );
        myVarY[gidx] = exp(2.0*(  alpha*log(myX[i])   
                                    + myY[j]             
                                    - 0.5*nu*nu*myTimeline[t] )
                                ); // nu*nu
    }
}

#endif