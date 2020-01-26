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
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x;
    myTimeline[gidx] = t*gidx/(numT-1);
}

///numX iterations
__global__ void InitMyX(
                const uint myXindex,
                const REAL s0,
                const REAL dx,
                REAL* myX
    ) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x;
    myX[gidx] = gidx*dx - myXindex*dx + s0;
}

///numY iterations
__global__ void InitMyY(
                const uint myYindex,
                const REAL logAlpha,
                const REAL dy,
                REAL* myY
    ) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x;
    myY[gidx] = gidx*dy - myYindex*dy + logAlpha;
}

///numZ iterations
__global__ void InitMyDzzT(
                const uint numZ,
                dvec<REAL> myZ,
                dvec<REAL> myDzzT
    ) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numZ * 4) {
        int row = gidx / numZ;
        int col = gidx / 4;
        REAL dl = col > 0 ? myZ[col] - myZ[col-1] : 0.0;
		REAL du = col < numZ-1 ? myZ[col+1] - myZ[col] : 0.0;

		myDzzT[gidx] = col > 0 && col < numZ-1 ?
                       (row == 0 ? 2.0/dl/(dl+du) :
                       (row == 1 ? -2.0*(1.0/dl + 1.0/du)/(dl+du) :
                       (row == 2 ? 2.0/du/(dl+du) :
                        0.0)))
                       : 0.0;
    }
}



__global__ void MyTimeline(const uint   outer,
                const uint   numX,
                const uint   numY,
                const uint   numT,
                const REAL   s0,
                const REAL   t, 
                const REAL   alpha, 
                const REAL   nu
    ) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x;

}

#endif