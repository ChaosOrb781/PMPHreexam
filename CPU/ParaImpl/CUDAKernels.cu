#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include <cuda_runtime.h>
#include "Constants.h"
#include "TridagKernel.cu.h"

///numT iterations, Coalesced
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

///numX iterations, Coalesced
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

///numY iterations, Coalesced
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

///numZ * 4 iterations
__global__ void InitMyDzz(
        const uint numZ,
        REAL* myZ,
        REAL* Dzz
    ) {
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numZ * 4) {
        uint row = gidx % 4;
        uint col = gidx / 4;
        REAL dl, du;
        __syncthreads();
        dl = (col == 0) ? 0.0 : myZ[col] - myZ[col - 1];
        du = (col == numZ - 1) ? 0.0 : myZ[col + 1] - myZ[col];
        __syncthreads();
        Dzz[gidx] = col > 0 && col < numZ-1 ?
                    (row == 0 ? 2.0/dl/(dl+du) :
                    (row == 1 ? -2.0*(1.0/dl + 1.0/du)/(dl+du) :
                    (row == 2 ? 2.0/du/(dl+du) :
                    0.0)))
                    : 0.0;
    }
}

///numZ coalesced, but transposed
__global__ void InitMyDzzTCoalesced(
        const uint numZ,
        REAL* myZ,
        REAL* DzzT
    ) {
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numZ) {
        //Coalesced
        REAL low = gidx > 0 ? myZ[gidx - 1] : 0.0;
        __syncthreads();
        //Coalesced
        REAL mid = myZ[gidx];
        __syncthreads();
        //Coalesced
        REAL high = gidx < numZ - 1 ? myZ[gidx + 1] : 0.0;

        REAL dl = mid - low;
        REAL du = high - mid;


        __syncthreads();
        DzzT[0 * numZ + gidx] = gidx > 0 && gidx < numZ - 1 ? 2.0 / dl / (dl + du) : 0.0;
        __syncthreads();
        DzzT[0 * numZ + gidx] = gidx > 0 && gidx < numZ - 1 ? -2.0 / (1.0 / dl + 1.0 / du) / (dl + du) : 0.0;
        __syncthreads();
        DzzT[0 * numZ + gidx] = gidx > 0 && gidx < numZ - 1 ? 2.0 / du / (dl + du) : 0.0;
        __syncthreads();
        DzzT[0 * numZ + gidx] = 0.0;
    }
}

//outer * numX * numY iterations
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
        __syncthreads();
        REAL a = myX[i]-0.001*(REAL)o;
        myResult[gidx] = a > 0.0 ? a : (REAL)0.0;
    }
}

//numX iterations
__global__ void InitMyResultTCoalesced(
        const uint outer,
        const uint numX,
        const uint numY,
        REAL* myX,
        REAL* myResultT //[outer][numY][numX]
    ) {
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numX) {
        REAL x = myX[gidx];
        for (int o = 0; o < outer; o++) {
            for (int j = 0; j < numY; j++) {
                REAL payoff = x-0.001*(REAL)o;
                __syncthreads();
                myResultT[((o * numY) + j) * numX + gidx] = payoff > 0.0 ? payoff : (REAL)0.0;
            }
        }
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
        __syncthreads();
        myVarY[gidx] = exp(2.0*(  alpha*log(myX[i])   
                                    + myY[j]             
                                    - 0.5*nu*nu*myTimeline[t] )
                                ); // nu*nu
    }
}


__global__ void InitParamsVarXTCoalesced(
        const uint numT,
        const uint numX,
        const uint numY,
        const REAL alpha,
        const REAL beta,
        const REAL nu,
        REAL* myX,
        REAL* myY,
        REAL* myTimeline,
        REAL* myVarXT
    ) {
    //size: sizeof(REAL) * numX * numT + sizeof(REAL) [buffer for staying in bounds]
    extern __shared__ char sh_mem[];
    volatile REAL* myT_sh = (volatile REAL*) sh_mem;
    volatile REAL* myY_sh = (volatile REAL*) (myT_sh + numT);


    uint tidx = threadIdx.x;
    uint gidx = blockIdx.x*blockDim.x + tidx;

    __syncthreads();
    //load in myT, not good if gab between blocksize and numY is large
    for (int i = 0; i < numT; i += blockDim.x) {
        if (i + tidx < numT) {
            myT_sh[i + tidx] = myTimeline[i + tidx];
        }
    }
    __syncthreads();

    //load in myX
    for (int i = 0; i < numY; i += blockDim.x) {
        if (i + tidx < numY) {
            myY_sh[i + tidx] = myY[i + tidx];
        }
    }
    __syncthreads();

    if (gidx < numX) {
        for (int t = 0; t < numT; t++) {
            for (int j = 0; j < numY; j++) { 
                __syncthreads();
                REAL val1 = beta*log(myX[gidx]);
                REAL val2 = myY_sh[j];
                REAL val3 = - 0.5*nu*nu*myT_sh[t];

                __syncthreads();
                myVarXT[((t * numY) + j) * numX + gidx] = exp(2.0*(val1 + val2 + val3));
            }
        }
    }
}

__global__ void InitParamsVarYCoalesced(
        const uint numT,
        const uint numX,
        const uint numY,
        const REAL alpha,
        const REAL beta,
        const REAL nu,
        REAL* myX,
        REAL* myY,
        REAL* myTimeline,
        REAL* myVarY
    ) {
    //size: sizeof(REAL) * numX * numT + sizeof(REAL) [buffer for staying in bounds]
    extern __shared__ char sh_mem[];
    volatile REAL* myT_sh = (volatile REAL*) sh_mem;
    volatile REAL* myX_sh = (volatile REAL*) (myT_sh + numT);


    uint tidx = threadIdx.x;
    uint gidx = blockIdx.x*blockDim.x + tidx;

    __syncthreads();
    //load in myT, not good if gab between blocksize and numY is large
    for (int i = 0; i < numT; i += blockDim.x) {
        if (i + tidx < numT) {
            myT_sh[i + tidx] = myTimeline[i + tidx];
        }
    }
    __syncthreads();

    //load in myX
    for (int i = 0; i < numX; i += blockDim.x) {
        if (i + tidx < numX) {
            myX_sh[i + tidx] = myX[i + tidx];
        }
    }
    __syncthreads();

    if (gidx < numY) {
        for (int t = 0; t < numT; t++) {
            for (int i = 0; i < numX; i++) { 
                REAL val1 = alpha*log(myX_sh[i]);
                __syncthreads();
                REAL val2 = myY[gidx];
                REAL val3 = - 0.5*nu*nu*myT_sh[t];

                __syncthreads();
                myVarY[((t * numX) + i) * numY + gidx] = exp(2.0*(val1 + val2 + val3));
            }
        }
    }
}

__global__ void Rollback_1 (
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    REAL* myTimeline, 
    REAL* myDxx,
    REAL* myVarX,
    REAL* u,
    REAL* myResult
){
    /*
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        u[((o * numY) + j) * numX + i] = dtInv*myResult[((o * numX) + i) * numY + j];

        if(i > 0) { 
            u[((o * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                            * myDxx[i * 4 + 0] ) 
                            * myResult[((o * numX) + (i-1)) * numY + j];
        }
        u[((o * numY) + j) * numX + i]  +=  0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                        * myDxx[i * 4 + 1] )
                        * myResult[((o * numX) + i) * numY + j];
        if(i < numX-1) {
            u[((o * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                            * myDxx[i * 4 + 2] )
                            * myResult[((o * numX) + (i+1)) * numY + j];
        }
    }
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < outer * numX * numY) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        u[((o * numY) + j) * numX + i] = dtInv*myResult[((o * numX) + i) * numY + j];

        if(i > 0) { 
            u[((o * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                            * myDxx[i * 4 + 0] ) 
                            * myResult[((o * numX) + (i-1)) * numY + j];
        }
        u[((o * numY) + j) * numX + i]  +=  0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                        * myDxx[i * 4 + 1] )
                        * myResult[((o * numX) + i) * numY + j];
        if(i < numX-1) {
            u[((o * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                            * myDxx[i * 4 + 2] )
                            * myResult[((o * numX) + (i+1)) * numY + j];
        }
    }
}


__global__ void Rollback_1Coalesced (
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    REAL* myTimeline, 
    REAL* myDxxT,
    REAL* myVarXT,
    REAL* u,
    REAL* myResultT
){
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

    if (gidx < numX) {

        uint numZ = max(numX,numY);
        /*
        If threads in the same warp access (read) the same location in the same instruction/clock cycle, 
        the memory subsystem will use a "broadcast" mechanism so that only one read of that location is 
        required, and all threads using that data will receive it in the same transaction. This will not 
        result in additional transactions to service the request from multiple threads in this scenario.
        https://devtalk.nvidia.com/default/topic/1043273/cuda-programming-and-performance/accessing-same-global-memory-address-within-warps/
        */
        __syncthreads();
        //Just 2 reads, should not require shared memory to spare one memory 1 cycle
        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);


        for (int o = 0; o < outer; o++) {
            for (int j = 0; j < numY; j++) {
                __syncthreads();
                REAL myDxxT0 = myDxxT[0 * numX + gidx];
                REAL myDxxT1 = myDxxT[1 * numX + gidx];
                REAL myDxxT2 = myDxxT[2 * numX + gidx];

                __syncthreads();
                REAL myResultT_low  = myResultT[((o * numY) + j) * numX + gidx - 1];
                REAL myResultT_mid  = myResultT[((o * numY) + j) * numX + gidx];
                REAL myResultT_high = myResultT[((o * numY) + j) * numX + gidx + 1];

                __syncthreads();
                REAL myVarXT_val = 0.5 * myVarXT[((t * numY) + j) * numX + gidx];

                __syncthreads();
                u[((o * numY) + j) * numX + gidx] = dtInv * myResultT_mid;

                __syncthreads();
                if(gidx > 0) { 
                    u[((o * numY) + j) * numX + gidx] += 
                        0.5*( myVarXT_val * myDxxT0 ) * myResultT_low;
                }

                __syncthreads();
                u[((o * numY) + j) * numX + gidx]  +=  
                    0.5*( myVarXT_val * myDxxT1 ) * myResultT_mid;

                __syncthreads();
                if(gidx < numX-1) {
                    u[((o * numY) + j) * numX + gidx] += 
                        0.5*( myVarXT_val * myDxxT2 ) * myResultT_high;
                }
            }
        }
    }
}

__global__ void Rollback_2 (
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    REAL* myTimeline,
    REAL* myDyy,
    REAL* myVarY,
    REAL* u,
    REAL* v,
    REAL* myResult
){
    //cout << "test 2" << endl;
    /*
    for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
        uint o = gidx / (numY * numX);
        uint plane_remain = gidx % (numY * numX);
        uint j = plane_remain / numX;
        uint i = plane_remain % numX;
        uint numZ = max(numX,numY);
        v[((o * numX) + i) * numY + j] = 0.0;

        if(j > 0) {
            v[((o * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                            * myDyy[j * 4 + 0] )
                            * myResult[((o * numX) + i) * numY + j - 1];
        }
        v[((o * numX) + i) * numY + j]  += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                            * myDyy[j * 4 + 1] )
                            * myResult[((o * numX) + i) * numY + j];
        if(j < numY-1) {
            v[((o * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                            * myDyy[j * 4 + 2] )
                            * myResult[((o * numX) + i) * numY + j + 1];
        }
        u[((o * numY) + j) * numX + i] += v[((o * numX) + i) * numY + j];
    }
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

    if (gidx < outer * numX * numY) {
        uint o = gidx / (numY * numX);
        uint plane_remain = gidx % (numY * numX);
        uint j = plane_remain / numX;
        uint i = plane_remain % numX;
        uint numZ = max(numX,numY);
        v[((o * numX) + i) * numY + j] = 0.0;

        if(j > 0) {
            v[((o * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                            * myDyy[j * 4 + 0] )
                            * myResult[((o * numX) + i) * numY + j - 1];
        }
        v[((o * numX) + i) * numY + j]  += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                            * myDyy[j * 4 + 1] )
                            * myResult[((o * numX) + i) * numY + j];
        if(j < numY-1) {
            v[((o * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                            * myDyy[j * 4 + 2] )
                            * myResult[((o * numX) + i) * numY + j + 1];
        }
        u[((o * numY) + j) * numX + i] += v[((o * numX) + i) * numY + j];
    }
}

__global__ void Rollback_2Coalesced1 (
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    REAL* myTimeline,
    REAL* myDyyT,
    REAL* myVarY,
    REAL* v,
    REAL* myResult
){
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

    if (gidx < numY) {
        /*
        If threads in the same warp access (read) the same location in the same instruction/clock cycle, 
        the memory subsystem will use a "broadcast" mechanism so that only one read of that location is 
        required, and all threads using that data will receive it in the same transaction. This will not 
        result in additional transactions to service the request from multiple threads in this scenario.
        https://devtalk.nvidia.com/default/topic/1043273/cuda-programming-and-performance/accessing-same-global-memory-address-within-warps/
        */
        __syncthreads();
        //Just 2 reads, should not require shared memory to spare one memory 1 cycle
        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);


        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                __syncthreads();
                REAL myDyyT0 = myDyyT[0 * numY + gidx];
                REAL myDyyT1 = myDyyT[1 * numY + gidx];
                REAL myDyyT2 = myDyyT[2 * numY + gidx];

                __syncthreads();
                REAL myResult_low  = myResult[((o * numX) + i) * numY + gidx - 1];
                REAL myResult_mid  = myResult[((o * numX) + i) * numY + gidx];
                REAL myResult_high = myResult[((o * numX) + i) * numY + gidx + 1];

                __syncthreads();
                REAL myVarY_val = 0.5 * myVarY[((t * numX) + i) * numY + gidx];

                __syncthreads();
                v[((o * numX) + i) * numY + gidx] = dtInv * myResult_mid;

                __syncthreads();
                if(gidx > 0) { 
                    v[((o * numX) + i) * numY + gidx] += 
                        0.5*( myVarY_val * myDyyT0 ) * myResult_low;
                }

                __syncthreads();
                v[((o * numX) + i) * numY + gidx]  +=  
                    0.5*( myVarY_val * myDyyT1 ) * myResult_mid;

                __syncthreads();
                if(gidx < numY-1) {
                    v[((o * numX) + i) * numY + gidx] += 
                        0.5*( myVarY_val * myDyyT2 ) * myResult_high;
                }
            }
        }
    }
}

__global__ void Rollback_2Coalesced2 (
    int t,
    const uint outer,
    const uint numX, 
    const uint numY,
    REAL* uT,
    REAL* v,
    REAL* myResult
){
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

    if (gidx < numY) {
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                __syncthreads();
                uT[((o * numX) + i) * numY + gidx] += v[((o * numX) + i) * numY + gidx];
            }
        }
    }
}


__global__ void Rollback_3 (
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    REAL* myTimeline, 
    REAL* myDxx,
    REAL* myVarX,
    REAL* a,
    REAL* b,
    REAL* c
){
    //cout << "test 3" << endl;
    /*
    for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
        uint o = gidx / (numY * numX);
        uint plane_remain = gidx % (numY * numX);
        uint j = plane_remain / numX;
        uint i = plane_remain % numX;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        a[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 0]);
        b[((o * numZ) + j) * numZ + i] = dtInv - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 1]);
        c[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 2]);
    }
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

    if (gidx < outer * numX * numY) {
        uint o = gidx / (numY * numX);
        uint plane_remain = gidx % (numY * numX);
        uint j = plane_remain / numX;
        uint i = plane_remain % numX;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        a[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 0]);
        b[((o * numZ) + j) * numZ + i] = dtInv - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 1]);
        c[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 2]);
    }
}

__global__ void Rollback_3Coalesced (
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    REAL* myTimeline, 
    REAL* myDxxT,
    REAL* myVarXT,
    REAL* a,
    REAL* b,
    REAL* c
){
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

    if (gidx < outer * numY * numX) {
        uint o = gidx / (numY * numX);
        uint plane_remain = gidx % (numY * numX);
        uint i = plane_remain % numX;
        uint j = plane_remain / numX;

        uint numZ = max(numX,numY);

        __syncthreads();
        //Just 2 reads, should not require shared memory to spare one memory 1 cycle
        //Could also just have two threads read at once? then put into shared?
        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);

        __syncthreads();
        a[((o * numZ) + j) * numZ + i] =       - 0.5*(0.5*myVarXT[((t * numY) + j) * numY + i]*myDxxT[0 * numX + i]);
        b[((o * numZ) + j) * numZ + i] = dtInv - 0.5*(0.5*myVarXT[((t * numY) + j) * numY + i]*myDxxT[1 * numX + i]);
        c[((o * numZ) + j) * numZ + i] =       - 0.5*(0.5*myVarXT[((t * numY) + j) * numY + i]*myDxxT[2 * numX + i]);
    }
}

__global__ void Rollback_5 (
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    REAL* myTimeline,
    REAL* myDyy,
    REAL* myVarY,
    REAL* a,
    REAL* b,
    REAL* c
){
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

    if (gidx < outer * numX * numY) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        a[((o * numZ) + i) * numZ + j] =	   - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 0]);
        b[((o * numZ) + i) * numZ + j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 1]);
        c[((o * numZ) + i) * numZ + j] =	   - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 2]);
    }
}

__global__ void Rollback_5Coalesced (
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    REAL* myTimeline,
    REAL* myDyyT,
    REAL* myVarY,
    REAL* a,
    REAL* b,
    REAL* c
){
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

    if (gidx < outer * numX * numY) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);

        __syncthreads();
        //Just 2 reads, should not require shared memory to spare one memory 1 cycle
        //Could also just have two threads read at once? then put into shared?
        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);

        __syncthreads();
        a[((o * numZ) + i) * numZ + j] =       - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyyT[0 * numY + j]);
        b[((o * numZ) + i) * numZ + j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyyT[1 * numY + j]);
        c[((o * numZ) + i) * numZ + j] =       - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyyT[2 * numY + j]);
    }
}

__global__ void Rollback_6 (
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    REAL* myTimeline,
    REAL* u,
    REAL* v,
    REAL* y
){
    //cout << "test 6" << endl;
    /*
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        y[((o * numZ) + i) * numZ + j] = dtInv*u[((o * numY) + j) * numX + i] - 0.5*v[((o * numX) + i) * numY + j];
    }
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

    if (gidx < outer * numX * numY) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        y[((o * numZ) + i) * numZ + j] = dtInv*u[((o * numY) + j) * numX + i] - 0.5*v[((o * numX) + i) * numY + j];
    }
}

__global__ void Rollback_6Coalesced (
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    REAL* myTimeline,
    REAL* uT,
    REAL* v,
    REAL* y
){
    //cout << "test 6" << endl;
    /*
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        y[((o * numZ) + i) * numZ + j] = dtInv*u[((o * numY) + j) * numX + i] - 0.5*v[((o * numX) + i) * numY + j];
    }
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

    if (gidx < outer * numX * numY) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);

        __syncthreads();
        //Just 2 reads, should not require shared memory to spare one memory 1 cycle
        //Could also just have two threads read at once? then put into shared?
        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);

        __syncthreads();
        y[((o * numZ) + i) * numZ + j] = dtInv*uT[((o * numX) + i) * numY + j] - 0.5*v[((o * numX) + i) * numY + j];
    }
}


__global__ void Rollback_7 (
    int t,
    const int blocksize, 
    const int sgm_size, 
    const uint outer, 
    const uint numT, 
    const uint numX, 
    const uint numY, 
    REAL* myTimeline, 
    REAL* myDxx,
    REAL* myDyy,
    REAL* myVarX,
    REAL* myVarY,
    REAL* u,
    REAL* v,
    REAL* a,
    REAL* b,
    REAL* c,
    REAL* y,
    REAL* yy,
    REAL* myResult
){
    //cout << "test 7" << endl;
    /*
    for(uint i=0;i<numX;i++) {
        for (int gidx = 0; gidx < outer; gidx++) {
            // here yy should have size [numY]
            uint numZ = max(numX,numY);
            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
            tridagPar(a,((gidx * numZ) + i) * numZ,b,((gidx * numZ) + i) * numZ,c,((gidx * numZ) + i) * numZ,y,((gidx * numZ) + i) * numZ,numY,myResult, (gidx * numX + i) * numY,yy,(gidx * numZ));
        }
    }
    */
    
    //uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

}

#endif