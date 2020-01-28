#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include <cuda_runtime.h>
#include "Constants.h"
#include "TridagKernel.cu.h"

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

__global__ void Rollback_1 (
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

__global__ void Rollback_2 (
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

__global__ void Rollback_3 (
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

__global__ void Rollback_4 (
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
    //cout << "test 4" << endl;
    /*
    for(uint j=0;j<numY;j++) {
        for (int gidx = 0; gidx < outer; gidx++) {
            uint numZ = max(numX,numY);
            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
            // here yy should have size [numX]
            tridagPar(a,((gidx * numZ) + j) * numZ,b,((gidx * numZ) + j) * numZ,c,((gidx * numZ) + j) * numZ,u,((gidx * numY) + j) * numX,numX,u,((gidx * numY) + j) * numX,yy,(gidx * numZ));
        }
    }
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

}

__global__ void Rollback_5 (
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
    //cout << "test 5" << endl;
    /*
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        a[((o * numZ) + i) * numZ + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 0]);
        b[((o * numZ) + i) * numZ + j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 1]);
        c[((o * numZ) + i) * numZ + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 2]);
    }
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

}

__global__ void Rollback_6 (
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
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;

}

#endif