#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include "cuda.h"
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
        dl = (col == 0) ? 0.0 : myZ[col] - myZ[col - 1];
        du = (col == numZ - 1) ? 0.0 : myZ[col + 1] - myZ[col];
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
        myVarY[gidx] = exp(2.0*(  alpha*log(myX[i])   
                                    + myY[j]             
                                    - 0.5*nu*nu*myTimeline[t] )
                                ); // nu*nu
    }
}

#define SGM_SIZE 8
__global__ void Rollback(
    uint t, 
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
) {
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < outer) {
        uint numZ = max(numX,numY);

        uint i, j;

        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);

        //vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
        //vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
        //vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
        //vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

        //cout << "explicit x, t: " << t << " o: " << gidx << endl;
        //	explicit x
        for(i=0;i<numX;i++) {
            for(j=0;j<numY;j++) {
                u[((gidx * numY) + j) * numX + i] = dtInv*myResult[((gidx * numX) + i) * numY + j];

                if(i > 0) { 
                    u[((gidx * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                    * myDxx[i * 4 + 0] ) 
                                    * myResult[((gidx * numX) + (i-1)) * numY + j];
                }
                u[((gidx * numY) + j) * numX + i]  +=  0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                * myDxx[i * 4 + 1] )
                                * myResult[((gidx * numX) + i) * numY + j];
                if(i < numX-1) {
                    u[((gidx * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                    * myDxx[i * 4 + 2] )
                                    * myResult[((gidx * numX) + (i+1)) * numY + j];
                }
            }
        }

        //cout << "explicit y, t: " << t << " o: " << gidx << endl;
        //	explicit y
        for(j=0;j<numY;j++)
        {
            for(i=0;i<numX;i++) {
                v[((gidx * numX) + i) * numY + j] = 0.0;

                if(j > 0) {
                    v[((gidx * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                    * myDyy[j * 4 + 0] )
                                    * myResult[((gidx * numX) + i) * numY + j - 1];
                }
                v[((gidx * numX) + i) * numY + j]  += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                    * myDyy[j * 4 + 1] )
                                    * myResult[((gidx * numX) + i) * numY + j];
                if(j < numY-1) {
                    v[((gidx * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                    * myDyy[j * 4 + 2] )
                                    * myResult[((gidx * numX) + i) * numY + j + 1];
                }
                u[((gidx * numY) + j) * numX + i] += v[((gidx * numX) + i) * numY + j]; 
            }
        }

        //cout << "implicit x, t: " << t << " o: " << gidx << endl;
        //	implicit x
        for(j=0;j<numY;j++) {
            for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
                a[(gidx * numZ) + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 0]);
                b[(gidx * numZ) + i] = dtInv - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 1]);
                c[(gidx * numZ) + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 2]);
            }
            // here yy should have size [numX]
            uint num_blocks = (outer + blockDim.x - 1) / blockDim.x;
            //Dynamic parallelism (kernel child)
            TRIDAG_SOLVER<<<num_blocks, blockDim.x, blockDim.x * 32>>>(&a[gidx * numZ],&b[gidx * numZ],&c[gidx * numZ],&u[((gidx * numY) + j) * numX],numX,SGM_SIZE,&u[((gidx * numY) + j) * numX],&yy[gidx * numZ]);
            cudaDeviceSynchronize();
            __syncthreads();
        }

        //cout << "implicit y, t: " << t << " o: " << gidx << endl;
        //	implicit y
        for(i=0;i<numX;i++) { 
            for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
                a[(gidx * numZ) + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 0]);
                b[(gidx * numZ) + j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 1]);
                c[(gidx * numZ) + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 2]);
            }

            for(j=0;j<numY;j++)
                y[(gidx * numZ) + j] = dtInv*u[((gidx * numY) + j) * numX + i] - 0.5*v[((gidx * numX) + i) * numY + j];

            // here yy should have size [numY]
            uint num_blocks = (outer + blockDim.x - 1) / blockDim.x;
            TRIDAG_SOLVER<<<num_blocks, blockDim.x, blockDim.x * 32>>>(&a[gidx * numZ],&b[gidx * numZ],&c[gidx * numZ],&y[gidx * numZ],numY, SGM_SIZE,&myResult[(gidx * numX + i) * numY],&yy[gidx * numZ]);
            cudaDeviceSynchronize();
            __syncthreads();
        }
    }
}

#endif