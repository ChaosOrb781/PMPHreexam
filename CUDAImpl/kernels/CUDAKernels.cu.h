#ifndef CALIB_KERS
#define CALIB_KERS

#include <cuda_runtime.h>
#include "../includes/Constants.cu.h"

__global__ void MyTimeline(REAL* myTimeline, const REAL t, const uint numT) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numT) {
        myTimeline[gidx] = t * gidx / (numT - 1);
    }
}

__global__ void MyX(REAL* myX, const REAL s0, const REAL dx, 
        const uint myXindex, const uint numX) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numX) {
        myX[gidx] = gidx*dx - myXindex*dx + s0;
    }
}

__global__ void MyY(REAL* myY, const REAL logAlpha, const REAL dy, 
        const uint myYindex, const uint numY) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numY) {
        myY[gidx] = gidx*dy - myYindex*dy + logAlpha;
    }
}

////DEVICE SYNCHRONIZE!////

__global__ void Dl1(REAL* myZ, REAL* dl, const uint numZ) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x + 1;
    if (gidx > 0 && gidx < numZ - 1) {
        dl[gidx] = myZ[gidx];
    }
}

__global__ void Dl2(REAL* myZ, REAL* dl, const uint numZ) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x + 1;
    if (gidx > 0 && gidx < numZ - 1) {
        dl[gidx] -= myZ[gidx-1];
    }
}

__global__ void Du1(REAL* myZ, REAL* du, const uint numZ) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x + 1;
    if (gidx > 0 && gidx < numZ - 1) {
        du[gidx] = myZ[gidx+1];
    }
}

__global__ void Du2(REAL* myZ, REAL* du, const uint numZ) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x + 1;
    if (gidx > 0 && gidx < numZ - 1) {
        du[gidx] -= myZ[gidx];
    }
}

////DEVICE SYNCHRONIZE!////

__global__ void TrMyDzz(REAL* trMyDzz, REAL* dl, REAL* du,
        const uint col_max) {
    unsigned gidx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned row = gidx / col_max;
    unsigned col = gidx % col_max; //thread offset: 1
    if (gidx < 4 * col_max) {
        if (col > 0 && col < col_max-1) {
            trMyDzz[gidx] = 
                (row == 0) ?  2.0/dl[col]/(dl[col]+du[col]) :
                ((row == 1) ? -2.0*(1.0/dl[col] + 1.0/du[col])/(dl[col]+du[col]) :
                ((row == 2) ?  2.0/du[col]/(dl[col]+du[col]) :
                0.0)); //row == 4 -> 0.0
        } else {
            trMyDzz[gidx] =  0.0;
        }
    }
}

//initialize with shared memory size: numX+1
__global__ void MyResult(REAL* myResult, REAL* myX, 
        const uint outer, const uint numX, const uint numY) {
    extern __shared__ REAL myXs[];
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    //copy to shared due to uncoalesced access
    //index for myX is always below or equal to column
    if (gidx < numX) {
        myXs[gidx] = myX[gidx];
    }
    __syncthreads();

    unsigned row = gidx / (numX * numY);
    unsigned row_remain = gidx % (numX * numY);
    unsigned col = row_remain / numY;
    //Shared memory for myX
    if (gidx < outer * numX * numY) {
        REAL strike = 0.001*row;
        REAL payoff = max(myXs[col]-strike, (REAL)0.0);
        myResult[gidx] = payoff;
    }
}

__global__ void MyResult_dif(REAL* myResult, REAL* myX, 
        const uint outer, const uint numX, const uint numY) {
    extern __shared__ REAL myXs[];
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned row = gidx / (numX * numY);
    unsigned row_remain = gidx % (numX * numY);
    unsigned col = row_remain / numY;
    //Shared memory for myX
    if (gidx < outer * numX * numY) {
        REAL strike = 0.001*row;
        REAL payoff = max(myX[col]-strike, (REAL)0.0);
        myResult[gidx] = payoff;
    }
}

__global__ void MyResult_dif_test(REAL* myResult, REAL* myX, 
        const uint outer, const uint numX, const uint numY) {
    extern __shared__ REAL myXs[];
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned row = gidx / (numX * numY);
    unsigned row_remain = gidx % (numX * numY);
    unsigned col = row_remain / numY;
    //Shared memory for myX
    if (gidx < outer * numX * numY) {
        REAL strike = 0.001*row;
        REAL payoff = max(myX[col] - strike, (REAL)0.0);
        myResult[gidx] = payoff;
    }
}

////DEVICE SYNCHRONIZE!////

//initialize with shared memory size: numX+1
__global__ void MyVarXY1(REAL* myVarX, REAL* myVarY, REAL* myX, 
        REAL beta, REAL alpha, const uint numT, const uint numX, const uint numY) {
    extern __shared__ REAL myXs[];
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    //copy to shared due to uncoalesced access
    //index for myX is always below or equal to column
    if (gidx < numX) {
        myXs[gidx] = myX[gidx];
    }
    __syncthreads();

    unsigned row_remain = gidx % (numX * numY);
    unsigned col = row_remain / numY;
    //Shared memory for myX
    if (gidx < numT * numX * numY) {
        myVarX[gidx] = beta  * log(myXs[col]);
        myVarY[gidx] = alpha * log(myXs[col]);
    }
}

__global__ void MyVarXY1_dif(REAL* myVarX, REAL* myVarY, REAL* myX, 
        REAL beta, REAL alpha, const uint numT, const uint numX, const uint numY) {
    extern __shared__ REAL myXs[];
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned row_remain = gidx % (numX * numY);
    unsigned col = row_remain / numY;
    if (gidx < numT * numX * numY) {
        myVarX[gidx] = beta  * log(myX[col]);
        myVarY[gidx] = alpha * log(myX[col]);
    }
}

__global__ void MyVarXY2(REAL* myVarX, REAL* myVarY, REAL* myY,
        const uint numT, const uint numX, const uint numY) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned depth = gidx % numY;
    if (gidx < numT * numX * numY) {
        myVarX[gidx] += myY[depth];
        myVarY[gidx] += myY[depth];
    }
}

////DEVICE SYNCHRONIZE!////

//initialize with shared memory size: numT+1
__global__ void MyVarXY3(REAL* myVarX, REAL* myVarY, REAL* myTimeline,
        REAL nu, const uint numT, const uint numX, const uint numY) {
    extern __shared__ REAL myTimeline_s[];
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    //copy to shared due to uncoalesced access
    //index for myTimeline is always below or equal to column
    if (gidx < numT) {
        myTimeline_s[gidx] = myTimeline[gidx];
    }
    __syncthreads();

    unsigned row = gidx / (numX * numY);
    //Shared memory for myTimeline
    if (gidx < numT * numX * numY) {
        myVarX[gidx] = exp(2.0*(myVarX[gidx] - 0.5*nu*nu*myTimeline_s[row]));
        myVarY[gidx] = exp(2.0*(myVarY[gidx] - 0.5*nu*nu*myTimeline_s[row]));// nu*nu
    }
}

__global__ void MyVarXY3_dif(REAL* myVarX, REAL* myVarY, REAL* myTimeline,
        REAL nu, const uint numT, const uint numX, const uint numY) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned row = gidx / (numX * numY);
    if (gidx < numT * numX * numY) {
        myVarX[gidx] = exp(2.0*(myVarX[gidx] - 0.5*nu*nu*myTimeline[row]));
        myVarY[gidx] = exp(2.0*(myVarY[gidx] - 0.5*nu*nu*myTimeline[row]));// nu*nu
    }
}

__global__ void DtInv1(REAL* dtInv, REAL* myTimeline, const uint numT) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx < numT - 1) {
        dtInv[gidx] = myTimeline[gidx+1];
    }
}


////DEVICE SYNCHRONIZE!////

__global__ void DtInv2(REAL* dtInv, REAL* myTimeline, const uint numT) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx < numT - 1) {
        dtInv[gidx] = 1.0/(dtInv[gidx]-myTimeline[gidx]);
    }
}

////DEVICE SYNCHRONIZE!////

//initialize with shared memory size: numT+1
__global__ void ABCX(REAL* aX, REAL* bX, REAL* cX, REAL* dtInv,
        REAL* trMyVarX, REAL* trMyDxx,
        const uint numT, const uint numY, const uint numX) {
    extern __shared__ REAL dtInvs[];
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    //copy to shared due to uncoalesced access
    //index for myTimeline is always below or equal to column
    if (gidx < numT) {
        dtInvs[gidx] = dtInv[gidx];
    }
    __syncthreads();

    unsigned row = gidx / (numX * numY);
    unsigned depth = gidx % numX; //thread offset: 1
    //Shared memory for dtInv
    if (gidx < numT * numY * numX) {
        aX[gidx] =		       - 0.5*(0.5*trMyVarX[gidx]*trMyDxx[0 * numX + depth]);
        bX[gidx] = dtInvs[row] - 0.5*(0.5*trMyVarX[gidx]*trMyDxx[1 * numX + depth]);
        cX[gidx] =		       - 0.5*(0.5*trMyVarX[gidx]*trMyDxx[2 * numX + depth]);
    }
}

__global__ void ABCX_dif(REAL* aX, REAL* bX, REAL* cX, REAL* dtInv,
        REAL* trMyVarX, REAL* trMyDxx,
        const uint numT, const uint numY, const uint numX) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned row = gidx / (numX * numY);
    unsigned depth = (gidx % (numX * numY)) % numX; //thread offset: 1
    //Shared memory for dtInv
    if (gidx < numT * numY * numX) {
        aX[gidx] =		      - 0.5*(0.5*trMyVarX[gidx]*trMyDxx[0 * numX + depth]);
        bX[gidx] = dtInv[row] - 0.5*(0.5*trMyVarX[gidx]*trMyDxx[1 * numX + depth]);
        cX[gidx] =		      - 0.5*(0.5*trMyVarX[gidx]*trMyDxx[2 * numX + depth]);
    }
}

__global__ void AX_dif(REAL* aX, REAL* dtInv,
        REAL* trMyVarX, REAL* trMyDxx,
        const uint numT, const uint numY, const uint numX) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    //unsigned row = gidx / (numX * numY);
    unsigned depth = (gidx % (numX * numY)) % numX; //thread offset: 1
    //Shared memory for dtInv
    if (gidx < numT * numY * numX) {
        aX[gidx] = - 0.5*(0.5*trMyVarX[gidx]*trMyDxx[depth]);
    }
}

__global__ void BX_dif(REAL* bX, REAL* dtInv,
        REAL* trMyVarX, REAL* trMyDxx,
        const uint numT, const uint numY, const uint numX) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned row = gidx / (numX * numY);
    unsigned depth = (gidx % (numX * numY)) % numX; //thread offset: 1
    //Shared memory for dtInv
    if (gidx < numT * numY * numX) {
        bX[gidx] = dtInv[row] - 0.5*(0.5*trMyVarX[gidx]*trMyDxx[1 * numX + depth]);
    }
}

__global__ void CX_dif(REAL* cX, REAL* dtInv,
        REAL* trMyVarX, REAL* trMyDxx,
        const uint numT, const uint numY, const uint numX) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    //unsigned row = gidx / (numX * numY);
    unsigned depth = (gidx % (numX * numY)) % numX; //thread offset: 1
    //Shared memory for dtInv
    if (gidx < numT * numY * numX) {
        cX[gidx] = - 0.5*(0.5*trMyVarX[gidx]*trMyDxx[2 * numX + depth]);
    }
}

//initialize with shared memory size: numT+1
__global__ void ABCY(REAL* aY, REAL* bY, REAL* cY, REAL* dtInv,
        REAL* myVarY, REAL* trMyDyy,
        const uint numT, const uint numX, const uint numY) {
    extern __shared__ REAL dtInvs[];
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    //copy to shared due to uncoalesced access
    //index for myTimeline is always below or equal to column
    if (gidx < numT) {
        dtInvs[gidx] = dtInv[gidx];
    }
    __syncthreads();

    unsigned row = gidx / (numX * numY);
    unsigned depth = (gidx % (numX * numY)) % numY; //thread offset: 1
    if (gidx < numT * numX * numY) {
        aY[gidx] =		       - 0.5*(0.5*myVarY[gidx]*trMyDyy[0 * numY + depth]);
        bY[gidx] = dtInvs[row] - 0.5*(0.5*myVarY[gidx]*trMyDyy[1 * numY + depth]);
        cY[gidx] =		       - 0.5*(0.5*myVarY[gidx]*trMyDyy[2 * numY + depth]);
    }
}

__global__ void ABCY_dif(REAL* aY, REAL* bY, REAL* cY, REAL* dtInv,
        REAL* myVarY, REAL* trMyDyy,
        const uint numT, const uint numX, const uint numY) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned row = gidx / (numX * numY);
    unsigned depth = (gidx % (numX * numY)) % numY; //thread offset: 1
    if (gidx < numT * numX * numY) {
        aY[gidx] =		      - 0.5*(0.5*myVarY[gidx]*trMyDyy[0 * numY + depth]);
        bY[gidx] = dtInv[row] - 0.5*(0.5*myVarY[gidx]*trMyDyy[1 * numY + depth]);
        cY[gidx] =		      - 0.5*(0.5*myVarY[gidx]*trMyDyy[2 * numY + depth]);
    }
}

__global__ void U1(REAL* u, REAL* trMyVarX, REAL* trMyDxx, REAL* trMyResult, 
        REAL dtInv_plane, unsigned plane,
        const uint outer, const uint numX, const uint numY) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned row_remain = gidx % (numX * numY);
    unsigned col = row_remain / numX;
    unsigned depth = row_remain % numX; //thread offset: 1

    if (gidx < outer * numY * numX) {
        u[gidx] = dtInv_plane*trMyResult[gidx];

        if(depth > 0) { 
            u[gidx] += 0.5*( 0.5*trMyVarX[(plane * numY + col) * numX + depth]*trMyDxx[0 * numX + depth] ) 
                        * trMyResult[gidx-1];
        }
        u[gidx]     +=  0.5*( 0.5*trMyVarX[(plane * numY + col) * numX + depth]*trMyDxx[1 * numX + depth] )
                        * trMyResult[gidx];
        if(depth < numX-1) {
            u[gidx] += 0.5*( 0.5*trMyVarX[(plane * numY + col) * numX + depth]*trMyDxx[2 * numX + depth] )
                        * trMyResult[gidx+1];
        }
    }
}

__global__ void V(REAL* v, REAL* myVarY, REAL* trMyDyy, REAL* myResult, 
        unsigned plane, const uint outer, const uint numX, const uint numY) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned row_remain = gidx % (numX * numY);
    unsigned col = row_remain / numY;
    unsigned depth = row_remain % numY; //thread offset: 1

    if (gidx < outer * numX * numY) {
        v[gidx] = 0.0;

        if(depth > 0) {
            v[gidx] +=  ( 0.5*myVarY[(plane * numX + col) * numY + depth]*trMyDyy[0 * numY + depth] )
                    *  myResult[gidx-1];
        }
        v[gidx]     +=   ( 0.5*myVarY[(plane * numX + col) * numY + depth]*trMyDyy[1 * numY + depth] )
                    *  myResult[gidx];
        if(depth < numY-1) {
            v[gidx] +=  ( 0.5*myVarY[(plane * numX + col) * numY + depth]*trMyDyy[2 * numY + depth] )
                    *  myResult[gidx+1];
        } 
    }
}

__global__ void U2(REAL* u, REAL* trV,
        const uint outer, const uint numX, const uint numY) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx < outer * numX * numY) {
        u[gidx] += trV[gidx];
    }
}

__global__ void Y(REAL* y, REAL* trU, REAL* v, REAL dtInv_plane,
        const uint outer, const uint numX, const uint numY) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx < outer * numX * numY) {
        y[gidx] = dtInv_plane*trU[gidx] - 0.5*v[gidx];
    }
}


//Shared memory = n * sizeof(MyReal4) + n * sizeof(MyReal2);
__global__ void SeqTridagPar1(REAL* aX, REAL* bX, REAL* cX, REAL* u, unsigned i, unsigned j, unsigned k, unsigned numX, unsigned numY, REAL* yy) {
    //Just some 4 byte type to then later split
    extern __shared__ MyReal4 arrs[];
    MyReal4* mats  = arrs; 
    //Offset by the size of mats and recast the remaining shared memory to MyReal2
    MyReal2* lfuns = (MyReal2*) &arrs[numX];

    if (threadIdx.x == 0) {
        //MyReal4* mats = (MyReal4*)malloc(n*sizeof(MyReal4));    // supposed to be in shared memory!
        REAL b0 = bX[(j * numY + k) * numX + 0];
        for(int h=0; h<numX; h++) { //parallel, map-like semantics
            if (h==0) { 
                mats[h].x = 1.0;  
                mats[h].y = 0.0;          
                mats[h].z = 0.0; 
                mats[h].w = 1.0; 
            } else { 
                mats[h].x = bX[(j * numY + k) * numX + h]; 
                mats[h].y = -aX[(j * numY + k) * numX + h]*cX[(j * numY + k) * numX + h-1]; 
                mats[h].z = 1.0; 
                mats[h].w = 0.0; 
            }
        }
        inplaceScanInc<MatMult2b2>(numX,mats);
        for(int h=0; h<numX; h++) { //parallel, map-like semantics
            yy[h] = (mats[h].x*b0 + mats[h].y) / (mats[h].z*b0 + mats[h].w);
        }

        REAL y0 = u[(i * numY + k) * numX + 0];
        for(int h=0; h<numX; h++) { //parallel, map-like semantics
            if (h==0) { 
                lfuns[0].x = 0.0;     
                lfuns[0].y = 1.0;           
            } else { 
                lfuns[h].x = u[(i * numY + k) * numX + h]; 
                lfuns[h].y = -aX[(j * numY + k) * numX + h]/yy[h-1]; }
        }
        inplaceScanInc<LinFunComp>(numX,lfuns);
        for(int h=0; h<numX; h++) { //parallel, map-like semantics
            u[(i * numY + k) * numX + h] = lfuns[h].x + y0*lfuns[h].y;
        }
        //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

        //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
        //S11.3, distribution and array expansion applied to i, j and k due to dependency
        //in other iterations
        REAL yn = u[(i * numY + k) * numX + numX-1]/yy[numX-1];
        for(int h=0; h<numX; h++) { //parallel, map-like semantics
            if (h==0) { 
                lfuns[0].x = 0.0;  
                lfuns[0].y = 1.0;           
            } else { 
                lfuns[h].x = u[(i * numY + k) * numX + numX-h-1]/yy[numX-h-1]; 
                lfuns[h].y = -cX[(j * numY + k) * numX + numX-h-1]/yy[numX-h-1]; }
        }
        inplaceScanInc<LinFunComp>(numX,lfuns);
        for(int h=0; h<numX; h++) { //parallel, map-like semantics
            u[(i * numY + k) * numX + numX-h-1] = lfuns[h].x + yn*lfuns[h].y;
        }
    }
}

//Shared memory = n * sizeof(MyReal4) + n * sizeof(MyReal2);
__global__ void SeqTridagPar2(REAL* aY, REAL* bY, REAL* cY, REAL* y, unsigned i, unsigned j, unsigned k, unsigned numY, unsigned numX, REAL* myResult, REAL* yy) {
    extern __shared__ MyReal4 arrs[];
    MyReal4* mats  = arrs; 
    //Offset by the size of mats and recast the remaining shared memory to MyReal2
    MyReal2* lfuns = (MyReal2*) &arrs[numY];

    if (threadIdx.x == 0) {
        //MyReal4* mats = (MyReal4*)malloc(n*sizeof(MyReal4));    // supposed to be in shared memory!
        REAL b0 = bY[(j * numX + k) * numY + 0];
        for(int h=0; h<numY; h++) { //parallel, map-like semantics
            if (h==0) { 
                mats[h].x = 1.0;  
                mats[h].y = 0.0;          
                mats[h].z = 0.0; 
                mats[h].w = 1.0; 
            } else { 
                mats[h].x = bY[(j * numX + k) * numY + h]; 
                mats[h].y = -aY[(j * numX + k) * numY + h]*cY[(j * numX + k) * numY + h-1]; 
                mats[h].z = 1.0; 
                mats[h].w = 0.0; 
            }
        }
        inplaceScanInc<MatMult2b2>(numY,mats);
        for(int h=0; h<numY; h++) { //parallel, map-like semantics
            yy[h] = (mats[h].x*b0 + mats[h].y) / (mats[h].z*b0 + mats[h].w);
        }

        REAL y0 = y[(i * numX + k) * numY + 0];
        for(int h=0; h<numY; h++) { //parallel, map-like semantics
            if (h==0) { 
                lfuns[0].x = 0.0;  
                lfuns[0].y = 1.0;           
            } else { 
                lfuns[h].x = y[(i * numX + k) * numY + h]; 
                lfuns[h].y = -aY[(j * numX + k) * numY + h]/yy[h-1]; 
            }
        }
        inplaceScanInc<LinFunComp>(numY,lfuns);
        for(int h=0; h<numY; h++) { //parallel, map-like semantics
            myResult[(i * numX + k) * numY + h] = lfuns[h].x + y0*lfuns[h].y;
        }
        
        REAL yn = myResult[(i * numX + k) * numY + numY-1]/yy[numY-1];
        for(int h=0; h<numY; h++) { //parallel, map-like semantics
            if (h==0) { 
                lfuns[0].x = 0.0;  
                lfuns[0].y = 1.0;           
            } else { 
                lfuns[h].x = myResult[(i * numX + k) * numY + numY-h-1]/yy[numY-h-1]; 
                lfuns[h].y = -cY[(j * numX + k) * numY + numY-h-1]/yy[numY-h-1]; 
            }
        }
        inplaceScanInc<LinFunComp>(numY,lfuns);
        for(int h=0; h<numY; h++) { //parallel, map-like semantics
            myResult[(i * numX + k) * numY + numY-h-1] = lfuns[h].x + yn*lfuns[h].y;
        }
    }
}

__global__ void Res(REAL* res, REAL* myResult, unsigned myXindex, unsigned myYindex,
        const uint outer, const uint numX, const uint numY) {
    unsigned gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx < outer) {
        res[gidx] = myResult[(gidx * numX + myXindex) * numY + myYindex];
    }
}

#endif //CALIB_KERS