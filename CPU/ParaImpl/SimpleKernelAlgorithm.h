#ifndef SIMPLE_KERNEL_ALGORITHM
#define SIMPLE_KERNEL_ALGORITHM
#include "CUDAKernels.h"


int run_SimpleKernel(
                const uint   outer,
                const uint   numX,
                const uint   numY,
                const uint   numT,
                const REAL   s0,
                const REAL   t, 
                const REAL   alpha, 
                const REAL   nu, 
                const REAL   beta,
                const uint   blocksize,
                      REAL*  res   // [outer] RESULT
) {
    hvec<REAL>               dmyX(numX);       // [numX]
    hvec<REAL>               dmyY(numY);       // [numY]
    hvec<REAL>               dmyTimeline(numT);// [numT]
    hvec<hvec<REAL> >        dmyDxx(numX);     // [numX][4]
    hvec<hvec<REAL> >        dmyDyy(numY);     // [numY][4]
    hvec<hvec<hvec<REAL> > > dmyResult(outer); // [outer][numX][numY]
    hvec<hvec<hvec<REAL> > > dmyVarX(numT);    // [numT][numX][numY]
    hvec<hvec<hvec<REAL> > > dmyVarY(numT);    // [numT][numX][numY]

    int myXindex = 0, myYindex = 0;
    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    int myXindex = static_cast<unsigned>(s0/dx) % numX;
    
    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    myYindex = static_cast<unsigned>(numY/2.0);

    for (int i = 0; i < numX; i++) {
        dmyDxx[i].resize(4);
    }
    for (int i = 0; i < numY; i++) {
        dmyDyy[i].resize(4);
    }
    for (int i = 0; i < outer; i++) {
        dmyResult[i].resize(numX);
        for (int j = 0; j < numX; j++) {
            dmyResult[i][j].resize(numY);
        }
    }
    for (int i = 0; i < numT; i++) {
        dmyVarX[i].resize(numX);
        dmyVarY[i].resize(numX);
        for (int j = 0; j < numX; j++) {
            dmyVarX[i][j].resize(numY);
            dmyVarY[i][j].resize(numY);
        }
    }

	initGrid_Alt(s0, alpha, nu, t, numX, numY, numT, dmyX, dmyY, dmyTimeline, myXindex, myYindex);
	initOperator_Alt(numX, myX, myDxx);
    initOperator_Alt(numY, myY, myDyy);
    setPayoff_Alt(myX, outer, numX, numY, myResult);

    updateParams_Alt(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
	rollback_Alt(outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, myResult);
	
	for(uint i = 0; i < outer; i++) {
        res[i] = myResult[i][myXindex][myYindex];
    }
    return blocksize;
}
#endif