#ifndef KERNELIZED_ALGORITHM
#define KERNELIZED_ALGORITHM

#include "OriginalAlgorithm.h"
#include <omp.h>

int   run_Kernelized(  
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
	vector<PrivGlobs> globstastic;
	globstastic.resize(outer); //Generates list from default constructor
	for( unsigned ii = 0; ii < outer; ii += blocksize ) {
        for ( unsigned i = ii; i < min(outer, ii + blocksize); i++) {
            //Initialize each object as if called by the size constructor
            globstastic[i].Initialize(numX, numY, numT);
            initGrid(s0,alpha,nu,t, numX, numY, numT, globstastic[i]);
            initOperator(globstastic[i].myX,globstastic[i].myDxx);
            initOperator(globstastic[i].myY,globstastic[i].myDyy);
            REAL strike = 0.001*i;
            setPayoff(strike, globstastic[i]);
        }
	}
	for(int j = 0;j<=numT-2;++j) {
		for( unsigned ii = 0; ii < outer; ii += blocksize ) {
		    for( unsigned i = ii; i < min(outer, ii + blocksize); ++ i ) {
			{
				updateParams(j,alpha,beta,nu,globstastic[i]);
				rollback(j, globstastic[i]);
			}
		}
    }
	for( unsigned ii = 0; ii < outer; ii += blocksize ) {
        for( unsigned i = ii; i < min(outer, ii + blocksize); ++ i ) {
            res[i] = globstastic[i].myResult[globstastic[i].myXindex][globstastic[i].myYindex];
        }
    }
    return 1;
}
#endif