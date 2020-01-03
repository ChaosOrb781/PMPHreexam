#ifndef INTERCHANGED_ALGORITHM
#define INTERCHANGED_ALGORITHM

#include "OriginalAlgorithm.h"

int   run_Interchanged(  
                const uint   outer,
                const uint   numX,
                const uint   numY,
                const uint   numT,
                const REAL   s0,
                const REAL   t, 
                const REAL   alpha, 
                const REAL   nu, 
                const REAL   beta,
                      REAL*  res   // [outer] RESULT
) {

	PrivGlobs* globstastic = (PrivGlobs*) malloc(sizeof(PrivGlobs) * numT);
	for( unsigned i = 0; i < numT; ++ i ) {
		PrivGlobs globs(numX, numY, numY);
		globstastic[i] = globs;
	}
	for( unsigned i = 0; i < outer; ++ i ) {
		REAL strike = 0.001*i;
        strike = 0.001*i;
		initGrid(s0,alpha,nu,t, numX, numY, numT, globstastic[i]);
		initOperator(globstastic[i].myX,globstastic[i].myDxx);
		initOperator(globstastic[i].myY,globstastic[i].myDyy);

		setPayoff(strike, globstastic[i]);
		for(int j = numT-2;j>=0;--j)
		{
			updateParams(j,alpha,beta,nu,globstastic[i]);
			rollback(j, globstastic[i]);
		}
        res[i] = globstastic[i].myResult[globstastic[i].myXindex][globstastic[i].myYindex];
    }
    return 1;
}

#endif