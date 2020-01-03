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

	PrivGlobs* globstastic = (PrivGlobs*) malloc(sizeof(PrivGlobs) * outer);
	
	for( unsigned i = 0; i < outer; ++ i ) {
		PrivGlobs globs(numX, numY, numY);
		initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
		initOperator(globs.myX,globs.myDxx);
		initOperator(globs.myY,globs.myDyy);
		REAL strike = 0.001*i;
		setPayoff(strike, globs);
		globstastic[i] = globs;
	}
	for( unsigned i = 0; i < outer; ++ i ) {
		for(int j = numT-2;j>=0;--j)
		{
			updateParams(j,alpha,beta,nu,globstastic[i]);
			rollback(j, globstastic[i]);
		}
        res[i] = globstastic[i].myResult[globstastic[i].myXindex][globstastic[i].myYindex];
    }
	free(globstastic);
    return 1;
}

#endif