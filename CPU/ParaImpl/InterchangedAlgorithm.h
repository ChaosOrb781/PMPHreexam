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
    REAL strike;
    PrivGlobs    globs(numX, numY, numT);

    for( unsigned i = 0; i < outer; ++ i ) {
        strike = 0.001*i;
		initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
		initOperator(globs.myX,globs.myDxx);
		initOperator(globs.myY,globs.myDyy);

		setPayoff(strike, globs);
		for(int j = globs.myTimeline.size()-2;j>=0;--j)
		{
			updateParams(i,alpha,beta,nu,globs);
			rollback(i, globs);
		}
        res[i] = globs.myResult[globs.myXindex][globs.myYindex];
    }
    return 1;
}

#endif