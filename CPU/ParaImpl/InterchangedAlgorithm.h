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
	vector<PrivGlobs> globstastic;
	for( unsigned i = 0; i < outer; ++ i ) {
		globstastic.push_back(PrivGlobs(numX, numY, numY));
		initGrid(s0,alpha,nu,t, numX, numY, numT, globstastic[i]);
		initOperator(globstastic[i].myX,globstastic[i].myDxx);
		initOperator(globstastic[i].myY,globstastic[i].myDyy);
		REAL strike = 0.001*i;
		setPayoff(strike, globstastic[i]);
	}
	for(int j = numT-2;j>=0;--j)
		{
		for( unsigned i = 0; i < outer; ++ i ) {
			{
				updateParams(j,alpha,beta,nu,globstastic[i]);
				rollback(j, globstastic[i]);
			}
		}
    }
	for( unsigned i = 0; i < outer; ++ i ) {
        res[i] = globstastic[i].myResult[globstastic[i].myXindex][globstastic[i].myYindex];
    }
    return 1;
}

#endif