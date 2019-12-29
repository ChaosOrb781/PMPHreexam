#ifndef INTERCHANGED_ALGORITHM
#define INTERCHANGED_ALGORITHM

#include "OriginalAlgorithm.h"

REAL   value_interchanged(
  int outer_i,
  const REAL s0,
	//const REAL strike,
	const REAL t,
	const REAL alpha,
	const REAL nu,
	const REAL beta,
	const unsigned int numX,
	const unsigned int numY,
	const unsigned int numT,
  const unsigned int outer,
  REAL* res
) {

	for (int i = 0; i < outer; ++i)
	{
    PrivGlobs globs(numX, numY, numT);
    initGrid(s0, alpha, nu, t, numX, numY, numT, globs);
    initOperator(globs.myX, globs.myDxx);
    initOperator(globs.myY, globs.myDyy);
    REAL strike = 0.001 * outer_i;
    setPayoff(strike, globs);
		updateParams(i, alpha, beta, nu, globs);
		rollback(i, globs);
    res[outer_i] = globs.myResult[globs.myXindex][globs.myYindex];
	}
}

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
	REAL* res   // [outer] RESULT
) {
	for (unsigned i = numT - 2; i >= 0; --i) {
		value_interchanged(i, s0, t,
			alpha, nu, beta,
			numX, numY, numT, outer, res);
	}
	return 1;
}

#endif