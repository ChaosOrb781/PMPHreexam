#ifndef INTERCHANGED_ALGORITHM
#define INTERCHANGED_ALGORITHM

#include "OriginalAlgorithm.h"

REAL   value_interchanged(PrivGlobs    globs,
	const REAL s0,
	const REAL strike,
	const REAL t,
	const REAL alpha,
	const REAL nu,
	const REAL beta,
	const unsigned int numX,
	const unsigned int numY,
	const unsigned int numT
) {
	initGrid(s0, alpha, nu, t, numX, numY, numT, globs);
	initOperator(globs.myX, globs.myDxx);
	initOperator(globs.myY, globs.myDyy);

	setPayoff(strike, globs);
	for (int i = globs.myTimeline.size() - 2; i >= 0; --i)
	{
		updateParams(i, alpha, beta, nu, globs);
		rollback(i, globs);
	}

	return globs.myResult[globs.myXindex][globs.myYindex];
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
	REAL strike;
	PrivGlobs    globs(numX, numY, numT);

	for (unsigned i = 0; i < outer; ++i) {
		strike = 0.001 * i;
		res[i] = value_interchanged(globs, s0, strike, t,
			alpha, nu, beta,
			numX, numY, numT);
	}
	return 1;
}

#endif