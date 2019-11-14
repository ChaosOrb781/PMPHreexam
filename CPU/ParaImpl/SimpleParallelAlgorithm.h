#include "OriginalAlgorithm.h"

int   run_SimpleParallel(  
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

#pragma omp parallel for private(globs, strike)
    for( unsigned i = 0; i < outer; ++ i ) {
        strike = 0.001*i;
        res[i] = value( globs, s0, strike, t,
                        alpha, nu,    beta,
                        numX,  numY,  numT );
    }
}

//#endif // PROJ_CORE_ORIG
