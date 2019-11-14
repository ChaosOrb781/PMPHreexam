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
#pragma omp parallel for
    for( unsigned i = 0; i < outer; ++ i ) {
        REAL strike = i * 0.001;
        PrivGlobs    globs(numX, numY, numT);
        res[i] = value( globs, s0, strike, t,
                        alpha, nu,    beta,
                        numX,  numY,  numT );
    }
}

//#endif // PROJ_CORE_ORIG
