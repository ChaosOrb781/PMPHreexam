#include "OriginalAlgorithm.h"
#include <omp.h>

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
    int procs = 0;
#pragma omp parallel for
    for( unsigned i = 0; i < outer; ++ i ) {
        {
            int th_id = omp_get_thread_num();
            if(th_id == 0) { procs = omp_get_num_threads(); }
        }
        REAL strike = i * 0.001;
        PrivGlobs    globs(numX, numY, numT);
        res[i] = value( globs, s0, strike, t,
                        alpha, nu,    beta,
                        numX,  numY,  numT );
    }
    return procs;
}

int   run_SimpleParallelStatic(  
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
    int procs = 0;
#pragma omp parallel for schedule(static)
    for( unsigned i = 0; i < outer; ++ i ) {
        {
            int th_id = omp_get_thread_num();
            if(th_id == 0) { procs = omp_get_num_threads(); }
        }
        REAL strike = i * 0.001;
        PrivGlobs    globs(numX, numY, numT);
        res[i] = value( globs, s0, strike, t,
                        alpha, nu,    beta,
                        numX,  numY,  numT );
    }
    return procs;
}

int   run_SimpleParallelDynamic(  
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
    int procs = 0;
    omp_set_dynamic(32);
#pragma omp parallel for schedule(dynamic)
    for( unsigned i = 0; i < outer; ++ i ) {
        {
            int th_id = omp_get_thread_num();
            if(th_id == 0) { procs = omp_get_num_threads(); }
        }
        REAL strike = i * 0.001;
        PrivGlobs    globs(numX, numY, numT);
        res[i] = value( globs, s0, strike, t,
                        alpha, nu,    beta,
                        numX,  numY,  numT );
    }
    return procs;
}

//#endif // PROJ_CORE_ORIG
