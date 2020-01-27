#ifndef KERNEL_ALGORITHM
#define KERNEL_ALGORITHM

int run_SimpleKernel(
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
);

#endif