#ifndef SCAN_KERS
#define SCAN_KERS

#include <cuda_runtime.h>
#include "../includes/Constants.cu.h"
#include "../includes/CudaUtilProj.cu.h"

/*********************/
/*** Tridag Kernel ***/
/*********************/
// Try to optimize it: for example,
//    (The allocated shared memory is enough for 8 floats / thread): 
//    1. the shared memory space for "mat_sh" can be reused for "lin_sh"
//    2. with 1., now you have space to hold "u" and "uu" in shared memory.
//    3. you may hold "a[gid]" in a register, since it is accessed twice, etc.
__global__ void 
TRIDAG_SOLVER(  REAL* a,
                REAL* b,
                REAL* c,
                REAL* r,
                const unsigned int n,
                const unsigned int sgm_sz,
                REAL* u,
                REAL* uu
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + tid;

    // total shared memory (declared outside)
    extern __shared__ char sh_mem[];
    // shared memory space for the 2x2 matrix multiplication SCAN
    volatile MyReal4* mat_sh = (volatile MyReal4*)sh_mem;
    // shared memory space for the linear-function composition SCAN
    volatile MyReal2* lin_sh = (volatile MyReal2*) (mat_sh + blockDim.x);
    // shared memory space for the flag array
    volatile int*     flg_sh = (volatile int*    ) (lin_sh + blockDim.x);
    
    // make the flag array
    flg_sh[tid] = (tid % sgm_sz == 0) ? 1 : 0;
    __syncthreads();
    
    //--------------------------------------------------
    // Recurrence 1: b[i] = b[i] - a[i]*c[i-1]/b[i-1] --
    //   solved by scan with 2x2 matrix mult operator --
    //--------------------------------------------------
    // 1.a) first map
    const unsigned int beg_seg_ind = (gid / sgm_sz) * sgm_sz;
    REAL b0 = (gid < n) ? b[beg_seg_ind] : 1.0;
    mat_sh[tid] = (gid!=beg_seg_ind && gid < n) ?
                    MyReal4(b[gid], -a[gid]*c[gid-1], 1.0, 0.0) :
                    MyReal4(1.0,                 0.0, 0.0, 1.0) ;
    // 1.b) inplaceScanInc<MatMult2b2>(n,mats);
    __syncthreads();
    MyReal4 res4 = sgmScanIncBlock <MatMult2b2, MyReal4, int>(mat_sh, flg_sh, tid);
    // 1.c) second map
    if(gid < n) {
        uu[gid] = (res4.x*b0 + res4.y) / (res4.z*b0 + res4.w) ;
    }
    __syncthreads();

    // make the flag array
    flg_sh[tid] = (tid % sgm_sz == 0) ? 1 : 0;
    __syncthreads();

    //----------------------------------------------------
    // Recurrence 2: y[i] = y[i] - (a[i]/b[i-1])*y[i-1] --
    //   solved by scan with linear func comp operator  --
    //----------------------------------------------------
    // 2.a) first map
    REAL y0 = (gid < n) ? r[beg_seg_ind] : 1.0;
    lin_sh[tid] = (gid!=beg_seg_ind && gid < n) ?
                    MyReal2(r[gid], -a[gid]/uu[gid-1]) :
                    MyReal2(0.0,    1.0              ) ;
    // 2.b) inplaceScanInc<LinFunComp>(n,lfuns);
    __syncthreads();
    MyReal2 res2 = sgmScanIncBlock <LinFunComp, MyReal2, int>(lin_sh, flg_sh, tid);
    // 2.c) second map
    if(gid < n) {
        u[gid] = res2.x + y0*res2.y;
    }
    __syncthreads();

    // make the flag array
    flg_sh[tid] = (tid % sgm_sz == 0) ? 1 : 0;
    __syncthreads();
#if 1
    //----------------------------------------------------
    // Recurrence 3: backward recurrence solved via     --
    //             scan with linear func comp operator  --
    //----------------------------------------------------
    // 3.a) first map
    const unsigned int end_seg_ind = (beg_seg_ind + sgm_sz) - 1;
    const unsigned int k = (end_seg_ind - gid) + beg_seg_ind ;  
    REAL yn = u[end_seg_ind] / uu[end_seg_ind];
    lin_sh[tid] = (gid!=beg_seg_ind && gid < n) ?
                    MyReal2( u[k]/uu[k], -c[k]/uu[k] ) :
                    MyReal2( 0.0,        1.0         ) ;
    // 3.b) inplaceScanInc<LinFunComp>(n,lfuns);
    __syncthreads();
    MyReal2 res3 = sgmScanIncBlock <LinFunComp, MyReal2, int>(lin_sh, flg_sh, tid);
    __syncthreads();
    // 3.c) second map
    if(gid < n) {
        u[k] = res3.x + yn*res3.y;
    }
#endif
}

#endif //SCAN_KERS

