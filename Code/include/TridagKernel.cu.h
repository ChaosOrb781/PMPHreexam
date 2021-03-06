#ifndef SCAN_KERS
#define SCAN_KERS

#include <cuda_runtime.h>
#include "Constants.h"

class MyReal2_ker {
  public:
    REAL x; REAL y;

    __device__ __host__ inline MyReal2_ker() {
        x = 0.0; y = 0.0; 
    }
    __device__ __host__ inline MyReal2_ker(const REAL& a, const REAL& b) {
        x = a; y = b;
    }
    __device__ __host__ inline MyReal2_ker(const MyReal2_ker& i4) { 
        x = i4.x; y = i4.y;
    }
    volatile __device__ __host__ inline MyReal2_ker& operator=(const MyReal2_ker& i4) volatile {
        x = i4.x; y = i4.y;
        return *this;
    }
    __device__ __host__ inline MyReal2_ker& operator=(const MyReal2_ker& i4) {
        x = i4.x; y = i4.y;
        return *this;
    }
};

class MyReal4_ker {
  public:
    REAL x; REAL y; REAL z; REAL w;

    __device__ __host__ inline MyReal4_ker() {
        x = 0.0; y = 0.0; z = 0.0; w = 0.0; 
    }
    __device__ __host__ inline MyReal4_ker(const REAL& a, const REAL& b, const REAL& c, const REAL& d) {
        x = a; y = b; z = c; w = d; 
    }
    __device__ __host__ inline MyReal4_ker(const MyReal4_ker& i4) { 
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
    }
    volatile __device__ __host__ inline MyReal4_ker& operator=(const MyReal4_ker& i4) volatile {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
        return *this;
    }
    __device__ __host__ inline MyReal4_ker& operator=(const MyReal4_ker& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
        return *this;
    }
};

class LinFunComp_ker {
  public:
    typedef MyReal2_ker BaseType;

    static __device__ __host__ inline
    MyReal2_ker apply(volatile MyReal2_ker& a, volatile MyReal2_ker& b) {
      return MyReal2_ker( b.x + b.y*a.x, a.y*b.y );
    }

    static __device__ __host__ inline 
    MyReal2_ker identity() { 
      return MyReal2_ker(0.0, 1.0);
    }
};

class MatMult2b2_ker {
  public:
    typedef MyReal4_ker BaseType;

    static __device__ __host__ inline
    MyReal4_ker apply(volatile MyReal4_ker& a, volatile MyReal4_ker& b) {
      REAL val = 1.0/(a.x*b.x);
      return MyReal4_ker( (b.x*a.x + b.y*a.z)*val,
                      (b.x*a.y + b.y*a.w)*val,
                      (b.z*a.x + b.w*a.z)*val,
                      (b.z*a.y + b.w*a.w)*val );
    }

    static __device__ __host__ inline 
    MyReal4_ker identity() { 
      return MyReal4_ker(1.0,  0.0, 0.0, 1.0);
    }
};

/***************************************/
/*** Scan Inclusive Helpers & Kernel ***/
/***************************************/
template<class OP, class T>
__device__ inline
T scanIncWarp( volatile T* ptr, const unsigned int idx ) {
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)  ptr[idx] = OP::apply(ptr[idx-1],  ptr[idx]); 
    if (lane >= 2)  ptr[idx] = OP::apply(ptr[idx-2],  ptr[idx]);
    if (lane >= 4)  ptr[idx] = OP::apply(ptr[idx-4],  ptr[idx]);
    if (lane >= 8)  ptr[idx] = OP::apply(ptr[idx-8],  ptr[idx]);
    if (lane >= 16) ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]);

    return const_cast<T&>(ptr[idx]);
}

template<class OP, class T>
__device__ inline
T scanIncBlock(volatile T* ptr, const unsigned int idx) {
    const unsigned int lane   = idx &  31;
    const unsigned int warpid = idx >> 5;

    T val = scanIncWarp<OP,T>(ptr,idx);
    __syncthreads();

    // place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and 
    //   max block size = 32^2 = 1024
    if (lane == 31) { ptr[warpid] = const_cast<T&>(ptr[idx]); } 
    __syncthreads();

    //
    if (warpid == 0) scanIncWarp<OP,T>(ptr, idx);
    __syncthreads();

    if (warpid > 0) {
        val = OP::apply(ptr[warpid-1], val);
    }

    return val;
}


/*************************************************/
/*************************************************/
/*** Segmented Inclusive Scan Helpers & Kernel ***/
/*************************************************/
/*************************************************/
template<class OP, class T, class F>
__device__ inline
T sgmScanIncWarp(volatile T* ptr, volatile F* flg, const unsigned int idx) {
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-1], ptr[idx]); }
        flg[idx] = flg[idx-1] | flg[idx];
    }
    if (lane >= 2)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-2], ptr[idx]); }
        flg[idx] = flg[idx-2] | flg[idx];
    }
    if (lane >= 4)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-4], ptr[idx]); }
        flg[idx] = flg[idx-4] | flg[idx];
    }
    if (lane >= 8)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-8], ptr[idx]); }
        flg[idx] = flg[idx-8] | flg[idx];
    }
    if (lane >= 16)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]); }
        flg[idx] = flg[idx-16] | flg[idx];
    }

    return const_cast<T&>(ptr[idx]);
}

template<class OP, class T, class F>
__device__ inline
T sgmScanIncBlock(volatile T* ptr, volatile F* flg, const unsigned int idx) {
    const unsigned int lane   = idx &  31;
    const unsigned int warpid = idx >> 5;
    const unsigned int warplst= (warpid<<5) + 31;

    // 1a: record whether this warp begins with an ``open'' segment.
    bool warp_is_open = (flg[(warpid << 5)] == 0);
    __syncthreads();

    // 1b: intra-warp segmented scan for each warp
    T val = sgmScanIncWarp<OP,T>(ptr,flg,idx);

    // 2a: the last value is the correct partial result
    T warp_total = const_cast<T&>(ptr[warplst]);
    
    // 2b: warp_flag is the OR-reduction of the flags 
    //     in a warp, and is computed indirectly from
    //     the mindex in hd[]
    bool warp_flag = flg[warplst]!=0 || !warp_is_open;
    bool will_accum= warp_is_open && (flg[idx] == 0);

    __syncthreads();

    // 2c: the last thread in a warp writes partial results
    //     in the first warp. Note that all fit in the first
    //     warp because warp = 32 and max block size is 32^2
    if (lane == 31) {
        ptr[warpid] = warp_total; //ptr[idx]; 
        flg[warpid] = warp_flag;
    }
    __syncthreads();

    // 
    if (warpid == 0) sgmScanIncWarp<OP,T>(ptr, flg, idx);
    __syncthreads();

    if (warpid > 0 && will_accum) {
        val = OP::apply(ptr[warpid-1], val);
    }
    return val;
}

template <class T, int TILE> 
__global__ void matTransposeTiledKer(T* A, T* B, int heightA, int widthA) {

  __shared__ T shtileTR[TILE][TILE+1];

  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;

  if( x < widthA && y < heightA )
      shtileTR[threadIdx.y][threadIdx.x] = A[y*widthA + x];

  __syncthreads();

  x = blockIdx.y * TILE + threadIdx.x; 
  y = blockIdx.x * TILE + threadIdx.y;

  if( x < heightA && y < widthA )
      B[y*heightA + x] = shtileTR[threadIdx.x][threadIdx.y];
}

/**
 * Matrix Transposition CPU Stub:
 * INPUT:
 *    `inp_d'  input matrix (already in device memory) 
 *    `height' number of rows    of input matrix `inp_d'
 *    `width'  number of columns of input matrix `inp_d'
 * OUTPUT:
 *    `out_d'  the transposed matrix with 
 *                 `width' rows and `height' columns!
 */
template<class T, int tile>
void transpose( float*             inp_d,  
                float*             out_d, 
                const unsigned int height, 
                const unsigned int width
) {
   // 1. setup block and grid parameters
   int  dimy = ceil( ((float)height)/tile ); 
   int  dimx = ceil( ((float) width)/tile );
   dim3 block(tile, tile, 1);
   dim3 grid (dimx, dimy, 1);
 
   //2. execute the kernel
   matTransposeTiledKer<float,tile> <<< grid, block >>>
                       (inp_d, out_d, height, width);    
   cudaDeviceSynchronize();
}

template<class T, int tile>
void transpose_nosync( T*             inp_d,  
                T*             out_d, 
                const unsigned int height, 
                const unsigned int width
) {
   // 1. setup block and grid parameters
   int  dimy = ceil( ((float)height)/tile ); 
   int  dimx = ceil( ((float) width)/tile );
   dim3 block(tile, tile, 1);
   dim3 grid (dimx, dimy, 1);
 
   //2. execute the kernel
   matTransposeTiledKer<T,tile> <<< grid, block >>>
                       (inp_d, out_d, height, width);
}

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
    volatile MyReal4_ker* mat_sh = (volatile MyReal4_ker*)sh_mem;
    // shared memory space for the linear-function composition SCAN
    volatile MyReal2_ker* lin_sh = (volatile MyReal2_ker*) (mat_sh + blockDim.x);
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
                    MyReal4_ker(b[gid], -a[gid]*c[gid-1], 1.0, 0.0) :
                    MyReal4_ker(1.0,                 0.0, 0.0, 1.0) ;
    // 1.b) inplaceScanInc<MatMult2b2_ker>(n,mats);
    __syncthreads();
    MyReal4_ker res4 = sgmScanIncBlock <MatMult2b2_ker, MyReal4_ker, int>(mat_sh, flg_sh, tid);
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
                    MyReal2_ker(r[gid], -a[gid]/uu[gid-1]) :
                    MyReal2_ker(0.0,    1.0              ) ;
    // 2.b) inplaceScanInc<LinFunComp_ker>(n,lfuns);
    __syncthreads();
    MyReal2_ker res2 = sgmScanIncBlock <LinFunComp_ker, MyReal2_ker, int>(lin_sh, flg_sh, tid);
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
                    MyReal2_ker( u[k]/uu[k], -c[k]/uu[k] ) :
                    MyReal2_ker( 0.0,        1.0         ) ;
    // 3.b) inplaceScanInc<LinFunComp_ker>(n,lfuns);
    __syncthreads();
    MyReal2_ker res3 = sgmScanIncBlock <LinFunComp_ker, MyReal2_ker, int>(lin_sh, flg_sh, tid);
    __syncthreads();
    // 3.c) second map
    if(gid < n) {
        u[k] = res3.x + yn*res3.y;
    }
#endif
}

#endif //SCAN_KERS

