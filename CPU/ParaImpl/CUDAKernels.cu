#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include <cuda_runtime.h>
#include "Constants.h"

struct MyReal4_ker {
    REAL x;
    REAL y;
    REAL z;
    REAL w;
    
    // constructors
    inline MyReal4_ker() { x = y = z = w = 0.0; }
    inline MyReal4_ker(const REAL a, const REAL b, const REAL c, const REAL d) {
        x = a; y = b; z = c; w = d;
    }
    // copy constructor
    inline MyReal4_ker(const MyReal4_ker& i4) { 
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
    }
    // assignment operator
    inline MyReal4_ker& operator=(const MyReal4_ker& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
        return *this;
    }
};

struct MatMult2b2_ker {
  typedef MyReal4_ker OpTp;
  static MyReal4_ker apply(const MyReal4_ker a, const MyReal4_ker b) {
    REAL val = 1.0/(a.x*b.x);
    return MyReal4_ker( (b.x*a.x + b.y*a.z)*val,
                    (b.x*a.y + b.y*a.w)*val,
                    (b.z*a.x + b.w*a.z)*val,
                    (b.z*a.y + b.w*a.w)*val );
  }
};

struct MyReal2_ker {
    REAL x;
    REAL y;
    // constructors
    inline MyReal2_ker() { x = y = 0.0; }
    inline MyReal2_ker(const REAL a, const REAL b) {
        x = a; y = b;
    }
    // copy constructor
    inline MyReal2_ker(const MyReal2_ker& i4) { 
        x = i4.x; y = i4.y; 
    }
    // assignment operator
    inline MyReal2_ker& operator=(const MyReal2_ker& i4) {
        x = i4.x; y = i4.y; 
        return *this;
    }
};

struct LinFunComp_ker {
  typedef MyReal2_ker OpTp;
  static MyReal2_ker apply(const MyReal2_ker a, const MyReal2_ker b) {
    return MyReal2_ker( b.x + b.y*a.x, a.y*b.y );
  }
};

template<class OP>
void inplaceScanInc_ker(const int n, vector<typename OP::OpTp>& inpres) {
  typename OP::OpTp acc = inpres[0];
  for(int i=1; i<n; i++) {
    acc = OP::apply(acc,inpres[i]);
    inpres[i] = acc;    
  }
}

__device__ void tridagPar_seq(
    const REAL*   a,   // size [n]
    const int             a_start,
    const REAL*   b,   // size [n]
    const int             b_start,
    const REAL*   c,   // size [n]
    const int             c_start,
    const REAL*   r,   // size [n]
    const int             r_start,
    const int             n,
          REAL*   u,   // size [n]
    const int             u_start,
          REAL*   uu,   // size [n] temporary
    const int             uu_start
) {
    //int i, offset;

    //vector<MyReal4> scanres(n); // supposed to also be in shared memory and to reuse the space of mats
    //--------------------------------------------------
    // Recurrence 1: b[i] = b[i] - a[i]*c[i-1]/b[i-1] --
    //   solved by scan with 2x2 matrix mult operator --
    //--------------------------------------------------
    vector<MyReal4_ker> mats(n);    // supposed to be in shared memory!
    REAL b0 = b[b_start + 0];
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        if (i==0) { mats[i].x = 1.0;  mats[i].y = 0.0;          mats[i].z = 0.0; mats[i].w = 1.0; }
        else      { mats[i].x = b[b_start + i]; mats[i].y = -a[a_start + i]*c[c_start + i-1]; mats[i].z = 1.0; mats[i].w = 0.0; }
    }
    inplaceScanInc_ker<MatMult2b2_ker>(n,mats);
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        uu[uu_start + i] = (mats[i].x*b0 + mats[i].y) / (mats[i].z*b0 + mats[i].w);
    }
    // b -> uu
    //----------------------------------------------------
    // Recurrence 2: y[i] = y[i] - (a[i]/b[i-1])*y[i-1] --
    //   solved by scan with linear func comp operator  --
    //----------------------------------------------------
    vector<MyReal2_ker> lfuns(n);
    REAL y0 = r[r_start + 0];
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        if (i==0) { lfuns[0].x = 0.0;  lfuns[0].y = 1.0;           }
        else      { lfuns[i].x = r[r_start + i]; lfuns[i].y = -a[a_start + i]/uu[uu_start + i-1]; }
    }
    inplaceScanInc_ker<LinFunComp_ker>(n,lfuns);
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        u[u_start + i] = lfuns[i].x + y0*lfuns[i].y;
    }
    // y -> u

    //----------------------------------------------------
    // Recurrence 3: backward recurrence solved via     --
    //             scan with linear func comp operator  --
    //----------------------------------------------------
    REAL yn = u[u_start + n-1]/uu[uu_start + n-1];
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        int k = n - i - 1;
        if (i==0) { lfuns[0].x = 0.0;  lfuns[0].y = 1.0;           }
        else      { lfuns[i].x = u[u_start + k]/uu[uu_start + k]; lfuns[i].y = -c[c_start + k]/uu[uu_start + k]; }
    }
    inplaceScanInc_ker<LinFunComp_ker>(n,lfuns);
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        u[u_start + n-i-1] = lfuns[i].x + yn*lfuns[i].y;
    }
}

///numT iterations
__global__ void InitMyTimeline(
        const uint   numT,
        const REAL   t,
        REAL* myTimeline
    ) {
    /*
    for(unsigned i=0;i<numT;++i)
        myTimeline[i] = t*i/(numT-1);
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numT)
        myTimeline[gidx] = t*gidx/(numT-1);
}

///numX iterations
__global__ void InitMyX(
        const uint numX,
        const uint myXindex,
        const REAL s0,
        const REAL dx,
        REAL* myX
    ) {
    /*
    for(unsigned i=0;i<numX;++i)
        myX[i] = i*dx - myXindex*dx + s0;
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numX)
        myX[gidx] = gidx*dx - myXindex*dx + s0;
}

///numY iterations
__global__ void InitMyY(
        const uint numY,
        const uint myYindex,
        const REAL logAlpha,
        const REAL dy,
        REAL* myY
    ) {
    /*
    for(unsigned i=0;i<numY;++i)
        myY[i] = i*dy - myYindex*dy + logAlpha;
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numY)
        myY[gidx] = gidx*dy - myYindex*dy + logAlpha;
}

///numZ iterations
__global__ void InitMyDzz(
        const uint numZ,
        REAL* myZ,
        REAL* Dzz
    ) {
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numZ * 4) {
        uint row = gidx % 4;
        uint col = gidx / 4;
        REAL dl, du;
        dl = (col == 0) ? 0.0 : myZ[col] - myZ[col - 1];
        du = (col == numZ - 1) ? 0.0 : myZ[col + 1] - myZ[col];
        Dzz[gidx] = col > 0 && col < numZ-1 ?
                    (row == 0 ? 2.0/dl/(dl+du) :
                    (row == 1 ? -2.0*(1.0/dl + 1.0/du)/(dl+du) :
                    (row == 2 ? 2.0/du/(dl+du) :
                    0.0)))
                    : 0.0;
    }
}

__global__ void InitMyResult(
        const uint outer,
        const uint numX,
        const uint numY,
        REAL* myX,
        REAL* myResult
    ) {
    /*
    for(uint gidx = 0; gidx < outer * numX * numY; gidx++) {
        int o = gidx / (numX * numY);
        int plane_remain = gidx % (numX * numY);
        int i = plane_remain / numY;
        //int j = plane_remain % numY
        myResult[gidx] = std::max(myX[i]-0.001*(REAL)o, (REAL)0.0);
    }
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < outer * numX * numY) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        //int j = plane_remain % numY
        REAL a = myX[i]-0.001*(REAL)o;
        myResult[gidx] = a > 0.0 ? a : (REAL)0.0;
    }
}

__global__ void InitParams(
        const uint numT,
        const uint numX,
        const uint numY,
        const REAL alpha,
        const REAL beta,
        const REAL nu,
        REAL* myX,
        REAL* myY,
        REAL* myTimeline,
        REAL* myVarX,
        REAL* myVarY
    ) {
    /*
    for(uint gidx = 0; gidx < numT * numX * numY; gidx++) {
        int t = gidx / (numX * numY);
        int plane_remain = gidx % (numX * numY);
        int i = plane_remain / numY;
        int j = plane_remain % numY;
        myVarX[gidx] = exp(2.0*(  beta*log(myX[i])   
                                    + myY[j]             
                                    - 0.5*nu*nu*myTimeline[t] )
                                );
        myVarY[gidx] = exp(2.0*(  alpha*log(myX[i])   
                                    + myY[j]             
                                    - 0.5*nu*nu*myTimeline[t] )
                                ); // nu*nu
    }
    */
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < numT * numX * numY) {
        int t = gidx / (numX * numY);
        int plane_remain = gidx % (numX * numY);
        int i = plane_remain / numY;
        int j = plane_remain % numY;
        myVarX[gidx] = exp(2.0*(  beta*log(myX[i])   
                                    + myY[j]             
                                    - 0.5*nu*nu*myTimeline[t] )
                                );
        myVarY[gidx] = exp(2.0*(  alpha*log(myX[i])   
                                    + myY[j]             
                                    - 0.5*nu*nu*myTimeline[t] )
                                ); // nu*nu
    }
}

__global__ void Rollback(
    uint t, 
    const uint outer, 
    const uint numT, 
    const uint numX, 
    const uint numY, 
    REAL* myTimeline, 
    REAL* myDxx,
    REAL* myDyy,
    REAL* myVarX,
    REAL* myVarY,
    REAL* u,
    REAL* v,
    REAL* a,
    REAL* b,
    REAL* c,
    REAL* y,
    REAL* yy,
    REAL* myResult
) {
    uint gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx < outer) {
        uint numZ = max(numX,numY);

        uint i, j;

        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);

        //vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
        //vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
        //vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
        //vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

        //cout << "explicit x, t: " << t << " o: " << gidx << endl;
        //	explicit x
        for(i=0;i<numX;i++) {
            for(j=0;j<numY;j++) {
                u[((gidx * numY) + j) * numX + i] = dtInv*myResult[((gidx * numX) + i) * numY + j];

                if(i > 0) { 
                    u[((gidx * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                    * myDxx[i * 4 + 0] ) 
                                    * myResult[((gidx * numX) + (i-1)) * numY + j];
                }
                u[((gidx * numY) + j) * numX + i]  +=  0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                * myDxx[i * 4 + 1] )
                                * myResult[((gidx * numX) + i) * numY + j];
                if(i < numX-1) {
                    u[((gidx * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                    * myDxx[i * 4 + 2] )
                                    * myResult[((gidx * numX) + (i+1)) * numY + j];
                }
            }
        }

        //cout << "explicit y, t: " << t << " o: " << gidx << endl;
        //	explicit y
        for(j=0;j<numY;j++)
        {
            for(i=0;i<numX;i++) {
                v[((gidx * numX) + i) * numY + j] = 0.0;

                if(j > 0) {
                    v[((gidx * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                    * myDyy[j * 4 + 0] )
                                    * myResult[((gidx * numX) + i) * numY + j - 1];
                }
                v[((gidx * numX) + i) * numY + j]  += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                    * myDyy[j * 4 + 1] )
                                    * myResult[((gidx * numX) + i) * numY + j];
                if(j < numY-1) {
                    v[((gidx * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                    * myDyy[j * 4 + 2] )
                                    * myResult[((gidx * numX) + i) * numY + j + 1];
                }
                u[((gidx * numY) + j) * numX + i] += v[((gidx * numX) + i) * numY + j]; 
            }
        }

        //cout << "implicit x, t: " << t << " o: " << gidx << endl;
        //	implicit x
        for(j=0;j<numY;j++) {
            for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
                a[(gidx * numZ) + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 0]);
                b[(gidx * numZ) + i] = dtInv - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 1]);
                c[(gidx * numZ) + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 2]);
            }
            // here yy should have size [numX]
            tridagPar_seq(a,(gidx * numZ),b,(gidx * numZ),c,(gidx * numZ),u,((gidx * numY) + j) * numX,numX,u,((gidx * numY) + j) * numX,yy,(gidx * numZ));
        }

        //cout << "implicit y, t: " << t << " o: " << gidx << endl;
        //	implicit y
        for(i=0;i<numX;i++) { 
            for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
                a[(gidx * numZ) + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 0]);
                b[(gidx * numZ) + j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 1]);
                c[(gidx * numZ) + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 2]);
            }

            for(j=0;j<numY;j++)
                y[(gidx * numZ) + j] = dtInv*u[((gidx * numY) + j) * numX + i] - 0.5*v[((gidx * numX) + i) * numY + j];

            // here yy should have size [numY]
            tridagPar_seq(a,(gidx * numZ),b,(gidx * numZ),c,(gidx * numZ),y,(gidx * numZ),numY,myResult, (gidx * numX + i) * numY,yy,(gidx * numZ));
        }
    }
}

#endif