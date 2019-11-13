#ifndef CONSTANTS
#define CONSTANTS

#include <vector>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>

using namespace std;

#if (WITH_FLOATS==0)
    typedef double REAL;
#else
    typedef float  REAL;
#endif

class MyReal2 {
  public:
    REAL x; REAL y;

    __device__ __host__ inline MyReal2() {
        x = 0.0; y = 0.0; 
    }
    __device__ __host__ inline MyReal2(const REAL& a, const REAL& b) {
        x = a; y = b;
    }
    __device__ __host__ inline MyReal2(const MyReal2& i4) { 
        x = i4.x; y = i4.y;
    }
    volatile __device__ __host__ inline MyReal2& operator=(const MyReal2& i4) volatile {
        x = i4.x; y = i4.y;
        return *this;
    }
    __device__ __host__ inline MyReal2& operator=(const MyReal2& i4) {
        x = i4.x; y = i4.y;
        return *this;
    }
};

class MyReal4 {
  public:
    REAL x; REAL y; REAL z; REAL w;

    __device__ __host__ inline MyReal4() {
        x = 0.0; y = 0.0; z = 0.0; w = 0.0; 
    }
    __device__ __host__ inline MyReal4(const REAL& a, const REAL& b, const REAL& c, const REAL& d) {
        x = a; y = b; z = c; w = d; 
    }
    __device__ __host__ inline MyReal4(const MyReal4& i4) { 
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
    }
    volatile __device__ __host__ inline MyReal4& operator=(const MyReal4& i4) volatile {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
        return *this;
    }
    __device__ __host__ inline MyReal4& operator=(const MyReal4& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
        return *this;
    }
};

class LinFunComp {
  public:
    typedef MyReal2 OpTp;
    static __device__ __host__ inline
    MyReal2 apply(volatile MyReal2& a, volatile MyReal2& b) {
      return MyReal2( b.x + b.y*a.x, a.y*b.y );
    }

    static __device__ __host__ inline 
    MyReal2 identity() { 
      return MyReal2(0.0, 1.0);
    }
};

class MatMult2b2 {
  public:
    typedef MyReal4 OpTp;
    static __device__ __host__ inline
    MyReal4 apply(volatile MyReal4& a, volatile MyReal4& b) {
      REAL val = 1.0/(a.x*b.x);
      return MyReal4( (b.x*a.x + b.y*a.z)*val,
                      (b.x*a.y + b.y*a.w)*val,
                      (b.z*a.x + b.w*a.z)*val,
                      (b.z*a.y + b.w*a.w)*val );
    }

    static __device__ __host__ inline 
    MyReal4 identity() { 
      return MyReal4(1.0,  0.0, 0.0, 1.0);
    }
};

template<class OP>
void __device__ __host__ inplaceScanInc(const int n, typename OP::OpTp* inpres) {
  typename OP::OpTp acc = inpres[0];
  for(int i=1; i<n; i++) {
    acc = OP::apply(acc,inpres[i]);
    inpres[i] = acc;
  }
}

class DataCenter {
  public:
    REAL* myX        ;
    REAL* myY        ;
    REAL* myTimeline ;
    REAL* myDxx      ;
    REAL* myDyy      ;
    REAL* trMyDxx    ;
    REAL* trMyDyy    ;
    REAL* myVarX     ;
    REAL* myVarY     ;
    REAL* trMyVarX   ;
    REAL* aX         ;
    REAL* bX         ;
    REAL* cX         ;
    REAL* aY         ;
    REAL* bY         ;
    REAL* cY         ;
    REAL* myResult   ;
    REAL* trMyResult ;
    REAL* u          ;
    REAL* trU        ;
    REAL* v          ;
    REAL* trV        ;
    REAL* y          ;
    REAL* yy         ;
    REAL* dtInv      ;
    REAL* dl         ;
    REAL* du         ;

    DataCenter(unsigned outer, unsigned numX, unsigned numY, unsigned numT) {
      unsigned numZ = max(numX,numY);
      //Invariants for OUTER, never written to beyond initialization (read-only)
      //Therefore not expanded
      myX           = (REAL*) malloc(numX * sizeof(REAL)); 
      myY           = (REAL*) malloc(numY * sizeof(REAL));
      myTimeline    = (REAL*) malloc(numT * sizeof(REAL));
      myDxx         = (REAL*) malloc(numX * 4 * sizeof(REAL));
      myDyy         = (REAL*) malloc(numY * 4 * sizeof(REAL));
      trMyDxx       = (REAL*) malloc(4 * numX * sizeof(REAL));
      trMyDyy       = (REAL*) malloc(4 * numY * sizeof(REAL));
      //Expanded due to distribution over 2nd outer loop (numT)
      myVarX        = (REAL*) malloc(numT * numX * numY * sizeof(REAL));
      myVarY        = (REAL*) malloc(numT * numX * numY * sizeof(REAL));
      trMyVarX      = (REAL*) malloc(numT * numY * numX * sizeof(REAL));
      aX            = (REAL*) malloc(numT * numY * numX * sizeof(REAL));
      bX            = (REAL*) malloc(numT * numY * numX * sizeof(REAL));
      cX            = (REAL*) malloc(numT * numY * numX * sizeof(REAL));
      aY            = (REAL*) malloc(numT * numX * numY * sizeof(REAL));
      bY            = (REAL*) malloc(numT * numX * numY * sizeof(REAL));
      cY            = (REAL*) malloc(numT * numX * numY * sizeof(REAL));

      //Expanded after interchange and distribution of outer loops
      myResult      = (REAL*) malloc(outer * numX * numY * sizeof(REAL));
      trMyResult    = (REAL*) malloc(outer * numY * numX * sizeof(REAL));
      //Tridag initialization values; Expanded by numT to inspect each iteration
      u             = (REAL*) malloc(outer * numY * numX * sizeof(REAL));
      trU           = (REAL*) malloc(outer * numX * numY * sizeof(REAL));
      v             = (REAL*) malloc(outer * numX * numY * sizeof(REAL));
      trV           = (REAL*) malloc(outer * numY * numX * sizeof(REAL));
      //Tridag temporaries, thereby not expanded
      y             = (REAL*) malloc(outer * numX * numY * sizeof(REAL));
      yy            = (REAL*) malloc(numZ * sizeof(REAL));

      //Variable expanded to array
      dtInv         = (REAL*) malloc(numT * sizeof(REAL));
      dl            = (REAL*) malloc(numZ * sizeof(REAL));
      du            = (REAL*) malloc(numZ * sizeof(REAL));
    }

    void printArray(const char* name, unsigned entirePerLine, REAL* inarr, unsigned size) {
      for (unsigned i = 0; i < size; i++) {
        cout << name << "[" << i << "]=" << inarr[i] << ((i % entirePerLine == 0) ? "" : "      ");
        if (i % entirePerLine == 0) {
          cout << endl;
        }
      }
    }

    void printTwoArray(const char* name, REAL* fst, REAL* snd, unsigned start, unsigned end) {
      for (unsigned i = start; i <= end; i++) {
        cout << name << "[" << i << "]->" << fst[i] << "=?=" << snd[i] << endl;
      }
    }

    bool compareArrays(const char* name, REAL* fst, REAL* snd, unsigned size, bool terminate, bool printRanges) {
      bool isvalid = true;
      bool hasPrinted = false;
      unsigned count = 0;
      unsigned start = 0;
      unsigned end = 0;
      for (unsigned i = 0; i < size; i++) {
        if (fst[i] - snd[i] > 0.0001 || fst[i] - snd[i] < -0.0001 || fst[i] != fst[i] || snd[i] != snd[i]) {
          if (!hasPrinted || !terminate) {
            cout << "Mismatch at: " << name << "[" << i << "]: " << fst[i] << " =/= " << snd[i] << endl;
            hasPrinted = true;
          }
          count++;
          if (start == 0) {
            start = i;
            end = start;
          } else {
            end ++;
          }
          isvalid = false;
        }
        else if (start > 0)
        {
          if (printRanges) {
            cout << "Range mismatch: " << name << "[" << start << "-" << end <<  "]" << endl;
          }
          start = 0;
          end = 0;
        }
      }
      if (!isvalid) {
        cout << "in total " << count << " indicies" << endl;
      }
      return isvalid;
    }

    bool NaNExists(REAL* A, unsigned size) {
      for (unsigned i = 0; i < size; i ++) {
        if (A[i] != A[i]) {
          return true;
        }
      }
      return false;
    }

    void dispose() {
      free(myX       );
      free(myY       );
      free(myTimeline);
      free(myDxx     );
      free(myDyy     );
      free(trMyDxx   );
      free(trMyDyy   );
      free(myVarX    );
      free(myVarY    );
      free(trMyVarX  );
      free(aX        );
      free(bX        );
      free(cX        );
      free(aY        );
      free(bY        );
      free(cY        );
      free(myResult  );
      free(trMyResult);
      free(u         );
      free(trU       );
      free(v         );
      free(trV       );
      free(y         );
      free(yy        );
      free(dtInv     );
      free(dl        );
      free(du        );
    }
};

#endif // CONSTANTS
