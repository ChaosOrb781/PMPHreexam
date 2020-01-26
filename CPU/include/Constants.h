#ifndef CONSTANTS
#define CONSTANTS

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

using namespace std;

typedef unsigned int uint;

#if (WITH_FLOATS==0)
    typedef double REAL;
#else
    typedef float  REAL;
#endif

struct PrivGlobs {

    //	grid
    vector<REAL>        myX;        // [numX]
    vector<REAL>        myY;        // [numY]
    vector<REAL>        myTimeline; // [numT]
    unsigned            myXindex;  
    unsigned            myYindex;

    //	variable
    vector<vector<REAL> > myResult; // [numX][numY]

    //	coeffs
    vector<vector<REAL> >   myVarX; // [numX][numY]
    vector<vector<REAL> >   myVarY; // [numX][numY]

    //	operators
    vector<vector<REAL> >   myDxx;  // [numX][4]
    vector<vector<REAL> >   myDyy;  // [numY][4]

    PrivGlobs( ) {
        //printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        //exit(0);
    }

    PrivGlobs(  const unsigned int& numX,
                const unsigned int& numY,
                const unsigned int& numT ) {
        this->  myX.resize(numX);
        this->myDxx.resize(numX);
        for(int k=0; k<numX; k++) {
            this->myDxx[k].resize(4);
        }

        this->  myY.resize(numY);
        this->myDyy.resize(numY);
        for(int k=0; k<numY; k++) {
            this->myDyy[k].resize(4);
        }

        this->myTimeline.resize(numT);

        this->  myVarX.resize(numX);
        this->  myVarY.resize(numX);
        this->myResult.resize(numX);
        for(unsigned i=0;i<numX;++i) {
            this->  myVarX[i].resize(numY);
            this->  myVarY[i].resize(numY);
            this->myResult[i].resize(numY);
        }

    }

    void Initialize(const unsigned int& numX,
      const unsigned int& numY,
      const unsigned int& numT) {
      this->myX.resize(numX);
      this->myDxx.resize(numX);
      for (int k = 0; k < numX; k++) {
        this->myDxx[k].resize(4);
      }

      this->myY.resize(numY);
      this->myDyy.resize(numY);
      for (int k = 0; k < numY; k++) {
        this->myDyy[k].resize(4);
      }

      this->myTimeline.resize(numT);

      this->myVarX.resize(numX);
      this->myVarY.resize(numX);
      this->myResult.resize(numX);
      for (unsigned i = 0; i < numX; ++i) {
        this->myVarX[i].resize(numY);
        this->myVarY[i].resize(numY);
        this->myResult[i].resize(numY);
      }

    }
} __attribute__ ((aligned (128)));

class ReturnStat {
    public: 
        unsigned long int time;
        uint numthreads;

        ReturnStat(unsigned long int timeTaken, uint numberOfThreads) {
            time = timeTaken;
            numthreads = numberOfThreads;
        }

        double Speedup(ReturnStat* other) {
            return ((double) other->time) / ((double) this->time);
        }
};

void matTranspose(vector<REAL> A, vector<REAL> trA, uint planeIndex, int rowsA, int colsA) {
    for(int i = 0; i < rowsA; i++) {
        for(int j = 0; j < colsA; j++) {
            trA[planeIndex + j*rowsA + i] = A[planeIndex + i*colsA + j];
        }
    }
}

void matTransposePlane(vector<REAL> A, vector<REAL> trA, int planes, int rowsA, int colsA) {
    for (unsigned i = 0; i < planes; i++) {
        matTranspose(A, trA, i * rowsA * colsA, rowsA, colsA);
    }
}

#endif // CONSTANTS
