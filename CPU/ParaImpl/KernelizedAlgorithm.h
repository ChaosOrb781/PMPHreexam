#ifndef KERNELIZED_ALGORITHM
#define KERNELIZED_ALGORITHM

#include "OriginalAlgorithm.h"

void initGrid_Alt(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const unsigned numX, const unsigned numY, const unsigned numT,
                REAL* myX, REAL* myY, REAL* myTimeline,
                uint& myXindex, uint& myYindex
) {
    for(unsigned i=0;i<numT;++i)
        myTimeline[i] = t*i/(numT-1);

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    myXindex = static_cast<unsigned>(s0/dx) % numX;

    for(unsigned i=0;i<numX;++i)
        myX[i] = i*dx - myXindex*dx + s0;

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    myYindex = static_cast<unsigned>(numY/2.0);

    for(unsigned i=0;i<numY;++i)
        myY[i] = i*dy - myYindex*dy + logAlpha;
}

//Have to flatten to use device vector!
void initOperator_Alt_Trans(  
    const uint& numZ, REAL* myZ, REAL* DzzT
) {
    for (int gidx = 0; gidx < numZ * 4; gidx ++) {
        int row = gidx / numZ;
        int col = gidx / 4;
        REAL dl = col > 0 ? myZ[col] - myZ[col-1] : 0.0;
        REAL du = col < numZ-1 ? myZ[col+1] - myZ[col] : 0.0;

        DzzT[gidx] = col > 0 && col < numZ-1 ?
                        (row == 0 ? 2.0/dl/(dl+du) :
                        (row == 1 ? -2.0*(1.0/dl + 1.0/du)/(dl+du) :
                        (row == 2 ? 2.0/du/(dl+du) :
                        0.0)))
                        : 0.0;
    }
}

void initOperator_Alt_Trans_para(  
    const uint& numZ, REAL* myZ, REAL* DzzT
) {
	//	standard case
#pragma omp parallel for schedule(static)
	for (int gidx = 0; gidx < numZ * 4; gidx ++) {
        int row = gidx / numZ;
        int col = gidx / 4;
        REAL dl = col > 0 ? myZ[col] - myZ[col-1] : 0.0;
        REAL du = col < numZ-1 ? myZ[col+1] - myZ[col] : 0.0;

        DzzT[gidx] = col > 0 && col < numZ-1 ?
                        (row == 0 ? 2.0/dl/(dl+du) :
                        (row == 1 ? -2.0*(1.0/dl + 1.0/du)/(dl+du) :
                        (row == 2 ? 2.0/du/(dl+du) :
                        0.0)))
                        : 0.0;
    }
}

void updateParams_Alt(const REAL alpha, const REAL beta, const REAL nu,
    const uint numX, const uint numY, const uint numT, 
    REAL* myX, REAL* myY, REAL* myTimeline,
    REAL* myVarX, REAL* myVarY)
{
    for(uint gidx = 0; gidx < numT * numX * numY; gidx++) {
        int i = gidx / (numX * numY);
        int j = gidx % (numX * numY);
        int k = j / numY;
        myVarX[gidx] = exp(2.0*(  beta*log(myX[j])   
                                    + myY[k]             
                                    - 0.5*nu*nu*myTimeline[i] )
                                );
        myVarY[gidx] = exp(2.0*(  alpha*log(myX[j])   
                                    + myY[k]             
                                    - 0.5*nu*nu*myTimeline[i] )
                                ); // nu*nu
    }
}

void updateParams_Alt_Trans_para(const REAL alpha, const REAL beta, const REAL nu,
    const uint numX, const uint numY, const uint numT, 
    REAL* myX, REAL* myY, REAL* myTimeline,
    REAL* myVarX, REAL* myVarY)
{
#pragma omp parallel for schedule(static)
    for(uint gidx = 0; gidx < numT * numX * numY; gidx++) {
        int i = gidx / (numX * numY);
        int j = gidx % (numX * numY);
        int k = j / numY;
        myVarX[gidx] = exp(2.0*(  beta*log(myX[j])   
                                    + myY[k]             
                                    - 0.5*nu*nu*myTimeline[i] )
                                );
        myVarY[gidx] = exp(2.0*(  alpha*log(myX[j])   
                                    + myY[k]             
                                    - 0.5*nu*nu*myTimeline[i] )
                                ); // nu*nu
    }
}

void setPayoff_Alt(REAL* myX, const uint outer,
    const uint numX, const uint numY,
    REAL* myResult)
{
    for(uint gidx = 0; gidx < outer * numX * numY; gidx++) {
        int o = gidx / (numX * numY);
        int j = gidx % (numX * numY);
        myResult[gidx] = max(myX[j]-0.001*(REAL)o, (REAL)0.0);
    }
}

void setPayoff_Alt_para(REAL* myX, const uint outer,
    const uint numX, const uint numY,
    REAL* myResult)
{
#pragma omp parallel for schedule(static)
    for(uint gidx = 0; gidx < outer * numX * numY; gidx++) {
        int o = gidx / (numX * numY);
        int j = gidx % (numX * numY);
        myResult[gidx] = max(myX[j]-0.001*(REAL)o, (REAL)0.0);
    }
}

void rollback_Alt(const uint outer, const uint numT, 
    const uint numX, const uint numY, 
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
    for (int t = 0; t <= numT - 2; t++) {
        for (int gidx = 0; gidx < outer; gidx++) {
            uint numZ = max(numX,numY);

            uint i, j;

            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);

            //REAL* u(numY * numX);   // [numY][numX]
            //REAL* v(numX * numY);   // [numX][numY]
            //REAL* a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
            //REAL* yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

            cout << "explicit x, t: " << t << " o: " << gidx << endl;
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

            cout << "explicit y, t: " << t << " o: " << gidx << endl;
            //	explicit y
            for(j=0;j<numY;j++)
            {
                for(i=0;i<numX;i++) {
                    v[((gidx * numX) + i) * numY + j] = 0.0;

                    if(j > 0) {
                        v[((gidx * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                        * myDyy[j * 4 + 0] )
                                        * myResult[((gidx * numX) + (i+1)) * numY + j - 1];
                    }
                    v[((gidx * numX) + i) * numY + j]  += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                     * myDyy[j * 4 + 1] )
                                     * myResult[((gidx * numX) + (i+1)) * numY + j];
                    if(j < numY-1) {
                        v[((gidx * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                        * myDyy[j * 4 + 2] )
                                        * myResult[((gidx * numX) + (i+1)) * numY + j + 1];
                    }
                    u[((gidx * numY) + j) * numX + i] += v[((gidx * numX) + i) * numY + j]; 
                }
            }

            cout << "implicit x, t: " << t << " o: " << gidx << endl;
            //	implicit x
            for(j=0;j<numY;j++) {
                for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
                    a[(gidx * numZ) + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 0]);
                    b[(gidx * numZ) + i] = dtInv - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 1]);
                    c[(gidx * numZ) + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 2]);
                }
                // here yy should have size [numX]
                tridagPar(&a[gidx * numZ],&b[gidx * numZ],&c[gidx * numZ],&u[((gidx * numY) + j)],numX,&u[((gidx * numY) + j)],&yy[gidx * numZ]);
            }

            cout << "implicit y, t: " << t << " o: " << gidx << endl;
            //	implicit y
            for(i=0;i<numX;i++) { 
                for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
                    a[(gidx * numZ) + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 0]);
                    b[(gidx * numZ) + j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 1]);
                    c[(gidx * numZ) + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 2]);
                }

                for(j=0;j<numY;j++)
                    y[gidx * numZ + j] = dtInv*u[((gidx * numY) + j) * numX + i] - 0.5*v[((gidx * numX) + i) * numY + j];

                // here yy should have size [numY]
                tridagPar(&a[gidx * numZ],&b[gidx * numZ],&c[gidx * numZ],&y[gidx * numZ],numY,&myResult[(gidx * numX + i) * numY],&yy[gidx * numZ]);
            }
        }
    }
}

void rollback_Alt_para(const uint outer, const uint numT, 
    const uint numX, const uint numY, 
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
    for (int t = 0; t <= numT - 2; t++) {
        for (int gidx = 0; gidx < outer; gidx++) {
            uint numZ = max(numX,numY);

            uint i, j;

            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);

            //REAL* u(numY * numX);   // [numY][numX]
            //REAL* v(numX * numY);   // [numX][numY]
            //REAL* a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
            //REAL* yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

            cout << "explicit x, t: " << t << " o: " << gidx << endl;
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

            cout << "explicit y, t: " << t << " o: " << gidx << endl;
            //	explicit y
            for(j=0;j<numY;j++)
            {
                for(i=0;i<numX;i++) {
                    v[i * numY + j] = 0.0;

                    if(j > 0) {
                        v[i * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                        * myDyy[j * 4 + 0] )
                                        * myResult[((gidx * numX) + (i+1)) * numY + j - 1];
                    }
                    v[i * numY + j]  += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                     * myDyy[j * 4 + 1] )
                                     * myResult[((gidx * numX) + (i+1)) * numY + j];
                    if(j < numY-1) {
                        v[i * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                        * myDyy[j * 4 + 2] )
                                        * myResult[((gidx * numX) + (i+1)) * numY + j + 1];
                    }
                    u[((gidx * numY) + j) * numX + i] += v[i * numY + j]; 
                }
            }

            cout << "implicit x, t: " << t << " o: " << gidx << endl;
            //	implicit x
            for(j=0;j<numY;j++) {
                for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
                    a[i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 0]);
                    b[i] = dtInv - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 1]);
                    c[i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 2]);
                }
                // here yy should have size [numX]
                tridagPar(a,b,c,&u[((gidx * numY) + j)],numX,&u[((gidx * numY) + j)],yy);
            }

            cout << "implicit y, t: " << t << " o: " << gidx << endl;
            //	implicit y
            for(i=0;i<numX;i++) { 
                for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
                    a[j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 0]);
                    b[j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 1]);
                    c[j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 2]);
                }

                for(j=0;j<numY;j++)
                    y[j] = dtInv*u[((gidx * numY) + j) * numX + i] - 0.5*v[i * numY + j];

                // here yy should have size [numY]
                tridagPar(a,b,c,y,numY,&myResult[(gidx * numX + i) * numY],yy);
            }
        }
    }
}

template<class T>
void matTranspose(T* A, T* trA, int rowsA, int colsA) {
    for(int i = 0; i < rowsA; i++) {
        for(int j = 0; j < colsA; j++) {
            trA[j*rowsA + i] = A[i*colsA + j];
        }
    }
}

template<class T>
void matTransposePlane(T* A, T* trA, int planes, int rowsA, int colsA) {
    for (unsigned i = 0; i < planes; i++) {
        matTranspose<T>(&A[i * rowsA * colsA], &trA[i * rowsA * colsA], rowsA, colsA);
    }
}

int   run_SimpleKernelized(  
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
) {
	REAL* myX       = (REAL*)malloc (sizeof(REAL) * numX); // [numX]
    REAL* myY       = (REAL*)malloc (sizeof(REAL) * numY); // [numY]
    REAL* myTimeline= (REAL*)malloc (sizeof(REAL) * numT); // [numT]
    REAL* myDxx     = (REAL*)malloc (sizeof(REAL) * numX * 4); // [numX][4]
    REAL* myDyy     = (REAL*)malloc (sizeof(REAL) * numY * 4); // [numY][4]
    REAL* myDxxT    = (REAL*)malloc (sizeof(REAL) * 4 * numX); // [4][numX]
    REAL* myDyyT    = (REAL*)malloc (sizeof(REAL) * 4 * numY); // [4][numY]
    REAL* myResult  = (REAL*)malloc (sizeof(REAL) * outer * numX * numY); // [outer][numX][numY]
    REAL* myVarX    = (REAL*)malloc (sizeof(REAL) * numT * numX * numY); // [numT][numX][numY]
    REAL* myVarY    = (REAL*)malloc (sizeof(REAL) * numT * numX * numY); // [numT][numX][numY]
    REAL* myVarXT   = (REAL*)malloc (sizeof(REAL) * numT * numY * numX); // [numT][numY][numX]

    uint numZ = max(numX, numY);
    REAL* u         = (REAL*)malloc (sizeof(REAL) * outer * numY * numX); // [outer][numY][numX]
    REAL* v         = (REAL*)malloc (sizeof(REAL) * outer * numX * numY); // [outer][numX][numY]
    REAL* a         = (REAL*)malloc (sizeof(REAL) * outer * numZ); // [outer][numZ]
    REAL* b         = (REAL*)malloc (sizeof(REAL) * outer * numZ); // [outer][numZ]
    REAL* c         = (REAL*)malloc (sizeof(REAL) * outer * numZ); // [outer][numZ]
    REAL* y         = (REAL*)malloc (sizeof(REAL) * outer * numZ); // [outer][numZ]
    REAL* yy        = (REAL*)malloc (sizeof(REAL) * outer * numZ); // [outer][numZ]


    uint myXindex = 0;
    uint myYindex = 0;

    cout << "Test1" << endl;
	initGrid_Alt(s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);
    cout << "Test2" << endl;
    initOperator_Alt_Trans(numX, myX, myDxxT);
    matTranspose<REAL>(myDxxT, myDxx, 4, numX);
    cout << "Test3" << endl;
    initOperator_Alt_Trans(numY, myY, myDyyT);
    matTranspose<REAL>(myDyyT, myDyy, 4, numY);
    cout << "Test4" << endl;
    setPayoff_Alt(myX, outer, numX, numY, myResult);

    cout << "Test5" << endl;
    updateParams_Alt(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
    cout << "Test6" << endl;
	rollback_Alt(outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, u, v, a, b, c, y, yy, myResult);
	
    cout << "Test7" << endl;
	for(uint i = 0; i < outer; i++) {
        res[i] = myResult[((i * numX) + myXindex) * numY + myYindex];
    }
    return 1;
}
#endif