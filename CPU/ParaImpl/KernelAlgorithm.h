#ifndef KERNEL_ALGORITHM
#define KERNEL_ALGORITHM
#include "CUDAKernels.cu"
#include "InterchangedAlgorithm.h"

#define TEST_INIT_CORRECTNESS true

void initGrid_Kernel(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const uint numX, const uint numY, const uint numT,
                dvec<REAL>& myX, dvec<REAL>& myY, dvec<REAL>& myTimeline,
                uint& myXindex, uint& myYindex
) {
    REAL* myTimeline_p = raw_pointer_cast(&myTimeline[0]);
    InitMyTimeline(numT, t, myTimeline_p);

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    myXindex = static_cast<unsigned>(s0/dx) % numX;

    REAL* myX_p = raw_pointer_cast(&myX[0]);
    InitMyX(numX, myXindex, s0, dx, myX_p);

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    myYindex = static_cast<unsigned>(numY/2.0);

    REAL* myY_p = raw_pointer_cast(&myY[0]);
    InitMyY(numY, myYindex, logAlpha, dy, myY_p);
}

void initOperator_Kernel(  const uint numZ, dvec<REAL>& myZ, 
                        dvec<REAL>& Dzz
) {
    REAL* myZ_p = raw_pointer_cast(&myZ[0]);
    REAL* Dzz_p = raw_pointer_cast(&Dzz[0]);
    InitMyDzz(numZ, myZ_p, Dzz_p);
}

void updateParams_Kernel(const REAL alpha, const REAL beta, const REAL nu,
    const uint numX, const uint numY, const uint numT, 
    dvec<REAL>& myX, dvec<REAL>& myY, dvec<REAL>& myTimeline,
    dvec<REAL>& myVarX, dvec<REAL>& myVarY)
{
    REAL* myX_p = raw_pointer_cast(&myX[0]);
    REAL* myY_p = raw_pointer_cast(&myY[0]);
    REAL* myTimeline_p = raw_pointer_cast(&myTimeline[0]);
    REAL* myVarX_p = raw_pointer_cast(&myVarX[0]);
    REAL* myVarY_p = raw_pointer_cast(&myVarY[0]);
    InitParams(numT, numX, numY, alpha, beta, nu, myX_p, myY_p, myTimeline_p, myVarX_p, myVarY_p);
}

void setPayoff_Kernel(dvec<REAL>& myX, const uint outer,
    const uint numX, const uint numY,
    dvec<REAL>& myResult)
{
    for(uint gidx = 0; gidx < outer * numX * numY; gidx++) {
        int o = gidx / (numX * numY);
        int plane_remain = gidx % (numX * numY);
        int i = plane_remain / numY;
        //int j = plane_remain % numY
        myResult[gidx] = std::max(myX[i]-0.001*(REAL)o, (REAL)0.0);
    }
}

void rollback_Kernel(const uint outer, const uint numT, 
    const uint numX, const uint numY, 
    dvec<REAL>& myTimeline, 
    dvec<REAL>& myDxx,
    dvec<REAL>& myDyy,
    dvec<REAL>& myVarX,
    dvec<REAL>& myVarY,
    dvec<REAL>& u,
    dvec<REAL>& v,
    dvec<REAL>& a,
    dvec<REAL>& b,
    dvec<REAL>& c,
    dvec<REAL>& y,
    dvec<REAL>& yy,
    dvec<REAL>& myResult
) {
    for (int t = 0; t <= numT - 2; t++) {
        for (int gidx = 0; gidx < outer; gidx++) {
            uint numZ = std::max(numX,numY);

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
                tridagPar(a,(gidx * numZ),b,(gidx * numZ),c,(gidx * numZ),u,((gidx * numY) + j) * numX,numX,u,((gidx * numY) + j) * numX,yy,(gidx * numZ));
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
                tridagPar(a,(gidx * numZ),b,(gidx * numZ),c,(gidx * numZ),y,(gidx * numZ),numY,myResult, (gidx * numX + i) * numY,yy,(gidx * numZ));
            }
        }
    }
}

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
) {
    int procs = 0;

	dvec<REAL> myX(numX);       // [numX]
    dvec<REAL> myY(numY);       // [numY]
    dvec<REAL> myTimeline(numT);// [numT]
    dvec<REAL> myDxx(numX * 4);     // [numX][4]
    dvec<REAL> myDyy(numY * 4);     // [numY][4]
    dvec<REAL> myDxxT(4 * numX);       // [4][numX]
    dvec<REAL> myDyyT(4 * numY);       // [4][numY]
    dvec<REAL> myResult(outer * numX * numY); // [outer][numX][numY]
    dvec<REAL> myVarX(numT * numX * numY);    // [numT][numX][numY]
    dvec<REAL> myVarY(numT * numX * numY);    // [numT][numX][numY]
    dvec<REAL> myVarXT(numT * numY * numX);    // [numT][numY][numX]

#if TEST_INIT_CORRECTNESS
    vector<REAL> myResultCopy(outer * numX * numY);
#endif

    uint numZ = std::max(numX, numY);
    dvec<REAL> u(outer * numY * numX);
    dvec<REAL> v(outer * numX * numY);
    dvec<REAL> a(outer * numZ);
    dvec<REAL> b(outer * numZ);
    dvec<REAL> c(outer * numZ);
    dvec<REAL> y(outer * numZ);
    dvec<REAL> yy(outer * numZ);

    uint myXindex = 0;
    uint myYindex = 0;

    //cout << "Test1" << endl;
	initGrid_Kernel(s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);

    //cout << "Test2" << endl;
    initOperator_Kernel(numX, myX, myDxx);

    //cout << "Test3" << endl;
    initOperator_Kernel(numY, myY, myDyy);

    //cout << "Test4" << endl;
    setPayoff_Kernel(myX, outer, numX, numY, myResult);
#if TEST_INIT_CORRECTNESS
    for (int o = 0; o < outer; o ++) {
        for (int i = 0; i < numX; i ++) {
            for (int j = 0; j < numY; j ++) {
                myResultCopy[((o * numX) + i) * numY + j] = myResult[((o * numX) + i) * numY + j]; 
            }
        }
    }
#endif

    //cout << "Test5" << endl;
    updateParams_Kernel(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
    //cout << "Test6" << endl;
	rollback_Kernel(outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, u, v, a, b, c, y, yy, myResult);
	
    //cout << "Test7" << endl;
	for(uint i = 0; i < outer; i++) {
        res[i] = myResult[((i * numX) + myXindex) * numY + myYindex];
    }

#if TEST_INIT_CORRECTNESS
    vector<REAL>                   TestmyX(numX);       // [numX]
    vector<REAL>                   TestmyY(numY);       // [numY]
    vector<REAL>                   TestmyTimeline(numT);// [numT]
    vector<vector<REAL> >          TestmyDxx(numX, vector<REAL>(4));     // [numX][4]
    vector<vector<REAL> >          TestmyDyy(numY, vector<REAL>(4));     // [numY][4]
    vector<vector<vector<REAL> > > TestmyResult(outer, vector<vector<REAL>>(numX, vector<REAL>(numY))); // [outer][numX][numY]
    vector<vector<vector<REAL> > > TestmyVarX(numT, vector<vector<REAL>>(numX, vector<REAL>(numY)));    // [numT][numX][numY]
    vector<vector<vector<REAL> > > TestmyVarY(numT, vector<vector<REAL>>(numX, vector<REAL>(numY)));    // [numT][numX][numY]

    initGrid_Interchanged(s0, alpha, nu, t, numX, numY, numT, TestmyX, TestmyY, TestmyTimeline, myXindex, myYindex);
    for (int i = 0; i < numX; i ++) {
        if (abs(myX[i] - TestmyX[i]) > 0.00001f) {
            cout << "myX[" << i << "] did not match! was " << myX[i] << " expected " << TestmyX[i] << endl;
            return procs;
        }
    }
    for (int i = 0; i < numY; i ++) {
        if (abs(myY[i] - TestmyY[i]) > 0.00001f) {
            cout << "myY[" << i << "] did not match! was " << myY[i] << " expected " << TestmyY[i] << endl;
            return procs;
        }
    }
    for (int i = 0; i < numT; i ++) {
        if (abs(myTimeline[i] - TestmyTimeline[i]) > 0.00001f) {
            cout << "myTimeline[" << i << "] did not match! was " << myTimeline[i] << " expected " << TestmyTimeline[i] << endl;
            return procs;
        }
    }

    initOperator_Interchanged(numX, TestmyX, TestmyDxx);
    for (int i = 0; i < numX; i ++) {
        for (int j = 0; j < 4; j ++) {
            if (abs(myDxx[i * 4 + j] - TestmyDxx[i][j]) > 0.00001f) {
                cout << "myDxx[" << i << "][" << j << "] did not match! was " << myDxx[i * 4 + j] << " expected " << TestmyDxx[i][j] << endl;
                return procs;
            }
        }
    }

    initOperator_Interchanged(numY, TestmyY, TestmyDyy);
    for (int i = 0; i < numY; i ++) {
        for (int j = 0; j < 4; j ++) {
            if (abs(myDyy[i * 4 + j] - TestmyDyy[i][j]) > 0.00001f) {
                cout << "myDyy[" << i << "][" << j << "] did not match! was " << myDyy[i * 4 + j] << " expected " << TestmyDyy[i][j] << endl;
                return procs;
            }
        }
    }

    setPayoff_Interchanged(TestmyX, outer, numX, numY, TestmyResult);
    for (int o = 0; o < outer; o ++) {
        for (int i = 0; i < numX; i ++) {
            for (int j = 0; j < numY; j ++) {
                if (abs(myResultCopy[((o * numX) + i) * numY + j] - TestmyResult[o][i][j]) > 0.00001f) {
                    cout << "myResult[" << o << "][" << i << "][" << j << "] did not match! was " << myResultCopy[((o * numX) + i) * numY + j] << " expected " << TestmyResult[o][i][j] << endl;
                    return procs;
                }
            }
        }
    }

    updateParams_Interchanged(alpha, beta, nu, numX, numY, numT, TestmyX, TestmyY, TestmyTimeline, TestmyVarX, TestmyVarY);
    for (int t = 0; t < numT; t ++) {
        for (int i = 0; i < numX; i ++) {
            for (int j = 0; j < numY; j ++) {
                if (abs(myVarX[((t * numX) + i) * numY + j] - TestmyVarX[t][i][j]) > 0.00001f) {
                    cout << "myVarX[" << t << "][" << i << "][" << j << "] did not match! was " << myVarX[((t * numX) + i) * numY + j] << " expected " << TestmyVarX[t][i][j] << endl;
                    return procs;
                }
                if (abs(myVarY[((t * numX) + i) * numY + j] - TestmyVarY[t][i][j]) > 0.00001f) {
                    cout << "myVarY[" << t << "][" << i << "][" << j << "] did not match! was " << myVarY[((t * numX) + i) * numY + j] << " expected " << TestmyVarY[t][i][j] << endl;
                    return procs;
                }
            }
        }
    }
#endif
    return procs;
}
#endif