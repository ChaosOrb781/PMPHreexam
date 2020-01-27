#ifndef KERNELIZED_ALGORITHM
#define KERNELIZED_ALGORITHM

#include "InterchangedAlgorithm.h"

void initGrid_Kernel(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const unsigned numX, const unsigned numY, const unsigned numT,
                vector<REAL>& myX, vector<REAL>& myY, vector<REAL>& myTimeline,
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

void initOperator_Kernel(  const uint& numZ, const vector<REAL>& myZ, 
                        vector<REAL>& Dzz
) {
    REAL dl, du;
	//	lower boundary
	dl		 =  0.0;
	du		 =  myZ[1] - myZ[0];
	
	Dzz[0 * 4 + 0] =  0.0;
	Dzz[0 * 4 + 1] =  0.0;
	Dzz[0 * 4 + 2] =  0.0;
    Dzz[0 * 4 + 3] =  0.0;
	
	//	standard case
	for(unsigned i=1;i<numZ;i++)
	{
		dl      = myZ[i]   - myZ[i-1];
		du      = myZ[i+1] - myZ[i];

		Dzz[i * 4 + 0] =  2.0/dl/(dl+du);
		Dzz[i * 4 + 1] = -2.0*(1.0/dl + 1.0/du)/(dl+du);
		Dzz[i * 4 + 2] =  2.0/du/(dl+du);
        Dzz[i * 4 + 3] =  0.0; 
	}

	//	upper boundary
	dl		   =  myZ[numZ-1] - myZ[numZ-2];
	du		   =  0.0;

	Dzz[(numZ-1) * 4 + 0] = 0.0;
	Dzz[(numZ-1) * 4 + 1] = 0.0;
	Dzz[(numZ-1) * 4 + 2] = 0.0;
    Dzz[(numZ-1) * 4 + 3] = 0.0;
}

void initOperator_Kernel_para(  const uint& numZ, const vector<REAL>& myZ, 
                        vector<REAL>& Dzz
) {
	REAL dl, du;
	//	lower boundary
	dl		 =  0.0;
	du		 =  myZ[1] - myZ[0];
	
	Dzz[0 * 4 + 0] =  0.0;
	Dzz[0 * 4 + 1] =  0.0;
	Dzz[0 * 4 + 2] =  0.0;
    Dzz[0 * 4 + 3] =  0.0;
	
	//	standard case
#pragma omp parallel for schedule(static)
	for(unsigned i=1;i<numZ;i++)
	{
		dl      = myZ[i]   - myZ[i-1];
		du      = myZ[i+1] - myZ[i];

		Dzz[i * 4 + 0] =  2.0/dl/(dl+du);
		Dzz[i * 4 + 1] = -2.0*(1.0/dl + 1.0/du)/(dl+du);
		Dzz[i * 4 + 2] =  2.0/du/(dl+du);
        Dzz[i * 4 + 3] =  0.0; 
	}

	//	upper boundary
	dl		   =  myZ[numZ-1] - myZ[numZ-2];
	du		   =  0.0;

	Dzz[(numZ-1) * 4 + 0] = 0.0;
	Dzz[(numZ-1) * 4 + 1] = 0.0;
	Dzz[(numZ-1) * 4 + 2] = 0.0;
    Dzz[(numZ-1) * 4 + 3] = 0.0;
}

//Have to flatten to use device vector!
void initOperator_Kernel_T(  const uint& numZ, const vector<REAL>& myZ, 
                        vector<REAL>& DzzT
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

void initOperator_Kernel_T_para(  const uint& numZ, const vector<REAL>& myZ, 
                        vector<REAL>& DzzT
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

void updateParams_Kernel(const REAL alpha, const REAL beta, const REAL nu,
    const uint numX, const uint numY, const uint numT, 
    const vector<REAL> myX, const vector<REAL> myY, const vector<REAL> myTimeline,
    vector<REAL>& myVarX, vector<REAL>& myVarY)
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

void updateParams_Kernel_para(const REAL alpha, const REAL beta, const REAL nu,
    const uint numX, const uint numY, const uint numT, 
    const vector<REAL> myX, const vector<REAL> myY, const vector<REAL> myTimeline,
    vector<REAL>& myVarX, vector<REAL>& myVarY)
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

void setPayoff_Kernel(const vector<REAL> myX, const uint outer,
    const uint numX, const uint numY,
    vector<REAL>& myResult)
{
    for(uint gidx = 0; gidx < outer * numX * numY; gidx++) {
        int o = gidx / (numX * numY);
        int j = (gidx % (numX * numY)) / numY;
        myResult[gidx] = max(myX[j]-0.001*(REAL)o, (REAL)0.0);
    }
}

void setPayoff_Kernel_para(const vector<REAL> myX, const uint outer,
    const uint numX, const uint numY,
    vector<REAL>& myResult)
{
#pragma omp parallel for schedule(static)
    for(uint gidx = 0; gidx < outer * numX * numY; gidx++) {
        int o = gidx / (numX * numY);
        int j = gidx % (numX * numY);
        myResult[gidx] = max(myX[j]-0.001*(REAL)o, (REAL)0.0);
    }
}

void rollback_Kernel(const uint outer, const uint numT, 
    const uint numX, const uint numY, 
    const vector<REAL> myTimeline, 
    const vector<REAL> myDxx,
    const vector<REAL> myDyy,
    const vector<REAL> myVarX,
    const vector<REAL> myVarY,
    vector<REAL>& myResult
) {
    for (int t = 0; t <= numT - 2; t++) {
        for (int gidx = 0; gidx < outer; gidx++) {
            uint numZ = max(numX,numY);

            uint i, j;

            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);

            vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
            vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
            vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
            vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

            //cout << "explicit x, t: " << t << " o: " << gidx << endl;
            //	explicit x
            for(i=0;i<numX;i++) {
                for(j=0;j<numY;j++) {
                    u[j][i] = dtInv*myResult[((gidx * numX) + i) * numY + j];

                    if(i > 0) { 
                        u[j][i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                      * myDxx[i * 4 + 0] ) 
                                      * myResult[((gidx * numX) + (i-1)) * numY + j];
                    }
                    u[j][i]  +=  0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                    * myDxx[i * 4 + 1] )
                                    * myResult[((gidx * numX) + i) * numY + j];
                    if(i < numX-1) {
                        u[j][i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
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
                    v[i][j] = 0.0;

                    if(j > 0) {
                        v[i][j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                        * myDyy[j * 4 + 0] )
                                        * myResult[((gidx * numX) + i) * numY + j - 1];
                    }
                    v[i][j]  += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                     * myDyy[j * 4 + 1] )
                                     * myResult[((gidx * numX) + i) * numY + j];
                    if(j < numY-1) {
                        v[i][j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                        * myDyy[j * 4 + 2] )
                                        * myResult[((gidx * numX) + i) * numY + j + 1];
                    }
                    u[j][i] += v[i][j]; 
                }
            }

            //cout << "implicit x, t: " << t << " o: " << gidx << endl;
            //	implicit x
            for(j=0;j<numY;j++) {
                for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
                    a[i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 0]);
                    b[i] = dtInv - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 1]);
                    c[i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 2]);
                }
                // here yy should have size [numX]
                tridagPar(a,b,c,u[j],numX,u[j],yy);
            }

            //cout << "implicit y, t: " << t << " o: " << gidx << endl;
            //	implicit y
            for(i=0;i<numX;i++) { 
                for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
                    a[j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 0]);
                    b[j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 1]);
                    c[j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 2]);
                }

                for(j=0;j<numY;j++)
                    y[j] = dtInv*u[j][i] - 0.5*v[i][j];

                // here yy should have size [numY]
                tridagPar(a,b,c,y,0,numY,myResult, (gidx * numX + i) * numY,yy);
            }
        }
    }
}

void rollback_Kernel_para(const uint outer, const uint numT, 
    const uint numX, const uint numY, 
    const vector<REAL> myTimeline, 
    const vector<REAL> myDxx,
    const vector<REAL> myDyy,
    const vector<REAL> myVarX,
    const vector<REAL> myVarY,
    vector<REAL>& myResult 
) {
    for (int t = 0; t <= numT - 2; t++) {
        for (int gidx = 0; gidx < outer; gidx++) {
            uint numZ = max(numX,numY);

            uint i, j;

            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);

            vector<REAL> u(numY * numX);   // [numY][numX]
            vector<REAL> v(numX * numY);   // [numX][numY]
            vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
            vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

            //	explicit x
            for(i=0;i<numX;i++) {
                for(j=0;j<numY;j++) {
                    u[j * numX + i] = dtInv*myResult[((gidx * numX) + i) * numY + j];

                    if(i > 0) { 
                        u[j * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                      * myDxx[i * 4 + 0] ) 
                                      * myResult[((gidx * numX) + (i-1)) * numY + j];
                    }
                    u[j * numX + i]  +=  0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                    * myDxx[i * 4 + 1] )
                                    * myResult[((gidx * numX) + i) * numY + j];
                    if(i < numX-1) {
                        u[j * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                      * myDxx[i * 4 + 2] )
                                      * myResult[((gidx * numX) + (i+1)) * numY + j];
                    }
                }
            }

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
                    u[j * numX + i] += v[i * numY + j]; 
                }
            }

            //	implicit x
            for(j=0;j<numY;j++) {
                for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
                    a[i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 0]);
                    b[i] = dtInv - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 1]);
                    c[i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 2]);
                }
                // here yy should have size [numX]
                tridagPar(a,b,c,u,j,numX,u,j,yy);
            }

            //	implicit y
            for(i=0;i<numX;i++) { 
                for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
                    a[j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 0]);
                    b[j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 1]);
                    c[j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 2]);
                }

                for(j=0;j<numY;j++)
                    y[j] = dtInv*u[j * numX + i] - 0.5*v[i * numY + j];

                // here yy should have size [numY]
                tridagPar(a,b,c,y,0,numY,myResult, (gidx * numX + i) * numY,yy);
            }
        }
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
	vector<REAL> myX(numX);       // [numX]
    vector<REAL> myY(numY);       // [numY]
    vector<REAL> myTimeline(numT);// [numT]
    vector<REAL> myDxx(numX * 4);     // [numX][4]
    vector<REAL> myDyy(numY * 4);     // [numY][4]
    vector<REAL> myDxxT(4 * numX);       // [4][numX]
    vector<REAL> myDyyT(4 * numY);       // [4][numY]
    vector<REAL> myResult(outer * numX * numY); // [outer][numX][numY]
    vector<REAL> myVarX(numT * numX * numY);    // [numT][numX][numY]
    vector<REAL> myVarY(numT * numX * numY);    // [numT][numX][numY]
    vector<REAL> myVarXT(numT * numY * numX);    // [numT][numY][numX]

    vector<REAL>                   TestmyX(numX);       // [numX]
    vector<REAL>                   TestmyY(numY);       // [numY]
    vector<REAL>                   TestmyTimeline(numT);// [numT]
    vector<vector<REAL> >          TestmyDxx(numX, vector<REAL>(4));     // [numX][4]
    vector<vector<REAL> >          TestmyDyy(numY, vector<REAL>(4));     // [numY][4]
    vector<vector<vector<REAL> > > TestmyResult(outer, vector<vector<REAL>>(numX, vector<REAL>(numY))); // [outer][numX][numY]
    vector<vector<vector<REAL> > > TestmyVarX(numT, vector<vector<REAL>>(numX, vector<REAL>(numY)));    // [numT][numX][numY]
    vector<vector<vector<REAL> > > TestmyVarY(numT, vector<vector<REAL>>(numX, vector<REAL>(numY)));    // [numT][numX][numY]

    uint myXindex = 0;
    uint myYindex = 0;

    cout << "Test1" << endl;
	initGrid_Kernel(s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);
    initGrid_Alt(s0, alpha, nu, t, numX, numY, numT, TestmyX, TestmyY, TestmyTimeline, myXindex, myYindex);

    for (int i = 0; i < numX; i ++) {
        if (myX[i] != TestmyX[i]) {
            cout << "myX[" << i << "] did not match! was " << myX[i] << " expected " << TestmyX[i] << endl;
        }
    }
    for (int i = 0; i < numY; i ++) {
        if (myY[i] != TestmyY[i]) {
            cout << "myY[" << i << "] did not match! was " << myY[i] << " expected " << TestmyY[i] << endl;
        }
    }
    for (int i = 0; i < numT; i ++) {
        if (myTimeline[i] != TestmyTimeline[i]) {
            cout << "myTimeline[" << i << "] did not match! was " << myTimeline[i] << " expected " << TestmyTimeline[i] << endl;
        }
    }

    cout << "Test2" << endl;
    initOperator_Kernel(numX, myX, myDxx);
    initOperator_Alt(numX, TestmyX, TestmyDxx);
    for (int i = 0; i < numX; i ++) {
        for (int j = 0; j < 4; j ++) {
            if (myDxx[i * 4 + j] != TestmyDxx[i][j]) {
                cout << "myDxx[" << i << "][" << j << "] did not match! was " << myDxx[i * 4 + j] << " expected " << TestmyDxx[i][j] << endl;
            }
        }
    }

    cout << "Test3" << endl;
    initOperator_Kernel(numY, myY, myDyy);
    initOperator_Alt(numY, TestmyY, TestmyDyy);
    for (int i = 0; i < numY; i ++) {
        for (int j = 0; j < 4; j ++) {
            if (myDyy[i * 4 + j] != TestmyDyy[i][j]) {
                cout << "myDyy[" << i << "][" << j << "] did not match! was " << myDyy[i * 4 + j] << " expected " << TestmyDyy[i][j] << endl;
            }
        }
    }



    cout << "Test4" << endl;
    setPayoff_Kernel(myX, outer, numX, numY, myResult);
    setPayoff_Alt(TestmyX, outer, numX, numY, TestmyResult);
    for (int o = 0; o < outer; o ++) {
        for (int i = 0; i < numX; i ++) {
            for (int j = 0; j < numY; j ++) {
                if (myResult[((o * numX) + i) * numY + j] != TestmyResult[o][i][j]) {
                    cout << "myResult[" << o << "][" << i << "][" << j << "] did not match! was " << myResult[((o * numX) + i) * numY + j] << " expected " << TestmyResult[o][i][j] << endl;
                }
            }
        }
    }

    cout << "Test5" << endl;
    updateParams_Kernel(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);

    cout << "Test6" << endl;
	rollback_Kernel(outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, myResult);
	
    cout << "Test7" << endl;
	for(uint i = 0; i < outer; i++) {
        res[i] = myResult[((i * numX) + myXindex) * numY + myYindex];
    }
    return 1;
}
#endif