#ifndef KERNELIZED_ALGORITHM
#define KERNELIZED_ALGORITHM

#include "InterchangedAlgorithm.h"

#define TEST_INIT_CORRECTNESS true

void initGrid_Kernelized(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
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

void initGrid_Kernelized_para(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const unsigned numX, const unsigned numY, const unsigned numT,
                vector<REAL>& myX, vector<REAL>& myY, vector<REAL>& myTimeline,
                uint& myXindex, uint& myYindex
) {
#pragma omp parallel for schedule(static)
    for(unsigned i=0;i<numT;++i)
        myTimeline[i] = t*i/(numT-1);

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    myXindex = static_cast<unsigned>(s0/dx) % numX;

#pragma omp parallel for schedule(static)
    for(unsigned i=0;i<numX;++i)
        myX[i] = i*dx - myXindex*dx + s0;

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    myYindex = static_cast<unsigned>(numY/2.0);

#pragma omp parallel for schedule(static)
    for(unsigned i=0;i<numY;++i)
        myY[i] = i*dy - myYindex*dy + logAlpha;
}

void initOperator_Kernelized(  const uint& numZ, const vector<REAL>& myZ, 
                        vector<REAL>& Dzz
) {
    for (int gidx = 0; gidx < numZ * 4; gidx++) {
        uint row = gidx / numZ;
        uint col = gidx / 4;
        REAL dl, du;
        dl = (col == 0) ? 0.0 : myZ[col] - myZ[col - 1];
        du = (col == numZ - 1) ? 0.0 : myZ[col + 1] - myZ[col];
        Dzz[gidx] = col > 0 && col < numZ-1 ?
                    (row == 0 ? 2.0/dl/(dl+du) :
                    (row == 1 ? -2.0*(-2.0*(1.0/dl + 1.0/du)/(dl+du)) :
                    (row == 2 ? 2.0/du/(dl+du) :
                    0.0)))
                    : 0.0;
    }
}

void initOperator_Kernelized_para(  const uint& numZ, const vector<REAL>& myZ, 
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
void initOperator_Kernelized_T(  const uint& numZ, const vector<REAL>& myZ, 
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

void initOperator_Kernelized_T_para(  const uint& numZ, const vector<REAL>& myZ, 
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

void updateParams_Kernelized(const REAL alpha, const REAL beta, const REAL nu,
    const uint numX, const uint numY, const uint numT, 
    const vector<REAL> myX, const vector<REAL> myY, const vector<REAL> myTimeline,
    vector<REAL>& myVarX, vector<REAL>& myVarY)
{
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
}

void updateParams_Kernelized_para(const REAL alpha, const REAL beta, const REAL nu,
    const uint numX, const uint numY, const uint numT, 
    const vector<REAL> myX, const vector<REAL> myY, const vector<REAL> myTimeline,
    vector<REAL>& myVarX, vector<REAL>& myVarY)
{
#pragma omp parallel for schedule(static)
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
}

void setPayoff_Kernelized(const vector<REAL> myX, const uint outer,
    const uint numX, const uint numY,
    vector<REAL>& myResult)
{
    for(uint gidx = 0; gidx < outer * numX * numY; gidx++) {
        int o = gidx / (numX * numY);
        int plane_remain = gidx % (numX * numY);
        int i = plane_remain / numY;
        //int j = plane_remain % numY
        myResult[gidx] = max(myX[i]-0.001*(REAL)o, (REAL)0.0);
    }
}

void setPayoff_Kernelized_para(const vector<REAL> myX, const uint outer,
    const uint numX, const uint numY,
    vector<REAL>& myResult)
{
#pragma omp parallel for schedule(static)
    for(uint gidx = 0; gidx < outer * numX * numY; gidx++) {
        int o = gidx / (numX * numY);
        int plane_remain = gidx % (numX * numY);
        int i = plane_remain / numY;
        //int j = plane_remain % numY
        myResult[gidx] = max(myX[i]-0.001*(REAL)o, (REAL)0.0);
    }
}

void rollback_Kernelized(const uint outer, const uint numT, 
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
                tridagPar(a,0,b,0,c,0,y,0,numY,myResult, (gidx * numX + i) * numY,yy,0);
            }
        }
    }
}

void rollback_Kernelized_para(const uint outer, const uint numT, 
    const uint numX, const uint numY, 
    const vector<REAL> myTimeline, 
    const vector<REAL> myDxx,
    const vector<REAL> myDyy,
    const vector<REAL> myVarX,
    const vector<REAL> myVarY,
    vector<REAL>& myResult 
) {
    for (int t = 0; t <= numT - 2; t++) {
#pragma omp parallel for schedule(static)
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
                tridagPar(a,0,b,0,c,0,y,0,numY,myResult, (gidx * numX + i) * numY,yy,0);
            }
        }
    }
}

void rollback_Kernelized_Alt(const uint outer, const uint numT, 
    const uint numX, const uint numY, 
    const vector<REAL> myTimeline, 
    const vector<REAL> myDxx,
    const vector<REAL> myDyy,
    const vector<REAL> myVarX,
    const vector<REAL> myVarY,
    vector<REAL>& u,
    vector<REAL>& v,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c,
    vector<REAL>& y,
    vector<REAL>& yy,
    vector<REAL>& myResult
) {
    for (int t = 0; t <= numT - 2; t++) {
        for (int gidx = 0; gidx < outer; gidx++) {
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

void rollback_Kernelized_Alt_para(const uint outer, const uint numT, 
    const uint numX, const uint numY, 
    const vector<REAL> myTimeline, 
    const vector<REAL> myDxx,
    const vector<REAL> myDyy,
    const vector<REAL> myVarX,
    const vector<REAL> myVarY,
    vector<REAL>& u,
    vector<REAL>& v,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c,
    vector<REAL>& y,
    vector<REAL>& yy,
    vector<REAL>& myResult
) {
    for (int t = 0; t <= numT - 2; t++) {
#pragma omp parallel for schedule(static)
        for (int gidx = 0; gidx < outer; gidx++) {
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

#if TEST_INIT_CORRECTNESS
    vector<REAL> myResultCopy(outer * numX * numY);
#endif

    uint myXindex = 0;
    uint myYindex = 0;

    //cout << "Test1" << endl;
	initGrid_Kernelized(s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);

    //cout << "Test2" << endl;
    initOperator_Kernelized(numX, myX, myDxx);

    //cout << "Test3" << endl;
    initOperator_Kernelized(numY, myY, myDyy);

    //cout << "Test4" << endl;
    setPayoff_Kernelized(myX, outer, numX, numY, myResult);
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
    updateParams_Kernelized(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
    //cout << "Test6" << endl;
	rollback_Kernelized(outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, myResult);
	
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
            return 1;
        }
    }
    for (int i = 0; i < numY; i ++) {
        if (abs(myY[i] - TestmyY[i]) > 0.00001f) {
            cout << "myY[" << i << "] did not match! was " << myY[i] << " expected " << TestmyY[i] << endl;
            return 1;
        }
    }
    for (int i = 0; i < numT; i ++) {
        if (abs(myTimeline[i] - TestmyTimeline[i]) > 0.00001f) {
            cout << "myTimeline[" << i << "] did not match! was " << myTimeline[i] << " expected " << TestmyTimeline[i] << endl;
            return 1;
        }
    }

    initOperator_Interchanged(numX, TestmyX, TestmyDxx);
    for (int i = 0; i < numX; i ++) {
        for (int j = 0; j < 4; j ++) {
            if (abs(myDxx[i * 4 + j] - TestmyDxx[i][j]) > 0.00001f) {
                cout << "myDxx[" << i << "][" << j << "] did not match! was " << myDxx[i * 4 + j] << " expected " << TestmyDxx[i][j] << endl;
                return 1;
            }
        }
    }

    initOperator_Interchanged(numY, TestmyY, TestmyDyy);
    for (int i = 0; i < numY; i ++) {
        for (int j = 0; j < 4; j ++) {
            if (abs(myDyy[i * 4 + j] - TestmyDyy[i][j]) > 0.00001f) {
                cout << "myDyy[" << i << "][" << j << "] did not match! was " << myDyy[i * 4 + j] << " expected " << TestmyDyy[i][j] << endl;
                return 1;
            }
        }
    }

    setPayoff_Interchanged(TestmyX, outer, numX, numY, TestmyResult);
    for (int o = 0; o < outer; o ++) {
        for (int i = 0; i < numX; i ++) {
            for (int j = 0; j < numY; j ++) {
                if (abs(myResultCopy[((o * numX) + i) * numY + j] - TestmyResult[o][i][j]) > 0.00001f) {
                    cout << "myResult[" << o << "][" << i << "][" << j << "] did not match! was " << myResultCopy[((o * numX) + i) * numY + j] << " expected " << TestmyResult[o][i][j] << endl;
                    return 1;
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
                    return 1;
                }
                if (abs(myVarY[((t * numX) + i) * numY + j] - TestmyVarY[t][i][j]) > 0.00001f) {
                    cout << "myVarY[" << t << "][" << i << "][" << j << "] did not match! was " << myVarY[((t * numX) + i) * numY + j] << " expected " << TestmyVarY[t][i][j] << endl;
                    return 1;
                }
            }
        }
    }
#endif
    return 1;
}

int   run_SimpleKernelized_Parallel(  
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

#if TEST_INIT_CORRECTNESS
    vector<REAL> myResultCopy(outer * numX * numY);
#endif

    uint myXindex = 0;
    uint myYindex = 0;

    //cout << "Test1" << endl;
	initGrid_Kernelized_para(s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);

    //cout << "Test2" << endl;
    initOperator_Kernelized_para(numX, myX, myDxx);

    //cout << "Test3" << endl;
    initOperator_Kernelized_para(numY, myY, myDyy);

    //cout << "Test4" << endl;
    setPayoff_Kernelized_para(myX, outer, numX, numY, myResult);
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
    updateParams_Kernelized_para(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
    //cout << "Test6" << endl;
	rollback_Kernelized_para(outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, myResult);
	
    //cout << "Test7" << endl;
#pragma omp parallel for schedule(static)
	for(uint i = 0; i < outer; i++) {
        {
            int th_id = omp_get_thread_num();
            if(th_id == 0) { procs = omp_get_num_threads(); }
        }
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

int   run_Kernelized_Rollback(  
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

#if TEST_INIT_CORRECTNESS
    vector<REAL> myResultCopy(outer * numX * numY);
#endif

    uint numZ = max(numX, numY);
    vector<REAL> u(outer * numY * numX);
    vector<REAL> v(outer * numX * numY);
    vector<REAL> a(outer * numZ);
    vector<REAL> b(outer * numZ);
    vector<REAL> c(outer * numZ);
    vector<REAL> y(outer * numZ);
    vector<REAL> yy(outer * numZ);

    uint myXindex = 0;
    uint myYindex = 0;

    //cout << "Test1" << endl;
	initGrid_Kernelized(s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);

    //cout << "Test2" << endl;
    initOperator_Kernelized(numX, myX, myDxx);

    //cout << "Test3" << endl;
    initOperator_Kernelized(numY, myY, myDyy);

    //cout << "Test4" << endl;
    setPayoff_Kernelized(myX, outer, numX, numY, myResult);
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
    updateParams_Kernelized(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
    //cout << "Test6" << endl;
	rollback_Kernelized_Alt(outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, u, v, a, b, c, y, yy, myResult);
	
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
            return 1;
        }
    }
    for (int i = 0; i < numY; i ++) {
        if (abs(myY[i] - TestmyY[i]) > 0.00001f) {
            cout << "myY[" << i << "] did not match! was " << myY[i] << " expected " << TestmyY[i] << endl;
            return 1;
        }
    }
    for (int i = 0; i < numT; i ++) {
        if (abs(myTimeline[i] - TestmyTimeline[i]) > 0.00001f) {
            cout << "myTimeline[" << i << "] did not match! was " << myTimeline[i] << " expected " << TestmyTimeline[i] << endl;
            return 1;
        }
    }

    initOperator_Interchanged(numX, TestmyX, TestmyDxx);
    for (int i = 0; i < numX; i ++) {
        for (int j = 0; j < 4; j ++) {
            if (abs(myDxx[i * 4 + j] - TestmyDxx[i][j]) > 0.00001f) {
                cout << "myDxx[" << i << "][" << j << "] did not match! was " << myDxx[i * 4 + j] << " expected " << TestmyDxx[i][j] << endl;
                return 1;
            }
        }
    }

    initOperator_Interchanged(numY, TestmyY, TestmyDyy);
    for (int i = 0; i < numY; i ++) {
        for (int j = 0; j < 4; j ++) {
            if (abs(myDyy[i * 4 + j] - TestmyDyy[i][j]) > 0.00001f) {
                cout << "myDyy[" << i << "][" << j << "] did not match! was " << myDyy[i * 4 + j] << " expected " << TestmyDyy[i][j] << endl;
                return 1;
            }
        }
    }

    setPayoff_Interchanged(TestmyX, outer, numX, numY, TestmyResult);
    for (int o = 0; o < outer; o ++) {
        for (int i = 0; i < numX; i ++) {
            for (int j = 0; j < numY; j ++) {
                if (abs(myResultCopy[((o * numX) + i) * numY + j] - TestmyResult[o][i][j]) > 0.00001f) {
                    cout << "myResult[" << o << "][" << i << "][" << j << "] did not match! was " << myResultCopy[((o * numX) + i) * numY + j] << " expected " << TestmyResult[o][i][j] << endl;
                    return 1;
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
                    return 1;
                }
                if (abs(myVarY[((t * numX) + i) * numY + j] - TestmyVarY[t][i][j]) > 0.00001f) {
                    cout << "myVarY[" << t << "][" << i << "][" << j << "] did not match! was " << myVarY[((t * numX) + i) * numY + j] << " expected " << TestmyVarY[t][i][j] << endl;
                    return 1;
                }
            }
        }
    }
#endif
    return 1;
}

int   run_Kernelized_Rollback_Parallel(  
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

#if TEST_INIT_CORRECTNESS
    vector<REAL> myResultCopy(outer * numX * numY);
#endif

    uint numZ = max(numX, numY);
    vector<REAL> u(outer * numY * numX);
    vector<REAL> v(outer * numX * numY);
    vector<REAL> a(outer * numZ);
    vector<REAL> b(outer * numZ);
    vector<REAL> c(outer * numZ);
    vector<REAL> y(outer * numZ);
    vector<REAL> yy(outer * numZ);

    uint myXindex = 0;
    uint myYindex = 0;

    //cout << "Test1" << endl;
	initGrid_Kernelized_para(s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);

    //cout << "Test2" << endl;
    initOperator_Kernelized_para(numX, myX, myDxx);

    //cout << "Test3" << endl;
    initOperator_Kernelized_para(numY, myY, myDyy);

    //cout << "Test4" << endl;
    setPayoff_Kernelized_para(myX, outer, numX, numY, myResult);
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
    updateParams_Kernelized_para(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
    //cout << "Test6" << endl;
	rollback_Kernelized_Alt_para(outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, u, v, a, b, c, y, yy, myResult);
	
    //cout << "Test7" << endl;
#pragma omp parallel for schedule(static)
	for(uint i = 0; i < outer; i++) {
        {
            int th_id = omp_get_thread_num();
            if(th_id == 0) { procs = omp_get_num_threads(); }
        }
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