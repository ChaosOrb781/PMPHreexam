#ifndef DISTRIBUTED_ALGORITHM
#define DISTRIBUTED_ALGORITHM

#include "InterchangedAlgorithm.h"

#define TEST_INIT_CORRECTNESS false

void initGrid_Distributed(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
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

void initGrid_Distributed_para(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
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

void initOperator_Distributed(  const uint& numZ, const vector<REAL>& myZ, 
                        vector<REAL>& Dzz
) {
    for (int gidx = 0; gidx < numZ * 4; gidx++) {
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

void initOperator_Distributed_para(  const uint& numZ, const vector<REAL>& myZ, 
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
void initOperator_Distributed_T(  const uint& numZ, const vector<REAL>& myZ, 
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

void initOperator_Distributed_T_para(  const uint& numZ, const vector<REAL>& myZ, 
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

void updateParams_Distributed(const REAL alpha, const REAL beta, const REAL nu,
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

void updateParams_Distributed_para(const REAL alpha, const REAL beta, const REAL nu,
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

void setPayoff_Distributed(const vector<REAL> myX, const uint outer,
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

void setPayoff_Distributed_para(const vector<REAL> myX, const uint outer,
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

void rollback_Distributed(const uint outer, const uint numT, 
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
        //cout << "test 1" << endl;
        for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
            uint o = gidx / (numX * numY);
            uint plane_remain = gidx % (numX * numY);
            uint i = plane_remain / numY;
            uint j = plane_remain % numY;
            uint numZ = max(numX,numY);
            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
            u[((o * numY) + j) * numX + i] = dtInv*myResult[((o * numX) + i) * numY + j];

            if(i > 0) { 
                u[((o * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                * myDxx[i * 4 + 0] ) 
                                * myResult[((o * numX) + (i-1)) * numY + j];
            }
            u[((o * numY) + j) * numX + i]  +=  0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                            * myDxx[i * 4 + 1] )
                            * myResult[((o * numX) + i) * numY + j];
            if(i < numX-1) {
                u[((o * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                * myDxx[i * 4 + 2] )
                                * myResult[((o * numX) + (i+1)) * numY + j];
            }
        }

        //cout << "test 2" << endl;
        for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
            uint o = gidx / (numY * numX);
            uint plane_remain = gidx % (numY * numX);
            uint j = plane_remain / numX;
            uint i = plane_remain % numX;
            uint numZ = max(numX,numY);
            v[((o * numX) + i) * numY + j] = 0.0;

            if(j > 0) {
                v[((o * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                * myDyy[j * 4 + 0] )
                                * myResult[((o * numX) + i) * numY + j - 1];
            }
            v[((o * numX) + i) * numY + j]  += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                * myDyy[j * 4 + 1] )
                                * myResult[((o * numX) + i) * numY + j];
            if(j < numY-1) {
                v[((o * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                * myDyy[j * 4 + 2] )
                                * myResult[((o * numX) + i) * numY + j + 1];
            }
            u[((o * numY) + j) * numX + i] += v[((o * numX) + i) * numY + j];
        }

        //cout << "test 3" << endl;
        for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
            uint o = gidx / (numY * numX);
            uint plane_remain = gidx % (numY * numX);
            uint j = plane_remain / numX;
            uint i = plane_remain % numX;
            uint numZ = max(numX,numY);
            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
            a[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 0]);
            b[((o * numZ) + j) * numZ + i] = dtInv - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 1]);
            c[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 2]);
        }

        //cout << "test 4" << endl;
        for(uint j=0;j<numY;j++) {
            for (int gidx = 0; gidx < outer; gidx++) {
                uint numZ = max(numX,numY);
                // here yy should have size [numX]
                tridagPar(a,((gidx * numZ) + j) * numZ,b,((gidx * numZ) + j) * numZ,c,((gidx * numZ) + j) * numZ,u,((gidx * numY) + j) * numX,numX,u,((gidx * numY) + j) * numX,yy,(gidx * numZ));
            }
        }

        //cout << "test 5" << endl;
        for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
            uint o = gidx / (numX * numY);
            uint plane_remain = gidx % (numX * numY);
            uint i = plane_remain / numY;
            uint j = plane_remain % numY;
            uint numZ = max(numX,numY);
            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
            a[((o * numZ) + i) * numZ + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 0]);
            b[((o * numZ) + i) * numZ + j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 1]);
            c[((o * numZ) + i) * numZ + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 2]);
        }

        //cout << "test 6" << endl;
        for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
            uint o = gidx / (numX * numY);
            uint plane_remain = gidx % (numX * numY);
            uint i = plane_remain / numY;
            uint j = plane_remain % numY;
            uint numZ = max(numX,numY);
            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
            y[((o * numZ) + i) * numZ + j] = dtInv*u[((o * numY) + j) * numX + i] - 0.5*v[((o * numX) + i) * numY + j];
        }

        for(uint i=0;i<numX;i++) {
            for (int gidx = 0; gidx < outer; gidx++) {
                // here yy should have size [numY]
                uint numZ = max(numX,numY);
                tridagPar(a,((gidx * numZ) + i) * numZ,b,((gidx * numZ) + i) * numZ,c,((gidx * numZ) + i) * numZ,y,((gidx * numZ) + i) * numZ,numY,myResult, (gidx * numX + i) * numY,yy,(gidx * numZ));
            }
        }
    }
}

void rollback_Distributed_1(
    int t,
    const uint outer, 
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline, 
    const vector<REAL> myDxx,
    const vector<REAL> myVarX,
    vector<REAL>& u,
    vector<REAL>& myResult
) {
    //cout << "test 1" << endl;
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        u[((o * numY) + j) * numX + i]      = dtInv * myResult[((o * numX) + i) * numY + j];

        if(i > 0) { 
            u[((o * numY) + j) * numX + i] += 0.5* ( 0.5*myVarX[((t * numX) + i) * numY + j] * myDxx[i * 4 + 0] ) * myResult[((o * numX) + (i-1)) * numY + j];
        }
        u[((o * numY) + j) * numX + i]     += 0.5* ( 0.5*myVarX[((t * numX) + i) * numY + j] * myDxx[i * 4 + 1] ) * myResult[((o * numX) + i) * numY + j];
        if(i < numX-1) {
            u[((o * numY) + j) * numX + i] += 0.5* ( 0.5*myVarX[((t * numX) + i) * numY + j] * myDxx[i * 4 + 2] ) * myResult[((o * numX) + (i+1)) * numY + j];
        }
    }
}
void rollback_Distributed_2(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline,
    const vector<REAL> myDyy,
    const vector<REAL> myVarY,
    vector<REAL>& u,
    vector<REAL>& v,
    vector<REAL>& myResult
) {
    //cout << "test 2" << endl;
    for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
        uint o = gidx / (numY * numX);
        uint plane_remain = gidx % (numY * numX);
        uint j = plane_remain / numX;
        uint i = plane_remain % numX;
        uint numZ = max(numX,numY);
        v[((o * numX) + i) * numY + j] = 0.0;

        if(j > 0) {
            v[((o * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                            * myDyy[j * 4 + 0] )
                            * myResult[((o * numX) + i) * numY + j - 1];
        }
        v[((o * numX) + i) * numY + j]  += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                            * myDyy[j * 4 + 1] )
                            * myResult[((o * numX) + i) * numY + j];
        if(j < numY-1) {
            v[((o * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                            * myDyy[j * 4 + 2] )
                            * myResult[((o * numX) + i) * numY + j + 1];
        }
        u[((o * numY) + j) * numX + i] += v[((o * numX) + i) * numY + j];
    }
}

void rollback_Distributed_3(
    int t,
    const uint outer,
    const uint numX,
    const uint numY,
    const vector<REAL> myTimeline, 
    const vector<REAL> myDxx,
    const vector<REAL> myVarX,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c
) {
    //cout << "test 3" << endl;
    for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
        uint o = gidx / (numY * numX);
        uint plane_remain = gidx % (numY * numX);
        uint j = plane_remain / numX;
        uint i = plane_remain % numX;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        a[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 0]);
        b[((o * numZ) + j) * numZ + i] = dtInv - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 1]);
        c[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 2]);
    }
}


void rollback_Distributed_4(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY,
    vector<REAL>& u,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c,
    vector<REAL>& yy
) {
    //cout << "test 4" << endl;
    for(uint j=0;j<numY;j++) {
        for (int gidx = 0; gidx < outer; gidx++) {
            uint numZ = max(numX,numY);
            // here yy should have size [numX]
            tridagPar(a,((gidx * numZ) + j) * numZ,b,((gidx * numZ) + j) * numZ,c,((gidx * numZ) + j) * numZ,u,((gidx * numY) + j) * numX,numX,u,((gidx * numY) + j) * numX,yy,((gidx * numZ) + j) * numZ);
        }
    }
}

void rollback_Distributed_5(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline,
    const vector<REAL> myDyy,
    const vector<REAL> myVarY,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c
) {
    //cout << "test 5" << endl;
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        a[((o * numZ) + i) * numZ + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 0]);
        b[((o * numZ) + i) * numZ + j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 1]);
        c[((o * numZ) + i) * numZ + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 2]);
    }
}

void rollback_Distributed_6(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline,
    vector<REAL>& u,
    vector<REAL>& v,
    vector<REAL>& y
) {
    //cout << "test 6" << endl;
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        y[((o * numZ) + i) * numZ + j] = dtInv*u[((o * numY) + j) * numX + i] - 0.5*v[((o * numX) + i) * numY + j];
    }
}

void rollback_Distributed_7(
    int t,
    const uint outer, 
    const uint numX, 
    const uint numY,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c,
    vector<REAL>& y,
    vector<REAL>& yy,
    vector<REAL>& myResult
) {
    for(uint i=0;i<numX;i++) {
        for (int gidx = 0; gidx < outer; gidx++) {
            // here yy should have size [numY]
            uint numZ = max(numX,numY);
            tridagPar(a,((gidx * numZ) + i) * numZ,b,((gidx * numZ) + i) * numZ,c,((gidx * numZ) + i) * numZ,y,((gidx * numZ) + i) * numZ,numY,myResult, (gidx * numX + i) * numY,yy,(gidx * numZ));
        }
    }
}

void rollback_Distributed_1_para(
    int t,
    const uint outer, 
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline, 
    const vector<REAL> myDxx,
    const vector<REAL> myVarX,
    vector<REAL>& u,
    vector<REAL>& myResult
) {
    //cout << "test 1" << endl;
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        u[((o * numY) + j) * numX + i] = dtInv*myResult[((o * numX) + i) * numY + j];

        if(i > 0) { 
            u[((o * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                            * myDxx[i * 4 + 0] ) 
                            * myResult[((o * numX) + (i-1)) * numY + j];
        }
        u[((o * numY) + j) * numX + i]  +=  0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                        * myDxx[i * 4 + 1] )
                        * myResult[((o * numX) + i) * numY + j];
        if(i < numX-1) {
            u[((o * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                            * myDxx[i * 4 + 2] )
                            * myResult[((o * numX) + (i+1)) * numY + j];
        }
    }
}
void rollback_Distributed_2_para(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline,
    const vector<REAL> myDyy,
    const vector<REAL> myVarY,
    vector<REAL>& u,
    vector<REAL>& v,
    vector<REAL>& myResult
) {
    //cout << "test 2" << endl;
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
        uint o = gidx / (numY * numX);
        uint plane_remain = gidx % (numY * numX);
        uint j = plane_remain / numX;
        uint i = plane_remain % numX;
        uint numZ = max(numX,numY);
        v[((o * numX) + i) * numY + j] = 0.0;

        if(j > 0) {
            v[((o * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                            * myDyy[j * 4 + 0] )
                            * myResult[((o * numX) + i) * numY + j - 1];
        }
        v[((o * numX) + i) * numY + j]  += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                            * myDyy[j * 4 + 1] )
                            * myResult[((o * numX) + i) * numY + j];
        if(j < numY-1) {
            v[((o * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                            * myDyy[j * 4 + 2] )
                            * myResult[((o * numX) + i) * numY + j + 1];
        }
        u[((o * numY) + j) * numX + i] += v[((o * numX) + i) * numY + j];
    }
}

void rollback_Distributed_3_para(
    int t,
    const uint outer,
    const uint numX,
    const uint numY,
    const vector<REAL> myTimeline, 
    const vector<REAL> myDxx,
    const vector<REAL> myVarX,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c
) {
    //cout << "test 3" << endl;
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
        uint o = gidx / (numY * numX);
        uint plane_remain = gidx % (numY * numX);
        uint j = plane_remain / numX;
        uint i = plane_remain % numX;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        a[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 0]);
        b[((o * numZ) + j) * numZ + i] = dtInv - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 1]);
        c[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 2]);
    }
}


void rollback_Distributed_4_para(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY,
    vector<REAL>& u,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c,
    vector<REAL>& yy
) {
    //cout << "test 4" << endl;
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < outer * numY; gidx++) {
        uint o = gidx / numY;
        uint j = gidx % numY;
        uint numZ = max(numX,numY);
        // here yy should have size [numX]
        tridagPar(a,((o * numZ) + j) * numZ,b,((o * numZ) + j) * numZ,c,((o * numZ) + j) * numZ,u,((o * numY) + j) * numX,numX,u,((o * numY) + j) * numX,yy,((o * numZ) + j) * numZ);
    }
}

void rollback_Distributed_5_para(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline,
    const vector<REAL> myDyy,
    const vector<REAL> myVarY,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c
) {
    //cout << "test 5" << endl;
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        a[((o * numZ) + i) * numZ + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 0]);
        b[((o * numZ) + i) * numZ + j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 1]);
        c[((o * numZ) + i) * numZ + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 2]);
    }
}

void rollback_Distributed_6_para(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline,
    vector<REAL>& u,
    vector<REAL>& v,
    vector<REAL>& y
) {
    //cout << "test 6" << endl;
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        y[((o * numZ) + i) * numZ + j] = dtInv*u[((o * numY) + j) * numX + i] - 0.5*v[((o * numX) + i) * numY + j];
    }
}

void rollback_Distributed_7_para(
    int t,
    const uint outer, 
    const uint numX, 
    const uint numY,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c,
    vector<REAL>& y,
    vector<REAL>& yy,
    vector<REAL>& myResult
) {
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < outer * numX; gidx++) {
        uint o = gidx / numX;
        uint i = gidx % numX;
        // here yy should have size [numY]
        uint numZ = max(numX,numY);
        tridagPar(a,((o * numZ) + i) * numZ,b,((o * numZ) + i) * numZ,c,((o * numZ) + i) * numZ,y,((o * numZ) + i) * numZ,numY,myResult, (o * numX + i) * numY,yy,((o * numZ) + i) * numZ);
    }
}

void initMyTimeline_Distributed_Final(
    const REAL t,
    const uint numT, 
    vector<REAL>& myTimeline
) {
    for(uint gidx = 0; gidx < numT; gidx++)
        myTimeline[gidx] = t*gidx/(numT-1);
}

void initMyX_Distributed_Final(  
    const REAL s0, 
    const REAL alpha, 
    const REAL t, 
    const unsigned numX,
    vector<REAL>& myX,
    uint& myXindex
) {
    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    myXindex = static_cast<unsigned>(s0/dx) % numX;

    for(uint gidx = 0; gidx < numX; gidx++)
        myX[gidx] = gidx*dx - myXindex*dx + s0;
}

void initMyY_Distributed_Final(  
    const REAL s0,
    const REAL alpha,
    const REAL nu,
    const REAL t,
    const unsigned numY, 
    vector<REAL>& myY, 
    uint& myYindex
) {
    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    myYindex = static_cast<unsigned>(numY/2.0);

    for(uint gidx = 0; gidx < numY; gidx++)
        myY[gidx] = gidx*dy - myYindex*dy + logAlpha;
}

void initMyTimeline_Distributed_Final_para(
    const REAL t,
    const uint numT, 
    vector<REAL>& myTimeline
) {
    for(uint gidx = 0; gidx < numT; gidx++)
        myTimeline[gidx] = t*gidx/(numT-1);
}

void initMyX_Distributed_Final_para(  
    const REAL s0, 
    const REAL alpha, 
    const REAL t, 
    const unsigned numX,
    vector<REAL>& myX,
    uint& myXindex
) {
    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    myXindex = static_cast<unsigned>(s0/dx) % numX;

    for(uint gidx = 0; gidx < numX; gidx++)
        myX[gidx] = gidx*dx - myXindex*dx + s0;
}

void initMyY_Distributed_Final_para(  
    const REAL s0,
    const REAL alpha,
    const REAL nu,
    const REAL t,
    const unsigned numY, 
    vector<REAL>& myY, 
    uint& myYindex
) {
    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    myYindex = static_cast<unsigned>(numY/2.0);

    for(uint gidx = 0; gidx < numY; gidx++)
        myY[gidx] = gidx*dy - myYindex*dy + logAlpha;
}

void initOperator_Distributed_T_Final(  
    const uint& numZ, 
    const vector<REAL>& myZ, 
    vector<REAL>& DzzT
) {
    for (int gidx = 0; gidx < numZ; gidx++) {
        REAL low = gidx > 0 ? myZ[gidx - 1] : 0.0;
        REAL mid = myZ[gidx];
        REAL high = gidx < numZ - 1 ? myZ[gidx + 1] : 0.0;

        REAL dl = mid - low;
        REAL du = high - mid;

        DzzT[0 * numZ + gidx] = gidx > 0 && gidx < numZ - 1 ? 2.0 / dl / (dl + du) : 0.0;
        DzzT[1 * numZ + gidx] = gidx > 0 && gidx < numZ - 1 ? -2.0 * (1.0 / dl + 1.0 / du) / (dl + du) : 0.0;
        DzzT[2 * numZ + gidx] = gidx > 0 && gidx < numZ - 1 ? 2.0 / du / (dl + du) : 0.0;
        DzzT[3 * numZ + gidx] = 0.0;
    }
}

void initOperator_Distributed_T_Final_para(  
    const uint& numZ, 
    const vector<REAL>& myZ, 
    vector<REAL>& DzzT
) {
#pragma omp parallel for schedule(static)
	for (int gidx = 0; gidx < numZ; gidx++) {
        REAL low = gidx > 0 ? myZ[gidx - 1] : 0.0;
        REAL mid = myZ[gidx];
        REAL high = gidx < numZ - 1 ? myZ[gidx + 1] : 0.0;

        REAL dl = mid - low;
        REAL du = high - mid;

        DzzT[0 * numZ + gidx] = gidx > 0 && gidx < numZ - 1 ? 2.0 / dl / (dl + du) : 0.0;
        DzzT[0 * numZ + gidx] = gidx > 0 && gidx < numZ - 1 ? -2.0 / (1.0 / dl + 1.0 / du) / (dl + du) : 0.0;
        DzzT[0 * numZ + gidx] = gidx > 0 && gidx < numZ - 1 ? 2.0 / du / (dl + du) : 0.0;
        DzzT[0 * numZ + gidx] = 0.0;
    }
}

void updateParams_Distributed_VarXT_Final(
    const REAL alpha, 
    const REAL beta, 
    const REAL nu,
    const uint numX, 
    const uint numY, 
    const uint numT, 
    const vector<REAL> myX, 
    const vector<REAL> myY, 
    const vector<REAL> myTimeline,
    vector<REAL>& myVarXT
){
    //load myTimeline and myY into shared memory (entire)
    vector<REAL> myT_sh(myTimeline);
    vector<REAL> myY_sh(myY);

    for (int gidx = 0; gidx < numX; gidx++) {
        for (int t = 0; t < numT; t++) {
            for (int j = 0; j < numY; j++) { 
                REAL val1 = beta*log(myX[gidx]);
                REAL val2 = myY_sh[j];
                REAL val3 = - 0.5*nu*nu*myT_sh[t];

                myVarXT[((t * numY) + j) * numX + gidx] = exp(2.0*(val1 + val2 + val3));
            }
        }
    }
}

void updateParams_Distributed_VarY_Final(
    const REAL alpha, 
    const REAL beta, 
    const REAL nu,
    const uint numX, 
    const uint numY, 
    const uint numT, 
    const vector<REAL> myX, 
    const vector<REAL> myY, 
    const vector<REAL> myTimeline,
    vector<REAL>& myVarY
){
    //load myTimeline and myY into shared memory (entire)
    vector<REAL> myT_sh(myTimeline);
    vector<REAL> myX_sh(myX);
    
    for (int gidx = 0; gidx < numY; gidx++) {
        for (int t = 0; t < numT; t++) {
            for (int i = 0; i < numX; i++) { 
                REAL val1 = alpha*log(myX_sh[i]);
                REAL val2 = myY[gidx];
                REAL val3 = - 0.5*nu*nu*myT_sh[t];

                myVarY[((t * numX) + i) * numY + gidx] = exp(2.0*(val1 + val2 + val3));
            }
        }
    }
}

void updateParams_Distributed_VarXT_Final_para(
    const REAL alpha, 
    const REAL beta, 
    const REAL nu,
    const uint numX, 
    const uint numY, 
    const uint numT, 
    const vector<REAL> myX, 
    const vector<REAL> myY, 
    const vector<REAL> myTimeline,
    vector<REAL>& myVarXT
){
    //load myTimeline and myY into shared memory (entire)
    vector<REAL> myT_sh(myTimeline);
    vector<REAL> myY_sh(myY);

#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < numX; gidx++) {
        for (int t = 0; t < numT; t++) {
            for (int j = 0; j < numY; j++) { 
                REAL val1 = beta*log(myX[gidx]);
                REAL val2 = myY_sh[j];
                REAL val3 = - 0.5*nu*nu*myT_sh[t];

                myVarXT[((t * numY) + j) * numX + gidx] = exp(2.0*(val1 + val2 + val3));
            }
        }
    }
}

void updateParams_Distributed_VarY_Final_para(
    const REAL alpha, 
    const REAL beta, 
    const REAL nu,
    const uint numX, 
    const uint numY, 
    const uint numT, 
    const vector<REAL> myX, 
    const vector<REAL> myY, 
    const vector<REAL> myTimeline,
    vector<REAL>& myVarY
){
    //load myTimeline and myY into shared memory (entire)
    vector<REAL> myT_sh(myTimeline);
    vector<REAL> myX_sh(myX);

#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < numY; gidx++) {
        for (int t = 0; t < numT; t++) {
            for (int i = 0; i < numX; i++) { 
                REAL val1 = alpha*log(myX_sh[i]);
                REAL val2 = myY[gidx];
                REAL val3 = - 0.5*nu*nu*myT_sh[t];

                myVarY[((t * numX) + i) * numY + gidx] = exp(2.0*(val1 + val2 + val3));
            }
        }
    }
}

void setPayoff_Distributed_T_Final(
    const vector<REAL> myX, 
    const uint outer,
    const uint numX, 
    const uint numY,
    vector<REAL>& myResultT
){
    for(uint gidx = 0; gidx < numX; gidx++) {
        REAL x = myX[gidx];
        for (int o = 0; o < outer; o++) {
            for (int j = 0; j < numY; j++) {
                REAL payoff = x-0.001*(REAL)o;
                myResultT[((o * numY) + j) * numX + gidx] = payoff > 0.0 ? payoff : (REAL)0.0;
            }
        }
    }
}

void setPayoff_Distributed_T_Final_para(
    const vector<REAL> myX, 
    const uint outer,
    const uint numX, 
    const uint numY,
    vector<REAL>& myResultT
){
#pragma omp parallel for schedule(static)
    for(uint gidx = 0; gidx < numX; gidx++) {
        REAL x = myX[gidx];
        for (int o = 0; o < outer; o++) {
            for (int j = 0; j < numY; j++) {
                REAL payoff = x-0.001*(REAL)o;
                myResultT[((o * numY) + j) * numX + gidx] = payoff > 0.0 ? payoff : (REAL)0.0;
            }
        }
    }
}

void rollback_Distributed_1_Final(
    int t,
    const uint outer, 
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline, 
    const vector<REAL> myDxxT,
    const vector<REAL> myVarXT,
    const vector<REAL> myResultT,
    vector<REAL>& u
) {
    //cout << "test 1" << endl;
    for (int gidx = 0; gidx < numX; gidx++) {
        uint numZ = max(numX,numY);

        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);

        for (int o = 0; o < outer; o++) {
            for (int j = 0; j < numY; j++) {
                REAL myDxxT0 = myDxxT[0 * numX + gidx];
                REAL myDxxT1 = myDxxT[1 * numX + gidx];
                REAL myDxxT2 = myDxxT[2 * numX + gidx];

                REAL myResultT_low  = myResultT[((o * numY) + j) * numX + gidx - 1];
                REAL myResultT_mid  = myResultT[((o * numY) + j) * numX + gidx];
                REAL myResultT_high = myResultT[((o * numY) + j) * numX + gidx + 1];

                REAL myVarXT_val = 0.5 * myVarXT[((t * numY) + j) * numX + gidx];

                u[((o * numY) + j) * numX + gidx] = dtInv * myResultT_mid;

                if(gidx > 0) { 
                    u[((o * numY) + j) * numX + gidx] += 
                        0.5*( myVarXT_val * myDxxT0 ) * myResultT_low;
                }

                u[((o * numY) + j) * numX + gidx]  +=  
                    0.5*( myVarXT_val * myDxxT1 ) * myResultT_mid;

                if(gidx < numX-1) {
                    u[((o * numY) + j) * numX + gidx] += 
                        0.5*( myVarXT_val * myDxxT2 ) * myResultT_high;
                }
            }
        }
    }
}
void rollback_Distributed_2_Final1(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline,
    const vector<REAL> myDyyT,
    const vector<REAL> myVarY,
    vector<REAL>& u,
    vector<REAL>& v,
    vector<REAL>& myResult
) {
    //cout << "test 2" << endl;
    for (int gidx = 0; gidx < numY; gidx++) {
        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);


        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                REAL myDyyT0 = myDyyT[0 * numY + gidx];
                REAL myDyyT1 = myDyyT[1 * numY + gidx];
                REAL myDyyT2 = myDyyT[2 * numY + gidx];

                REAL myResult_low  = myResult[((o * numX) + i) * numY + gidx - 1];
                REAL myResult_mid  = myResult[((o * numX) + i) * numY + gidx];
                REAL myResult_high = myResult[((o * numX) + i) * numY + gidx + 1];

                REAL myVarY_val = 0.5 * myVarY[((t * numX) + i) * numY + gidx];

                v[((o * numX) + i) * numY + gidx] = dtInv * myResult_mid;

                if(gidx > 0) { 
                    v[((o * numX) + i) * numY + gidx] += 
                        0.5*( myVarY_val * myDyyT0 ) * myResult_low;
                }

                v[((o * numX) + i) * numY + gidx]  +=  
                    0.5*( myVarY_val * myDyyT1 ) * myResult_mid;

                if(gidx < numY-1) {
                    v[((o * numX) + i) * numY + gidx] += 
                        0.5*( myVarY_val * myDyyT2 ) * myResult_high;
                }
            }
        }
    }
}

void rollback_Distributed_2_Final2(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY,
    vector<REAL>& uT,
    vector<REAL>& v
) {
    //cout << "test 2" << endl;
    for (int gidx = 0; gidx < numY; gidx++) {
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                uT[((o * numX) + i) * numY + gidx] += v[((o * numX) + i) * numY + gidx];
            }
        }
    }
}

void rollback_Distributed_3_Final(
    int t,
    const uint outer,
    const uint numX,
    const uint numY,
    const vector<REAL> myTimeline, 
    const vector<REAL> myDxxT,
    const vector<REAL> myVarXT,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c
) {
    //cout << "test 3" << endl;
    for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
        uint o = gidx / (numY * numX);
        uint plane_remain = gidx % (numY * numX);
        uint i = plane_remain % numX;
        uint j = plane_remain / numX;

        uint numZ = max(numX,numY);

        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);

        a[((o * numZ) + j) * numZ + i] =       - 0.5*(0.5*myVarXT[((t * numY) + j) * numX + i]*myDxxT[0 * numX + i]);
        b[((o * numZ) + j) * numZ + i] = dtInv - 0.5*(0.5*myVarXT[((t * numY) + j) * numX + i]*myDxxT[1 * numX + i]);
        c[((o * numZ) + j) * numZ + i] =       - 0.5*(0.5*myVarXT[((t * numY) + j) * numX + i]*myDxxT[2 * numX + i]);
    }
}


void rollback_Distributed_4_Final(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY,
    vector<REAL>& u,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c,
    vector<REAL>& yy
) {
    //cout << "test 4" << endl;
    for(uint j=0;j<numY;j++) {
        for (int gidx = 0; gidx < outer; gidx++) {
            uint numZ = max(numX,numY);
            // here yy should have size [numX]
            tridagPar(a,((gidx * numZ) + j) * numZ,b,((gidx * numZ) + j) * numZ,c,((gidx * numZ) + j) * numZ,u,((gidx * numY) + j) * numX,numX,u,((gidx * numY) + j) * numX,yy,((gidx * numZ) + j) * numZ);
        }
    }
}

void rollback_Distributed_5_Final(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline,
    const vector<REAL> myDyyT,
    const vector<REAL> myVarY,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c
) {
    //cout << "test 5" << endl;
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);

        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);

        a[((o * numZ) + i) * numZ + j] =       - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyyT[0 * numY + j]);
        b[((o * numZ) + i) * numZ + j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyyT[1 * numY + j]);
        c[((o * numZ) + i) * numZ + j] =       - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyyT[2 * numY + j]);
    }
}

void rollback_Distributed_6_Final(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline,
    vector<REAL>& uT,
    vector<REAL>& v,
    vector<REAL>& y
) {
    //cout << "test 6" << endl;
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);

        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);

        y[((o * numZ) + i) * numZ + j] = dtInv*uT[((o * numX) + i) * numY + j] - 0.5*v[((o * numX) + i) * numY + j];
    }
}

void rollback_Distributed_7_Final(
    int t,
    const uint outer, 
    const uint numX, 
    const uint numY,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c,
    vector<REAL>& y,
    vector<REAL>& yy,
    vector<REAL>& myResult
) {
    for(uint i=0;i<numX;i++) {
        for (int gidx = 0; gidx < outer; gidx++) {
            // here yy should have size [numY]
            uint numZ = max(numX,numY);
            tridagPar(a,((gidx * numZ) + i) * numZ,b,((gidx * numZ) + i) * numZ,c,((gidx * numZ) + i) * numZ,y,((gidx * numZ) + i) * numZ,numY,myResult, (gidx * numX + i) * numY,yy,((gidx * numZ) + i) * numZ);
        }
    }
}

void rollback_Distributed_1_Final_para(
    int t,
    const uint outer, 
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline, 
    const vector<REAL> myDxxT,
    const vector<REAL> myVarXT,
    const vector<REAL>& myResultT,
    vector<REAL>& u
) {
    //cout << "test 1" << endl;
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < numX; gidx++) {
        uint numZ = max(numX,numY);

        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);

        for (int o = 0; o < outer; o++) {
            for (int j = 0; j < numY; j++) {
                REAL myDxxT0 = myDxxT[0 * numX + gidx];
                REAL myDxxT1 = myDxxT[1 * numX + gidx];
                REAL myDxxT2 = myDxxT[2 * numX + gidx];

                REAL myResultT_low  = myResultT[((o * numY) + j) * numX + gidx - 1];
                REAL myResultT_mid  = myResultT[((o * numY) + j) * numX + gidx];
                REAL myResultT_high = myResultT[((o * numY) + j) * numX + gidx + 1];

                REAL myVarXT_val = 0.5 * myVarXT[((t * numY) + j) * numX + gidx];

                u[((o * numY) + j) * numX + gidx] = dtInv * myResultT_mid;

                if(gidx > 0) { 
                    u[((o * numY) + j) * numX + gidx] += 
                        0.5*( myVarXT_val * myDxxT0 ) * myResultT_low;
                }

                u[((o * numY) + j) * numX + gidx]  +=  
                    0.5*( myVarXT_val * myDxxT1 ) * myResultT_mid;

                if(gidx < numX-1) {
                    u[((o * numY) + j) * numX + gidx] += 
                        0.5*( myVarXT_val * myDxxT2 ) * myResultT_high;
                }
            }
        }
    }
}

void rollback_Distributed_2_Final1_para(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline,
    const vector<REAL> myDyyT,
    const vector<REAL> myVarY,
    vector<REAL>& v,
    vector<REAL>& myResult
) {
    //cout << "test 2" << endl;
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < numY; gidx++) {
        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);


        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                REAL myDyyT0 = myDyyT[0 * numY + gidx];
                REAL myDyyT1 = myDyyT[1 * numY + gidx];
                REAL myDyyT2 = myDyyT[2 * numY + gidx];

                REAL myResult_low  = myResult[((o * numX) + i) * numY + gidx - 1];
                REAL myResult_mid  = myResult[((o * numX) + i) * numY + gidx];
                REAL myResult_high = myResult[((o * numX) + i) * numY + gidx + 1];

                REAL myVarY_val = 0.5 * myVarY[((t * numX) + i) * numY + gidx];

                v[((o * numX) + i) * numY + gidx] = dtInv * myResult_mid;

                if(gidx > 0) { 
                    v[((o * numX) + i) * numY + gidx] += 
                        0.5*( myVarY_val * myDyyT0 ) * myResult_low;
                }

                v[((o * numX) + i) * numY + gidx]  +=  
                    0.5*( myVarY_val * myDyyT1 ) * myResult_mid;

                if(gidx < numY-1) {
                    v[((o * numX) + i) * numY + gidx] += 
                        0.5*( myVarY_val * myDyyT2 ) * myResult_high;
                }
            }
        }
    }
}


void rollback_Distributed_2_Final2_para(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY,
    vector<REAL>& uT,
    vector<REAL>& v
) {
    //cout << "test 2" << endl;
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < numY; gidx++) {
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                uT[((o * numX) + i) * numY + gidx] += v[((o * numX) + i) * numY + gidx];
            }
        }
    }
}


void rollback_Distributed_3_Final_para(
    int t,
    const uint outer,
    const uint numX,
    const uint numY,
    const vector<REAL> myTimeline, 
    const vector<REAL> myDxxT,
    const vector<REAL> myVarXT,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c
) {
    //cout << "test 3" << endl;
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
        uint o = gidx / (numY * numX);
        uint plane_remain = gidx % (numY * numX);
        uint i = plane_remain % numX;
        uint j = plane_remain / numX;

        uint numZ = max(numX,numY);

        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);

        a[((o * numZ) + j) * numZ + i] =       - 0.5*(0.5*myVarXT[((t * numY) + j) * numY + i]*myDxxT[0 * numX + i]);
        b[((o * numZ) + j) * numZ + i] = dtInv - 0.5*(0.5*myVarXT[((t * numY) + j) * numY + i]*myDxxT[1 * numX + i]);
        c[((o * numZ) + j) * numZ + i] =       - 0.5*(0.5*myVarXT[((t * numY) + j) * numY + i]*myDxxT[2 * numX + i]);
    }
}


void rollback_Distributed_4_Final_para(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY,
    vector<REAL>& u,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c,
    vector<REAL>& yy
) {
    //cout << "test 4" << endl;
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < outer * numY; gidx++) {
        uint o = gidx / numY;
        uint j = gidx % numY;
        uint numZ = max(numX,numY);
        // here yy should have size [numX]
        tridagPar(a,((o * numZ) + j) * numZ,b,((o * numZ) + j) * numZ,c,((o * numZ) + j) * numZ,u,((o * numY) + j) * numX,numX,u,((o * numY) + j) * numX,yy,((o * numZ) + j) * numZ);
    }
}

void rollback_Distributed_5_Final_para(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline,
    const vector<REAL> myDyyT,
    const vector<REAL> myVarY,
    vector<REAL>& u,
    vector<REAL>& v,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c
) {
    //cout << "test 5" << endl;
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);

        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);

        a[((o * numZ) + i) * numZ + j] =       - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyyT[0 * numY + j]);
        b[((o * numZ) + i) * numZ + j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyyT[1 * numY + j]);
        c[((o * numZ) + i) * numZ + j] =       - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyyT[2 * numY + j]);
    }
}

void rollback_Distributed_6_Final_para(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const vector<REAL> myTimeline,
    vector<REAL>& uT,
    vector<REAL>& v,
    vector<REAL>& y
) {
    //cout << "test 6" << endl;
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = max(numX,numY);

        REAL dtInv1 = myTimeline[t];
        REAL dtInv2 = myTimeline[t+1];
        REAL dtInv = 1.0/(dtInv2-dtInv1);

        y[((o * numZ) + i) * numZ + j] = dtInv*uT[((o * numX) + i) * numY + j] - 0.5*v[((o * numX) + i) * numY + j];
    }
}

void rollback_Distributed_7_Final_para(
    int t,
    const uint outer, 
    const uint numX, 
    const uint numY,
    vector<REAL>& a,
    vector<REAL>& b,
    vector<REAL>& c,
    vector<REAL>& y,
    vector<REAL>& yy,
    vector<REAL>& myResult
) {
#pragma omp parallel for schedule(static)
    for (int gidx = 0; gidx < outer * numX; gidx++) {
        uint o = gidx / numX;
        uint i = gidx % numX;
        // here yy should have size [numY]
        uint numZ = max(numX,numY);
        tridagPar(a,((o * numZ) + i) * numZ,b,((o * numZ) + i) * numZ,c,((o * numZ) + i) * numZ,y,((o * numZ) + i) * numZ,numY,myResult, (o * numX + i) * numY,yy,((o * numZ) + i) * numZ);
    }
}

void rollback_Distributed_para(const uint outer, const uint numT, 
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
        //cout << "test 1" << endl;
#pragma omp parallel for schedule(static)
        for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
            uint o = gidx / (numX * numY);
            uint plane_remain = gidx % (numX * numY);
            uint i = plane_remain / numY;
            uint j = plane_remain % numY;
            uint numZ = max(numX,numY);
            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
            u[((o * numY) + j) * numX + i] = dtInv*myResult[((o * numX) + i) * numY + j];

            if(i > 0) { 
                u[((o * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                * myDxx[i * 4 + 0] ) 
                                * myResult[((o * numX) + (i-1)) * numY + j];
            }
            u[((o * numY) + j) * numX + i]  +=  0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                            * myDxx[i * 4 + 1] )
                            * myResult[((o * numX) + i) * numY + j];
            if(i < numX-1) {
                u[((o * numY) + j) * numX + i] += 0.5*( 0.5*myVarX[((t * numX) + i) * numY + j]
                                * myDxx[i * 4 + 2] )
                                * myResult[((o * numX) + (i+1)) * numY + j];
            }
        }

        //cout << "test 2" << endl;
#pragma omp parallel for schedule(static)
        for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
            uint o = gidx / (numY * numX);
            uint plane_remain = gidx % (numY * numX);
            uint j = plane_remain / numX;
            uint i = plane_remain % numX;
            uint numZ = max(numX,numY);
            v[((o * numX) + i) * numY + j] = 0.0;

            if(j > 0) {
                v[((o * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                * myDyy[j * 4 + 0] )
                                * myResult[((o * numX) + i) * numY + j - 1];
            }
            v[((o * numX) + i) * numY + j]  += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                * myDyy[j * 4 + 1] )
                                * myResult[((o * numX) + i) * numY + j];
            if(j < numY-1) {
                v[((o * numX) + i) * numY + j] += ( 0.5* myVarY[((t * numX) + i) * numY + j]
                                * myDyy[j * 4 + 2] )
                                * myResult[((o * numX) + i) * numY + j + 1];
            }
            u[((o * numY) + j) * numX + i] += v[((o * numX) + i) * numY + j];
        }

        //cout << "test 3" << endl;
#pragma omp parallel for schedule(static)
        for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
            uint o = gidx / (numY * numX);
            uint plane_remain = gidx % (numY * numX);
            uint j = plane_remain / numX;
            uint i = plane_remain % numX;
            uint numZ = max(numX,numY);
            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
            a[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 0]);
            b[((o * numZ) + j) * numZ + i] = dtInv - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 1]);
            c[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 2]);
        }

        //cout << "test 4" << endl;
        for(uint j=0;j<numY;j++) {
#pragma omp parallel for schedule(static)
            for (int gidx = 0; gidx < outer; gidx++) {
                uint numZ = max(numX,numY);
                // here yy should have size [numX]
                tridagPar(a,((gidx * numZ) + j) * numZ,b,((gidx * numZ) + j) * numZ,c,((gidx * numZ) + j) * numZ,u,((gidx * numY) + j) * numX,numX,u,((gidx * numY) + j) * numX,yy,(gidx * numZ));
            }
        }

        //cout << "test 5" << endl;
#pragma omp parallel for schedule(static)
        for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
            uint o = gidx / (numX * numY);
            uint plane_remain = gidx % (numX * numY);
            uint i = plane_remain / numY;
            uint j = plane_remain % numY;
            uint numZ = max(numX,numY);
            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
            a[((o * numZ) + i) * numZ + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 0]);
            b[((o * numZ) + i) * numZ + j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 1]);
            c[((o * numZ) + i) * numZ + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 2]);
        }

        //cout << "test 6" << endl;
#pragma omp parallel for schedule(static)
        for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
            uint o = gidx / (numX * numY);
            uint plane_remain = gidx % (numX * numY);
            uint i = plane_remain / numY;
            uint j = plane_remain % numY;
            uint numZ = max(numX,numY);
            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
            y[((o * numZ) + i) * numZ + j] = dtInv*u[((o * numY) + j) * numX + i] - 0.5*v[((o * numX) + i) * numY + j];
        }

        //cout << "test 7" << endl;
        for(uint i=0;i<numX;i++) {
#pragma omp parallel for schedule(static)
            for (int gidx = 0; gidx < outer; gidx++) {
                // here yy should have size [numY]
                uint numZ = max(numX,numY);
                tridagPar(a,((gidx * numZ) + i) * numZ,b,((gidx * numZ) + i) * numZ,c,((gidx * numZ) + i) * numZ,y,((gidx * numZ) + i) * numZ,numY,myResult, (gidx * numX + i) * numY,yy,(gidx * numZ));
            }
        }
    }
}

int   run_Distributed(  
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
    vector<REAL> a(outer * numZ * numZ);
    vector<REAL> b(outer * numZ * numZ);
    vector<REAL> c(outer * numZ * numZ);
    vector<REAL> y(outer * numZ * numZ);
    vector<REAL> yy(outer * numZ);

    uint myXindex = 0;
    uint myYindex = 0;

    //cout << "Test1" << endl;
	initGrid_Distributed(s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);

    //cout << "Test2" << endl;
    initOperator_Distributed(numX, myX, myDxx);

    //cout << "Test3" << endl;
    initOperator_Distributed(numY, myY, myDyy);

    //cout << "Test4" << endl;
    setPayoff_Distributed(myX, outer, numX, numY, myResult);
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
    updateParams_Distributed(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
    //cout << "Test6" << endl;
	rollback_Distributed(outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, u, v, a, b, c, y, yy, myResult);
	
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

int   run_Distributed_Parallel(  
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
    vector<REAL> a(outer * numZ * numZ);
    vector<REAL> b(outer * numZ * numZ);
    vector<REAL> c(outer * numZ * numZ);
    vector<REAL> y(outer * numZ * numZ);
    vector<REAL> yy(outer * numZ);

    uint myXindex = 0;
    uint myYindex = 0;

    //cout << "Test1" << endl;
	initGrid_Distributed_para(s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);

    //cout << "Test2" << endl;
    initOperator_Distributed_para(numX, myX, myDxx);

    //cout << "Test3" << endl;
    initOperator_Distributed_para(numY, myY, myDyy);

    //cout << "Test4" << endl;
    setPayoff_Distributed_para(myX, outer, numX, numY, myResult);
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
    updateParams_Distributed_para(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
    //cout << "Test6" << endl;
	rollback_Distributed_para(outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, u, v, a, b, c, y, yy, myResult);
	
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

int   run_Distributed_Separation(  
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
    vector<REAL> a(outer * numZ * numZ);
    vector<REAL> b(outer * numZ * numZ);
    vector<REAL> c(outer * numZ * numZ);
    vector<REAL> y(outer * numZ * numZ);
    vector<REAL> yy(outer * numZ);

    uint myXindex = 0;
    uint myYindex = 0;

    //cout << "Test1" << endl;
	initGrid_Distributed(s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);

    //cout << "Test2" << endl;
    initOperator_Distributed(numX, myX, myDxx);

    //cout << "Test3" << endl;
    initOperator_Distributed(numY, myY, myDyy);

    //cout << "Test4" << endl;
    setPayoff_Distributed(myX, outer, numX, numY, myResult);
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
    updateParams_Distributed(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
    //cout << "Test6" << endl;
    for (int t = 0; t <= numT - 2; t++) {
	    rollback_Distributed_1(t, outer, numX, numY, myTimeline, myDxx, myVarX, u, myResult);
	    rollback_Distributed_2(t, outer, numX, numY, myTimeline, myDyy, myVarY, u, v, myResult);
	    rollback_Distributed_3(t, outer, numX, numY, myTimeline, myDxx, myVarX, a, b, c);
	    rollback_Distributed_4(t, outer, numX, numY, u, a, b, c, yy);
	    rollback_Distributed_5(t, outer, numX, numY, myTimeline, myDyy, myVarY, a, b, c);
	    rollback_Distributed_6(t, outer, numX, numY, myTimeline, u, v, y);
	    rollback_Distributed_7(t, outer, numX, numY, a, b, c, y, yy, myResult);
    }
	
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

int   run_Distributed_Separation_Parallel(  
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
    vector<REAL> a(outer * numZ * numZ);
    vector<REAL> b(outer * numZ * numZ);
    vector<REAL> c(outer * numZ * numZ);
    vector<REAL> y(outer * numZ * numZ);
    vector<REAL> yy(outer * numZ * numZ);

    uint myXindex = 0;
    uint myYindex = 0;

    //cout << "Test1" << endl;
	initGrid_Distributed_para(s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);

    //cout << "Test2" << endl;
    initOperator_Distributed_para(numX, myX, myDxx);

    //cout << "Test3" << endl;
    initOperator_Distributed_para(numY, myY, myDyy);

    //cout << "Test4" << endl;
    setPayoff_Distributed_para(myX, outer, numX, numY, myResult);
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
    updateParams_Distributed_para(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
    //cout << "Test6" << endl;
	for (int t = 0; t <= numT - 2; t++) {
	    rollback_Distributed_1_para(t, outer, numX, numY, myTimeline, myDxx, myVarX, u, myResult);
	    rollback_Distributed_2_para(t, outer, numX, numY, myTimeline, myDyy, myVarY, u, v, myResult);
	    rollback_Distributed_3_para(t, outer, numX, numY, myTimeline, myDxx, myVarX, a, b, c);
	    rollback_Distributed_4_para(t, outer, numX, numY, u, a, b, c, yy);
	    rollback_Distributed_5_para(t, outer, numX, numY, myTimeline, myDyy, myVarY, a, b, c);
	    rollback_Distributed_6_para(t, outer, numX, numY, myTimeline, u, v, y);
	    rollback_Distributed_7_para(t, outer, numX, numY, a, b, c, y, yy, myResult);
    }
	
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

void matTransposeDist(vector<REAL>& A, vector<REAL>& trA, uint planeIndex, int rowsA, int colsA) {
    for(int i = 0; i < rowsA; i++) {
        for(int j = 0; j < colsA; j++) {
            trA[planeIndex + j*rowsA + i] = A[planeIndex + i*colsA + j];
        }
    }
}

void matTransposeDistPlane(vector<REAL>& A, vector<REAL>& trA, int planes, int rowsA, int colsA) {
    for (unsigned i = 0; i < planes; i++) {
        matTransposeDist(A, trA, i * rowsA * colsA, rowsA, colsA);
    }
}

int   run_Distributed_Final(  
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
    vector<REAL> myDxxT(4 * numX);       // [4][numX]
    vector<REAL> myDyyT(4 * numY);       // [4][numY]
    vector<REAL> myResult(outer * numX * numY); // [outer][numX][numY]
    vector<REAL> myResultT(outer * numY * numX); // [outer][numY][numX]
    vector<REAL> myVarX(numT * numX * numY);    // [numT][numX][numY]
    vector<REAL> myVarXT(numT * numY * numX);   // [numT][numY][numX]
    vector<REAL> myVarY(numT * numX * numY);    // [numT][numX][numY]

#if TEST_INIT_CORRECTNESS
    vector<REAL> myResultCopy(outer * numX * numY);
#endif

    uint numZ = max(numX, numY);
    vector<REAL> u(outer * numY * numX);
    vector<REAL> uT(outer * numX * numY);
    vector<REAL> v(outer * numX * numY);
    vector<REAL> a(outer * numZ * numZ);
    vector<REAL> b(outer * numZ * numZ);
    vector<REAL> c(outer * numZ * numZ);
    vector<REAL> y(outer * numZ * numZ);
    vector<REAL> yy(outer * numZ * numZ);

    uint myXindex = 0;
    uint myYindex = 0;

    //cout << "Test1" << endl;
    initMyTimeline_Distributed_Final(t, numT, myTimeline);
	initMyX_Distributed_Final(s0, alpha, t, numX, myX, myXindex);
    initMyY_Distributed_Final(s0, alpha, nu, t, numY, myY, myYindex);

    //cout << "Test2" << endl;
    initOperator_Distributed_T_Final(numX, myX, myDxxT);
    //cout << "Test3" << endl;
    initOperator_Distributed_T_Final(numY, myY, myDyyT);

#if TEST_INIT_CORRECTNESS
    vector<REAL> testMyDxx(numX * 4);
    initOperator_Distributed(numX, myX, testMyDxx);
    for (int i = 0; i < numX; i++) {
        for (int j = 0; j < 4; j++) {
            if (abs(testMyDxx[i * 4 + j] - myDxxT[j * numX + i]) > 0.00001f) {
                cout << "Transpose fail: myDxx[" << i << "][" << j << "] did not match! was " << myDxxT[j * numX + i] << " expected " << testMyDxx[i * 4 + j] << endl;
                return 1;
            }
        }
    }

    vector<REAL> testMyDyy(numY * 4);
    initOperator_Distributed(numY, myY, testMyDyy);
    for (int i = 0; i < numY; i++) {
        for (int j = 0; j < 4; j++) {
            if (abs(testMyDyy[i * 4 + j] - myDyyT[j * numY + i]) > 0.00001f) {
                cout << "Transpose fail: myDyy[" << i << "][" << j << "] did not match! was " << myDyyT[j * numY + i] << " expected " << testMyDyy[i * 4 + j] << endl;
                return 1;
            }
        }
    }

    vector<REAL> testA(2 * 3 * 4); //3 rows, 4 cols
    vector<REAL> testB(2 * 4 * 3); //4 rows, 3 cols
    for (int o = 0; o < 2; o++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                testA[((o * 3) + i) * 4 + j] = (i+1) * (j+1) + o;
                //cout << "testA[" << o << "][" << i << "][" << j << "] = " << (i+1) * (j+1) + o << endl;
            }
        }
    }

    matTransposeDistPlane(testA, testB, 2, 3, 4);
    for (int o = 0; o < 2; o++) {
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 3; i++) {
                //cout << "testB[" << o << "][" << j << "][" << i << "] = " << testB[((o * 4) + j) * 3 + i] << "?= testA[" << o << "][" << i << "][" << j << "] =" << testA[((o * 3) + i) * 4 + j] << endl;
            }
        }
    }
#endif


    //cout << "Test4" << endl;
    setPayoff_Distributed_T_Final(myX, outer, numX, numY, myResultT);
#if TEST_INIT_CORRECTNESS
    vector<REAL> myResultInit(outer * numX * numY);
    matTransposeDistPlane(myResultT, myResultInit, outer, numY, numX);
    for (int o = 0; o < outer; o ++) {
        for (int i = 0; i < numX; i ++) {
            for (int j = 0; j < numY; j ++) {
                myResultCopy[((o * numX) + i) * numY + j] = myResultInit[((o * numX) + i) * numY + j]; 
            }
        }
    }
#endif

    //cout << "Test5" << endl;
    updateParams_Distributed_VarXT_Final(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarXT);
    updateParams_Distributed_VarY_Final(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarY);

#if TEST_INIT_CORRECTNESS
    updateParams_Distributed(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
#endif

    //cout << "Test6" << endl;
	for (int t = 0; t <= numT - 2; t++) {
        cout << "t: " << t << "/" << numT - 3 << endl;
#if TEST_INIT_CORRECTNESS
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                for (int j = 0; j < numY; j++) {
                    //if (abs(test_u[((o * numY) + j) * numX + i] - u[((o * numY) + j) * numX + i]) > 0.0000001f) {
                    if (myResultInit[((o * numX) + i) * numY + j] != myResultT[((o * numY) + j) * numX + i]) {
                        cout << "myresult3 failed! myresult[" << o << "][" << i << "][" << j << "] did not match! was" << endl;
                        return 1;
                    }
                }
            }
        }
#endif
	    rollback_Distributed_1_Final(t, outer, numX, numY, myTimeline, myDxxT, myVarXT, myResultT, u);
#if TEST_INIT_CORRECTNESS
        vector<REAL> test_u(outer * numY * numX);
        rollback_Distributed_1(t, outer, numX, numY, myTimeline, testMyDxx, myVarX, test_u, myResultInit);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                for (int j = 0; j < numY; j++) {
                    //if (abs(test_u[((o * numY) + j) * numX + i] - u[((o * numY) + j) * numX + i]) > 0.0000001f) {
                    if (test_u[((o * numY) + j) * numX + i] != u[((o * numY) + j) * numX + i]) {
                        cout << "u failed! u[" << o << "][" << i << "][" << j << "] did not match! was " << u[((o * numY) + j) * numX + i] << " expected " << test_u[((o * numY) + j) * numX + i] << endl;
                        return 1;
                    }
                }
            }   
        }
#endif
        matTransposeDistPlane(myResultT, myResult, outer, numY, numX);
        //cout << "Test6.2" << endl;
        rollback_Distributed_2_Final1(t, outer, numX, numY, myTimeline, myDyyT, myVarY, u, v, myResult);
        //cout << "Test6.3" << endl;
        matTransposeDistPlane(u, uT, outer, numY, numX);
        //cout << "Test6.5" << endl;
        rollback_Distributed_2_Final2(t, outer, numX, numY, uT, v);
#if TEST_INIT_CORRECTNESS
        vector<REAL> test_v(outer * numX * numY);
        rollback_Distributed_2(t, outer, numX, numY, myTimeline, testMyDyy, myVarY, test_u, test_v, myResultInit);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                for (int j = 0; j < numY; j++) {
                    //if (abs(test_u[((o * numY) + j) * numX + i] - u[((o * numY) + j) * numX + i]) > 0.0000001f) {
                    if (test_u[((o * numY) + j) * numX + i] != u[((o * numY) + j) * numX + i]) {
                        cout << "u2 failed! u[" << o << "][" << j << "][" << i << "] did not match! was " << u[((o * numY) + j) * numX + i] << " expected " << test_u[((o * numY) + j) * numX + i] << endl;
                        return 1;
                    }
                    if (test_v[((o * numX) + i) * numY + j] != v[((o * numX) + i) * numY + j]) {
                        cout << "v failed! v[" << o << "][" << i << "][" << j << "] did not match! was " << u[((o * numX) + i) * numY + j] << " expected " << test_u[((o * numX) + i) * numY + j] << endl;
                        return 1;
                    }
                }
            }   
        }
#endif
        //cout << "Test6.6" << endl;
        matTransposeDistPlane(uT, u, outer, numX, numY);
        //cout << "Test6.7" << endl;
        rollback_Distributed_3_Final(t, outer, numX, numY, myTimeline, myDxxT, myVarXT, a, b, c);
#if TEST_INIT_CORRECTNESS
        vector<REAL> test_a(outer * numZ * numZ);
        vector<REAL> test_b(outer * numZ * numZ);
        vector<REAL> test_c(outer * numZ * numZ);
        rollback_Distributed_3(t, outer, numX, numY, myTimeline, testMyDxx, myVarX, test_a, test_b, test_c);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                for (int j = 0; j < numY; j++) {
                    //if (abs(test_u[((o * numY) + j) * numX + i] - u[((o * numY) + j) * numX + i]) > 0.0000001f) {
                    if (test_a[((o * numY) + j) * numX + i] != a[((o * numY) + j) * numX + i]) {
                        cout << "a failed! a[" << o << "][" << j << "][" << i << "] did not match! was " << u[((o * numY) + j) * numX + i] << " expected " << test_u[((o * numY) + j) * numX + i] << endl;
                        return 1;
                    }
                    if (test_b[((o * numY) + j) * numX + i] != b[((o * numY) + j) * numX + i]) {
                        cout << "b failed! b[" << o << "][" << j << "][" << i << "] did not match! was " << u[((o * numX) + i) * numY + j] << " expected " << test_u[((o * numX) + i) * numY + j] << endl;
                        return 1;
                    }
                    if (test_c[((o * numY) + j) * numX + i] != c[((o * numY) + j) * numX + i]) {
                        cout << "c failed! c[" << o << "][" << j << "][" << i << "] did not match! was " << u[((o * numX) + i) * numY + j] << " expected " << test_u[((o * numX) + i) * numY + j] << endl;
                        return 1;
                    }
                }
            }   
        }
#endif
        //cout << "Test6.8" << endl;
        rollback_Distributed_4_Final(t, outer, numX, numY, u, a, b, c, yy);
#if TEST_INIT_CORRECTNESS
        rollback_Distributed_4(t, outer, numX, numY, test_u, a, b, c, yy);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                for (int j = 0; j < numY; j++) {
                    //if (abs(test_u[((o * numY) + j) * numX + i] - u[((o * numY) + j) * numX + i]) > 0.0000001f) {
                    if (test_u[((o * numY) + j) * numX + i] != u[((o * numY) + j) * numX + i]) {
                        cout << "u3 failed! u[" << o << "][" << i << "][" << j << "] did not match! was " << u[((o * numY) + j) * numX + i] << " expected " << test_u[((o * numY) + j) * numX + i] << endl;
                        return 1;
                    }
                }
            }   
        }
#endif
        //cout << "Test6.9" << endl;
        rollback_Distributed_5_Final(t, outer, numX, numY, myTimeline, myDyyT, myVarY, a, b, c);
#if TEST_INIT_CORRECTNESS
        rollback_Distributed_5(t, outer, numX, numY, myTimeline, testMyDyy, myVarY, test_a, test_b, test_c);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                for (int j = 0; j < numY; j++) {
                    //if (abs(test_u[((o * numY) + j) * numX + i] - u[((o * numY) + j) * numX + i]) > 0.0000001f) {
                    if (test_a[((o * numX) + i) * numY + j] != a[((o * numX) + i) * numY + j]) {
                        cout << "a failed! a[" << o << "][" << i << "][" << j << "] did not match! was " << u[((o * numY) + j) * numX + i] << " expected " << test_u[((o * numY) + j) * numX + i] << endl;
                        return 1;
                    }
                    if (test_b[((o * numX) + i) * numY + j] != b[((o * numX) + i) * numY + j]) {
                        cout << "b failed! b[" << o << "][" << i << "][" << j << "] did not match! was " << u[((o * numX) + i) * numY + j] << " expected " << test_u[((o * numX) + i) * numY + j] << endl;
                        return 1;
                    }
                    if (test_c[((o * numX) + i) * numY + j] != c[((o * numX) + i) * numY + j]) {
                        cout << "c failed! c[" << o << "][" << i << "][" << j << "] did not match! was " << u[((o * numX) + i) * numY + j] << " expected " << test_u[((o * numX) + i) * numY + j] << endl;
                        return 1;
                    }
                }
            }   
        }
#endif
        matTransposeDistPlane(u, uT, outer, numY, numX);
        //cout << "Test6.10" << endl;
        rollback_Distributed_6_Final(t, outer, numX, numY, myTimeline, uT, v, y);
#if TEST_INIT_CORRECTNESS
        vector<REAL> test_y(outer * numZ * numZ);
        rollback_Distributed_6(t, outer, numX, numY, myTimeline, test_u, test_v, test_y);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                for (int j = 0; j < numY; j++) {
                    //if (abs(test_u[((o * numY) + j) * numX + i] - u[((o * numY) + j) * numX + i]) > 0.0000001f) {
                    if (test_y[((o * numX) + i) * numY + j] != y[((o * numX) + i) * numY + j]) {
                        cout << "a failed! a[" << o << "][" << i << "][" << j << "] did not match! was " << u[((o * numY) + j) * numX + i] << " expected " << test_u[((o * numY) + j) * numX + i] << endl;
                        return 1;
                    }
                }
            }   
        }
#endif
        //cout << "Test6.11" << endl;
        rollback_Distributed_7_Final(t, outer, numX, numY, a, b, c, y, yy, myResult);
        //cout << "Test6.12" << endl;
#if TEST_INIT_CORRECTNESS
        rollback_Distributed_7(t, outer, numX, numY, test_a, test_b, test_c, test_y, yy, myResultInit);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                for (int j = 0; j < numY; j++) {
                    //if (abs(test_u[((o * numY) + j) * numX + i] - u[((o * numY) + j) * numX + i]) > 0.0000001f) {
                    if (test_myResult[((o * numX) + i) * numY + j] != myResult[((o * numX) + i) * numY + j]) {
                        cout << "myresult failed! myresult[" << o << "][" << i << "][" << j << "] did not match! was" << endl;
                        return 1;
                    }
                }
            }
        }
#endif
        //cout << "Test6.13" << endl;
        matTransposeDistPlane(myResult, myResultT, outer, numX, numY);
#if TEST_INIT_CORRECTNESS
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < numX; i++) {
                for (int j = 0; j < numY; j++) {
                    //if (abs(test_u[((o * numY) + j) * numX + i] - u[((o * numY) + j) * numX + i]) > 0.0000001f) {
                    if (test_myResult[((o * numX) + i) * numY + j] != myResultT[((o * numY) + j) * numX + i]) {
                        cout << "myresult2 failed! myresult[" << o << "][" << i << "][" << j << "] did not match! was" << endl;
                        return 1;
                    }
                }
            }
        }
#endif
        //cout << "Test6.14" << endl;
    }
	
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

    vector<REAL> myDxx(numX * 4);
    matTransposeDist(myDxxT, myDxx, 0, 4, numX);
    initOperator_Interchanged(numX, TestmyX, TestmyDxx);
    for (int i = 0; i < numX; i ++) {
        for (int j = 0; j < 4; j ++) {
            if (abs(myDxx[i * 4 + j] - TestmyDxx[i][j]) > 0.00001f) {
                cout << "myDxx[" << i << "][" << j << "] did not match! was " << myDxx[i * 4 + j] << " expected " << TestmyDxx[i][j] << endl;
                return 1;
            }
        }
    }

    vector<REAL> myDyy(numY * 4);
    matTransposeDist(myDyyT, myDyy, 0, 4, numY);
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
    
    matTransposeDistPlane(myVarXT, myVarX, numT, numY, numX);
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

#endif