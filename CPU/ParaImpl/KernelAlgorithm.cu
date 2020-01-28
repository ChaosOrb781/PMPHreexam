#ifndef KERNEL_ALGORITHM
#define KERNEL_ALGORITHM
#include "CUDAKernels.cu"
#include "Constants.h"
#include "TridagPar.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace thrust;

#define TEST_INIT_CORRECTNESS true

#define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      cout << "GPUassert: " << cudaGetErrorString(code) << " in " << file << " : "<< line << endl;
      if (abort) exit(code);
   }
}

void initGrid_Control(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
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

void initOperator_Control(  const uint& numZ, const vector<REAL>& myZ, 
                        vector<vector<REAL> >& Dzz
) {
	REAL dl, du;
	//	lower boundary
	dl		 =  0.0;
	du		 =  myZ[1] - myZ[0];
	
	Dzz[0][0] =  0.0;
	Dzz[0][1] =  0.0;
	Dzz[0][2] =  0.0;
    Dzz[0][3] =  0.0;
	
	//	standard case
	for(unsigned i=1;i<numZ;i++)
	{
		dl      = myZ[i]   - myZ[i-1];
		du      = myZ[i+1] - myZ[i];

		Dzz[i][0] =  2.0/dl/(dl+du);
		Dzz[i][1] = -2.0*(1.0/dl + 1.0/du)/(dl+du);
		Dzz[i][2] =  2.0/du/(dl+du);
        Dzz[i][3] =  0.0; 
	}

	//	upper boundary
	dl		   =  myZ[numZ-1] - myZ[numZ-2];
	du		   =  0.0;

	Dzz[numZ-1][0] = 0.0;
	Dzz[numZ-1][1] = 0.0;
	Dzz[numZ-1][2] = 0.0;
    Dzz[numZ-1][3] = 0.0;
}

void updateParams_Control(const REAL alpha, const REAL beta, const REAL nu,
    const uint numX, const uint numY, const uint numT, 
    const vector<REAL> myX, const vector<REAL> myY, const vector<REAL> myTimeline,
    vector<vector<vector<REAL> > >& myVarX, vector<vector<vector<REAL> > >& myVarY)
{
    for(uint i = 0; i < numT; i++)
        for(uint j = 0; j < numX; j++)
            for(uint k = 0; k < numY; k++) {
                myVarX[i][j][k] = exp(2.0*(  beta*log(myX[j])   
                                            + myY[k]             
                                            - 0.5*nu*nu*myTimeline[i] )
                                        );
                myVarY[i][j][k] = exp(2.0*(  alpha*log(myX[j])   
                                            + myY[k]             
                                            - 0.5*nu*nu*myTimeline[i] )
                                        ); // nu*nu
            }
}

void setPayoff_Control(const vector<REAL> myX, const uint outer,
    const uint numX, const uint numY,
    vector<vector< vector<REAL> > >& myResult)
{
    for(uint i = 0; i < outer; i++) {
        for(uint j = 0; j < numX; j++)
        {
            REAL payoff = max(myX[j]-0.001*(REAL)i, (REAL)0.0);
            for(uint k = 0; k < numY; k++)
                myResult[i][j][k] = payoff;
        }
    }
}

void rollback_Control(const uint outer, const uint numT, 
    const uint numX, const uint numY, 
    const vector<REAL> myTimeline, 
    const vector<vector<REAL> > myDxx,
    const vector<vector<REAL> > myDyy,
    const vector<vector<vector<REAL> > > myVarX,
    const vector<vector<vector<REAL> > > myVarY,
    vector<vector< vector<REAL> > >& myResult 
) {
    for (int t = 0; t <= numT - 2; t++) {
        for (int o = 0; o < outer; o++) {
            uint numZ = max(numX,numY);

            uint i, j;

            REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);

            vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
            vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
            vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
            vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

            //	explicit x
            for(i=0;i<numX;i++) {
                for(j=0;j<numY;j++) {
                    u[j][i] = dtInv*myResult[o][i][j];

                    if(i > 0) { 
                    u[j][i] += 0.5*( 0.5*myVarX[t][i][j]*myDxx[i][0] ) 
                                    * myResult[o][i-1][j];
                    }
                    u[j][i]  +=  0.5*( 0.5*myVarX[t][i][j]*myDxx[i][1] )
                                    * myResult[o][i][j];
                    if(i < numX-1) {
                    u[j][i] += 0.5*( 0.5*myVarX[t][i][j]*myDxx[i][2] )
                                    * myResult[o][i+1][j];
                    }
                }
            }

            //	explicit y
            for(j=0;j<numY;j++)
            {
                for(i=0;i<numX;i++) {
                    v[i][j] = 0.0;

                    if(j > 0) {
                    v[i][j] +=  ( 0.5*myVarY[t][i][j]*myDyy[j][0] )
                                *  myResult[o][i][j-1];
                    }
                    v[i][j]  +=   ( 0.5*myVarY[t][i][j]*myDyy[j][1] )
                                *  myResult[o][i][j];
                    if(j < numY-1) {
                    v[i][j] +=  ( 0.5*myVarY[t][i][j]*myDyy[j][2] )
                                *  myResult[o][i][j+1];
                    }
                    u[j][i] += v[i][j]; 
                }
            }

            //	implicit x
            for(j=0;j<numY;j++) {
                for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
                    a[i] =		 - 0.5*(0.5*myVarX[t][i][j]*myDxx[i][0]);
                    b[i] = dtInv - 0.5*(0.5*myVarX[t][i][j]*myDxx[i][1]);
                    c[i] =		 - 0.5*(0.5*myVarX[t][i][j]*myDxx[i][2]);
                }
                // here yy should have size [numX]
                tridagPar(a,b,c,u[j],numX,u[j],yy);
            }

            //	implicit y
            for(i=0;i<numX;i++) { 
                for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
                    a[j] =		 - 0.5*(0.5*myVarY[t][i][j]*myDyy[j][0]);
                    b[j] = dtInv - 0.5*(0.5*myVarY[t][i][j]*myDyy[j][1]);
                    c[j] =		 - 0.5*(0.5*myVarY[t][i][j]*myDyy[j][2]);
                }

                for(j=0;j<numY;j++)
                    y[j] = dtInv*u[j][i] - 0.5*v[i][j];

                // here yy should have size [numY]
                tridagPar(a,b,c,y,numY,myResult[o][i],yy);
            }
        }
    }
}

void initGrid_Kernel(const int blocksize, const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const uint numX, const uint numY, const uint numT,
                device_vector<REAL>& myX, device_vector<REAL>& myY, device_vector<REAL>& myTimeline,
                uint& myXindex, uint& myYindex
) {
    REAL* myTimeline_p = raw_pointer_cast(&myTimeline[0]);
    uint num_blocks = (numT + blocksize - 1) / blocksize;
    InitMyTimeline<<<num_blocks, blocksize>>>(numT, t, myTimeline_p);

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    myXindex = static_cast<unsigned>(s0/dx) % numX;

    REAL* myX_p = raw_pointer_cast(&myX[0]);
    num_blocks = (numX + blocksize - 1) / blocksize;
    InitMyX<<<num_blocks, blocksize>>>(numX, myXindex, s0, dx, myX_p);

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    myYindex = static_cast<unsigned>(numY/2.0);

    REAL* myY_p = raw_pointer_cast(&myY[0]);
    num_blocks = (numY + blocksize - 1) / blocksize;
    InitMyY<<<num_blocks, blocksize>>>(numY, myYindex, logAlpha, dy, myY_p);
}

void initOperator_Kernel(const int blocksize, const uint numZ, device_vector<REAL>& myZ, 
                        device_vector<REAL>& Dzz
) {
    REAL* myZ_p = raw_pointer_cast(&myZ[0]);
    REAL* Dzz_p = raw_pointer_cast(&Dzz[0]);
    uint num_blocks= (numZ * 4 + blocksize - 1) / blocksize;
    InitMyDzz<<<num_blocks, blocksize>>>(numZ, myZ_p, Dzz_p);
}

void updateParams_Kernel(const int blocksize, const REAL alpha, const REAL beta, const REAL nu,
    const uint numX, const uint numY, const uint numT, 
    device_vector<REAL>& myX, device_vector<REAL>& myY, device_vector<REAL>& myTimeline,
    device_vector<REAL>& myVarX, device_vector<REAL>& myVarY)
{
    REAL* myX_p = raw_pointer_cast(&myX[0]);
    REAL* myY_p = raw_pointer_cast(&myY[0]);
    REAL* myTimeline_p = raw_pointer_cast(&myTimeline[0]);
    REAL* myVarX_p = raw_pointer_cast(&myVarX[0]);
    REAL* myVarY_p = raw_pointer_cast(&myVarY[0]);
    uint num_blocks= (numT * numX * numY + blocksize - 1) / blocksize;
    InitParams<<<num_blocks, blocksize>>>(numT, numX, numY, alpha, beta, nu, myX_p, myY_p, myTimeline_p, myVarX_p, myVarY_p);
}

void setPayoff_Kernel(const int blocksize, device_vector<REAL>& myX, const uint outer,
    const uint numX, const uint numY,
    device_vector<REAL>& myResult)
{
    REAL* myX_p = raw_pointer_cast(&myX[0]);
    REAL* myResult_p = raw_pointer_cast(&myResult[0]);
    uint num_blocks = (outer * numX * numY + blocksize - 1) / blocksize;
    InitMyResult<<<num_blocks, blocksize>>>(outer, numX, numY, myX_p, myResult_p);
}

void rollback_Kernel_CPU(
    const int blocksize, 
    const uint outer, 
    const uint numT, 
    const uint numX, 
    const uint numY, 
    device_vector<REAL>& myTimeline, 
    device_vector<REAL>& myDxx,
    device_vector<REAL>& myDyy,
    device_vector<REAL>& myVarX,
    device_vector<REAL>& myVarY,
    device_vector<REAL>& u,
    device_vector<REAL>& v,
    device_vector<REAL>& a,
    device_vector<REAL>& b,
    device_vector<REAL>& c,
    device_vector<REAL>& y,
    device_vector<REAL>& yy,
    device_vector<REAL>& myResult
) {
    REAL* myTimeline_p = raw_pointer_cast(&myTimeline[0]);
    REAL* myDxx_p = raw_pointer_cast(&myDxx[0]);
    REAL* myDyy_p = raw_pointer_cast(&myDyy[0]);
    REAL* myVarX_p = raw_pointer_cast(&myVarX[0]);
    REAL* myVarY_p = raw_pointer_cast(&myVarY[0]);
    REAL* u_p = raw_pointer_cast(&u[0]);
    REAL* v_p = raw_pointer_cast(&v[0]);
    REAL* a_p = raw_pointer_cast(&a[0]);
    REAL* b_p = raw_pointer_cast(&b[0]);
    REAL* c_p = raw_pointer_cast(&c[0]);
    REAL* y_p = raw_pointer_cast(&y[0]);
    REAL* yy_p = raw_pointer_cast(&yy[0]);
    REAL* myResult_p = raw_pointer_cast(&myResult[0]);

    uint numZ = numX > numY ? numX : numY;

    for (int t = 0; t <= numT - 2; t++) {
        /*
        uint num_blocks = (outer * numX * numY + blocksize - 1) / blocksize;
        Rollback_1<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, myDxx_p, myVarX_p, u_p, myResult_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());

        Rollback_2<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, myDyy_p, myVarY_p, u_p, v_p, myResult_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());

        Rollback_3<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, myDxx_p, myVarX_p, a_p, b_p, c_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());

        cout << "Rollback tridagpar 1" << endl;
        for (int j = 0; j < numY; j++) {
            cout << "t: " << t << endl;
            for (int o = 0; o < outer; o++) {
                cout << "o: " << t << endl;
                tridagPar(
                    &a_p[((o * numZ) + j) * numZ], 
                    &b_p[((o * numZ) + j) * numZ], 
                    &c_p[((o * numZ) + j) * numZ],
                    &u_p[((o * numY) + j) * numX],
                    numX,
                    &u_p[((o * numY) + j) * numX],
                    &yy_p[o * numZ]
                );
            };
        }

        Rollback_5<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, myDyy_p, myVarY_p, u_p, v_p, a_p, b_p, c_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());

        Rollback_6<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, u_p, v_p, y_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());

        cout << "Rollback tridagpar 2" << endl;
        for (int i = 0; i < numX; i++) {
            for (int o = 0; o < outer; o++) {
                tridagPar(
                    &a_p[((o * numZ) + i) * numZ], 
                    &b_p[((o * numZ) + i) * numZ], 
                    &c_p[((o * numZ) + i) * numZ],
                    &y_p[((o * numZ) + i) * numZ],
                    numY,
                    &myResult_p[((o * numX) + i) * numY],
                    &yy_p[o * numZ]
                );
            }
        }*/

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

void rollback_Kernel_GPU(
    const int blocksize, 
    const int sgm_size, 
    const uint outer, 
    const uint numT, 
    const uint numX, 
    const uint numY, 
    device_vector<REAL>& myTimeline, 
    device_vector<REAL>& myDxx,
    device_vector<REAL>& myDyy,
    device_vector<REAL>& myVarX,
    device_vector<REAL>& myVarY,
    device_vector<REAL>& u,
    device_vector<REAL>& v,
    device_vector<REAL>& a,
    device_vector<REAL>& b,
    device_vector<REAL>& c,
    device_vector<REAL>& y,
    device_vector<REAL>& yy,
    device_vector<REAL>& myResult
) {
    REAL* myTimeline_p = raw_pointer_cast(&myTimeline[0]);
    REAL* myDxx_p = raw_pointer_cast(&myDxx[0]);
    REAL* myDyy_p = raw_pointer_cast(&myDyy[0]);
    REAL* myVarX_p = raw_pointer_cast(&myVarX[0]);
    REAL* myVarY_p = raw_pointer_cast(&myVarY[0]);
    REAL* u_p = raw_pointer_cast(&u[0]);
    REAL* v_p = raw_pointer_cast(&v[0]);
    REAL* a_p = raw_pointer_cast(&a[0]);
    REAL* b_p = raw_pointer_cast(&b[0]);
    REAL* c_p = raw_pointer_cast(&c[0]);
    REAL* y_p = raw_pointer_cast(&y[0]);
    REAL* yy_p = raw_pointer_cast(&yy[0]);
    REAL* myResult_p = raw_pointer_cast(&myResult[0]);

    uint numZ = numX > numY ? numX : numY;

    for (int t = 0; t <= numT - 2; t++) {
        uint num_blocks = (outer * numX * numY + blocksize - 1) / blocksize;
        Rollback_1<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, myDxx_p, myVarX_p, u_p, myResult_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());

        Rollback_2<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, myDyy_p, myVarY_p, u_p, v_p, myResult_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());

        Rollback_3<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, myDxx_p, myVarX_p, a_p, b_p, c_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());

        uint num_blocks2 = (outer + blocksize - 1) / blocksize;
        if((blocksize % sgm_size)!=0) {
            printf("Invalid segment or block size. Exiting!\n\n!");
            exit(0);
        }
        if((numX % sgm_size)!=0) {
            printf("Invalid total size (not a multiple of segment size). Exiting!\n\n!");
            exit(0);
        }
        for (int j = 0; j < numY; j++) {
            //cout << "t: " << t << endl;
            for (int o = 0; o < outer; o++) {
                TRIDAG_SOLVER<<<num_blocks2, blocksize, sizeof(MyReal4_ker) * blocksize + sizeof(MyReal2_ker) * blocksize + sizeof(int) * blocksize>>>(
                    &a_p[((o * numZ) + j) * numZ], 
                    &b_p[((o * numZ) + j) * numZ], 
                    &c_p[((o * numZ) + j) * numZ],
                    &u_p[((o * numY) + j) * numX],
                    numX,
                    sgm_size,
                    &u_p[((o * numY) + j) * numX],
                    &yy_p[o * numZ]
                );
                cudaDeviceSynchronize();
                gpuErr(cudaPeekAtLastError());
            }
            //cudaDeviceSynchronize();
            //gpuErr(cudaPeekAtLastError());
        }

        Rollback_5<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, myDyy_p, myVarY_p, u_p, v_p, a_p, b_p, c_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());

        Rollback_6<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, u_p, v_p, y_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());

        if((blocksize % sgm_size)!=0) {
            printf("Invalid segment or block size. Exiting!\n\n!");
            exit(0);
        }
        if((numY % sgm_size)!=0) {
            printf("Invalid total size (not a multiple of segment size). Exiting!\n\n!");
            exit(0);
        }
        for (int i = 0; i < numX; i++) {
            for (int o = 0; o < outer; o++) {
                TRIDAG_SOLVER<<<num_blocks2, blocksize, sizeof(MyReal4_ker) * blocksize + sizeof(MyReal2_ker) * blocksize + sizeof(int) * blocksize>>>(
                    &a_p[((o * numZ) + i) * numZ], 
                    &b_p[((o * numZ) + i) * numZ], 
                    &c_p[((o * numZ) + i) * numZ],
                    &y_p[((o * numZ) + i) * numZ],
                    numY,
                    sgm_size,
                    &myResult_p[((o * numX) + i) * numY],
                    &yy_p[o * numZ]
                );
                cudaDeviceSynchronize();
                gpuErr(cudaPeekAtLastError());
            }
        }
    }
}

int run_CPUKernel(
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
    int procs = blocksize;

    cout << "initializing device memory" << endl;
	device_vector<REAL> myX(numX);       // [numX]
    device_vector<REAL> myY(numY);       // [numY]
    device_vector<REAL> myTimeline(numT);// [numT]
    device_vector<REAL> myDxx(numX * 4);     // [numX][4]
    device_vector<REAL> myDyy(numY * 4);     // [numY][4]
    device_vector<REAL> myResult(outer * numX * numY); // [outer][numX][numY]
    device_vector<REAL> myVarX(numT * numX * numY);    // [numT][numX][numY]
    device_vector<REAL> myVarY(numT * numX * numY);    // [numT][numX][numY]

#if TEST_INIT_CORRECTNESS
    vector<REAL> myResultCopy(outer * numX * numY);
#endif

    cout << "initializing rollback memory" << endl;
    uint numZ = std::max(numX, numY);
    device_vector<REAL> u(outer * numY * numX);
    device_vector<REAL> v(outer * numX * numY);
    device_vector<REAL> a(outer * numZ * numZ);
    device_vector<REAL> b(outer * numZ * numZ);
    device_vector<REAL> c(outer * numZ * numZ);
    device_vector<REAL> y(outer * numZ * numZ);
    device_vector<REAL> yy(outer * numZ);

    uint myXindex = 0;
    uint myYindex = 0;

    cout << "Test1" << endl;
    initGrid_Kernel(blocksize, s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cout << "Test2" << endl;
    initOperator_Kernel(blocksize, numX, myX, myDxx);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cout << "Test3" << endl;
    initOperator_Kernel(blocksize, numY, myY, myDyy);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cout << "Test4" << endl;
    setPayoff_Kernel(blocksize, myX, outer, numX, numY, myResult);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
#if TEST_INIT_CORRECTNESS
    for (int o = 0; o < outer; o ++) {
        for (int i = 0; i < numX; i ++) {
            for (int j = 0; j < numY; j ++) {
                myResultCopy[((o * numX) + i) * numY + j] = myResult[((o * numX) + i) * numY + j]; 
            }
        }
    }
#endif

    cout << "Test5" << endl;
    updateParams_Kernel(blocksize, alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cout << "Test6" << endl;
	rollback_Kernel_CPU(blocksize, outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, u, v, a, b, c, y, yy, myResult);
	cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cout << "Test7" << endl;
	for(uint i = 0; i < outer; i++) {
        res[i] = myResult[((i * numX) + myXindex) * numY + myYindex];
    }

#if TEST_INIT_CORRECTNESS
    vector<REAL>                   TestmyX(numX);       // [numX]
    vector<REAL>                   TestmyY(numY);       // [numY]
    vector<REAL>                   TestmyTimeline(numT);// [numT]
    vector<vector<REAL> >          TestmyDxx(numX, vector<REAL>(4));     // [numX][4]
    vector<vector<REAL> >          TestmyDyy(numY, vector<REAL>(4));     // [numY][4]
    vector<vector<vector<REAL> > > TestmyResult(outer, vector<vector<REAL> >(numX, vector<REAL>(numY))); // [outer][numX][numY]
    vector<vector<vector<REAL> > > TestmyVarX(numT, vector<vector<REAL> >(numX, vector<REAL>(numY)));    // [numT][numX][numY]
    vector<vector<vector<REAL> > > TestmyVarY(numT, vector<vector<REAL> >(numX, vector<REAL>(numY)));    // [numT][numX][numY]

    initGrid_Control(s0, alpha, nu, t, numX, numY, numT, TestmyX, TestmyY, TestmyTimeline, myXindex, myYindex);
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

    initOperator_Control(numX, TestmyX, TestmyDxx);
    for (int i = 0; i < numX; i ++) {
        for (int j = 0; j < 4; j ++) {
            if (abs(myDxx[i * 4 + j] - TestmyDxx[i][j]) > 0.00001f) {
                cout << "myDxx[" << i << "][" << j << "] did not match! was " << myDxx[i * 4 + j] << " expected " << TestmyDxx[i][j] << endl;
                return procs;
            }
        }
    }

    initOperator_Control(numY, TestmyY, TestmyDyy);
    for (int i = 0; i < numY; i ++) {
        for (int j = 0; j < 4; j ++) {
            if (abs(myDyy[i * 4 + j] - TestmyDyy[i][j]) > 0.00001f) {
                cout << "myDyy[" << i << "][" << j << "] did not match! was " << myDyy[i * 4 + j] << " expected " << TestmyDyy[i][j] << endl;
                return procs;
            }
        }
    }

    setPayoff_Control(TestmyX, outer, numX, numY, TestmyResult);
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

    updateParams_Control(alpha, beta, nu, numX, numY, numT, TestmyX, TestmyY, TestmyTimeline, TestmyVarX, TestmyVarY);
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

int run_GPUKernel(
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
    int procs = blocksize;

    int sgm_size = 8;

    cout << "initializing device memory" << endl;
	device_vector<REAL> myX(numX);       // [numX]
    device_vector<REAL> myY(numY);       // [numY]
    device_vector<REAL> myTimeline(numT);// [numT]
    device_vector<REAL> myDxx(numX * 4);     // [numX][4]
    device_vector<REAL> myDyy(numY * 4);     // [numY][4]
    device_vector<REAL> myResult(outer * numX * numY); // [outer][numX][numY]
    device_vector<REAL> myVarX(numT * numX * numY);    // [numT][numX][numY]
    device_vector<REAL> myVarY(numT * numX * numY);    // [numT][numX][numY]

#if TEST_INIT_CORRECTNESS
    vector<REAL> myResultCopy(outer * numX * numY);
#endif

    cout << "initializing rollback memory" << endl;
    uint numZ = std::max(numX, numY);
    device_vector<REAL> u(outer * numY * numX);
    device_vector<REAL> v(outer * numX * numY);
    device_vector<REAL> a(outer * numZ * numZ);
    device_vector<REAL> b(outer * numZ * numZ);
    device_vector<REAL> c(outer * numZ * numZ);
    device_vector<REAL> y(outer * numZ * numZ);
    device_vector<REAL> yy(outer * numZ);

    uint myXindex = 0;
    uint myYindex = 0;

    cout << "Test1" << endl;
    initGrid_Kernel(blocksize, s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cout << "Test2" << endl;
    initOperator_Kernel(blocksize, numX, myX, myDxx);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cout << "Test3" << endl;
    initOperator_Kernel(blocksize, numY, myY, myDyy);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cout << "Test4" << endl;
    setPayoff_Kernel(blocksize, myX, outer, numX, numY, myResult);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
#if TEST_INIT_CORRECTNESS
    for (int o = 0; o < outer; o ++) {
        for (int i = 0; i < numX; i ++) {
            for (int j = 0; j < numY; j ++) {
                myResultCopy[((o * numX) + i) * numY + j] = myResult[((o * numX) + i) * numY + j]; 
            }
        }
    }
#endif

    cout << "Test5" << endl;
    updateParams_Kernel(blocksize, alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cout << "Test6" << endl;
	rollback_Kernel_GPU(blocksize, sgm_size, outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, u, v, a, b, c, y, yy, myResult);
	cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cout << "Test7" << endl;
	for(uint i = 0; i < outer; i++) {
        res[i] = myResult[((i * numX) + myXindex) * numY + myYindex];
    }

#if TEST_INIT_CORRECTNESS
    vector<REAL>                   TestmyX(numX);       // [numX]
    vector<REAL>                   TestmyY(numY);       // [numY]
    vector<REAL>                   TestmyTimeline(numT);// [numT]
    vector<vector<REAL> >          TestmyDxx(numX, vector<REAL>(4));     // [numX][4]
    vector<vector<REAL> >          TestmyDyy(numY, vector<REAL>(4));     // [numY][4]
    vector<vector<vector<REAL> > > TestmyResult(outer, vector<vector<REAL> >(numX, vector<REAL>(numY))); // [outer][numX][numY]
    vector<vector<vector<REAL> > > TestmyVarX(numT, vector<vector<REAL> >(numX, vector<REAL>(numY)));    // [numT][numX][numY]
    vector<vector<vector<REAL> > > TestmyVarY(numT, vector<vector<REAL> >(numX, vector<REAL>(numY)));    // [numT][numX][numY]

    initGrid_Control(s0, alpha, nu, t, numX, numY, numT, TestmyX, TestmyY, TestmyTimeline, myXindex, myYindex);
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

    initOperator_Control(numX, TestmyX, TestmyDxx);
    for (int i = 0; i < numX; i ++) {
        for (int j = 0; j < 4; j ++) {
            if (abs(myDxx[i * 4 + j] - TestmyDxx[i][j]) > 0.00001f) {
                cout << "myDxx[" << i << "][" << j << "] did not match! was " << myDxx[i * 4 + j] << " expected " << TestmyDxx[i][j] << endl;
                return procs;
            }
        }
    }

    initOperator_Control(numY, TestmyY, TestmyDyy);
    for (int i = 0; i < numY; i ++) {
        for (int j = 0; j < 4; j ++) {
            if (abs(myDyy[i * 4 + j] - TestmyDyy[i][j]) > 0.00001f) {
                cout << "myDyy[" << i << "][" << j << "] did not match! was " << myDyy[i * 4 + j] << " expected " << TestmyDyy[i][j] << endl;
                return procs;
            }
        }
    }

    setPayoff_Control(TestmyX, outer, numX, numY, TestmyResult);
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

    updateParams_Control(alpha, beta, nu, numX, numY, numT, TestmyX, TestmyY, TestmyTimeline, TestmyVarX, TestmyVarY);
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