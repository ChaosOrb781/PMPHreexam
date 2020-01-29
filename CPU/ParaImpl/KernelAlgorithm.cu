#ifndef KERNEL_ALGORITHM
#define KERNEL_ALGORITHM
#include "CUDAKernels.cu"
#include "Constants.h"
#include "TridagPar.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

using namespace thrust;

#define TEST_INIT_CORRECTNESS false

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

void rollback_Control_1(
    int t,
    const uint outer, 
    const uint numX, 
    const uint numY, 
    const host_vector<REAL> myTimeline, 
    const host_vector<REAL> myDxx,
    const host_vector<REAL> myVarX,
    host_vector<REAL>& u,
    host_vector<REAL>& myResult
) {
    //cout << "test 1" << endl;
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        //uint numZ = numX > numY ? numX : numY;
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
void rollback_Control_2(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const host_vector<REAL> myTimeline,
    const host_vector<REAL> myDyy,
    const host_vector<REAL> myVarY,
    host_vector<REAL>& u,
    host_vector<REAL>& v,
    host_vector<REAL>& myResult
) {
    //cout << "test 2" << endl;
    for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
        uint o = gidx / (numY * numX);
        uint plane_remain = gidx % (numY * numX);
        uint j = plane_remain / numX;
        uint i = plane_remain % numX;
        //uint numZ = numX > numY ? numX : numY;
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

void rollback_Control_3(
    int t,
    const uint outer,
    const uint numX,
    const uint numY,
    const host_vector<REAL> myTimeline, 
    const host_vector<REAL> myDxx,
    const host_vector<REAL> myVarX,
    host_vector<REAL>& a,
    host_vector<REAL>& b,
    host_vector<REAL>& c
) {
    //cout << "test 3" << endl;
    for (int gidx = 0; gidx < outer * numY * numX; gidx++) {
        uint o = gidx / (numY * numX);
        uint plane_remain = gidx % (numY * numX);
        uint j = plane_remain / numX;
        uint i = plane_remain % numX;
        uint numZ = numX > numY ? numX : numY;
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        a[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 0]);
        b[((o * numZ) + j) * numZ + i] = dtInv - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 1]);
        c[((o * numZ) + j) * numZ + i] =		 - 0.5*(0.5*myVarX[((t * numX) + i) * numY + j]*myDxx[i * 4 + 2]);
    }
}


void rollback_Control_4(
    int t,
    int j,
    const uint outer,
    const uint numX, 
    const uint numY,
    host_vector<REAL>& u,
    host_vector<REAL>& a,
    host_vector<REAL>& b,
    host_vector<REAL>& c,
    host_vector<REAL>& yy
) {
    //cout << "test 4" << endl;
    for (int gidx = 0; gidx < outer; gidx++) {
        uint numZ = numX > numY ? numX : numY;
        // here yy should have size [numX]
        tridagPar(a,((gidx * numZ) + j) * numZ,b,((gidx * numZ) + j) * numZ,c,((gidx * numZ) + j) * numZ,u,((gidx * numY) + j) * numX,numX,u,((gidx * numY) + j) * numX,yy,(gidx * numZ));
    }
}

void rollback_Control_5(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const host_vector<REAL> myTimeline,
    const host_vector<REAL> myDyy,
    const host_vector<REAL> myVarY,
    host_vector<REAL>& u,
    host_vector<REAL>& v,
    host_vector<REAL>& a,
    host_vector<REAL>& b,
    host_vector<REAL>& c
) {
    //cout << "test 5" << endl;
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = numX > numY ? numX : numY;
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        a[((o * numZ) + i) * numZ + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 0]);
        b[((o * numZ) + i) * numZ + j] = dtInv - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 1]);
        c[((o * numZ) + i) * numZ + j] =		 - 0.5*(0.5*myVarY[((t * numX) + i) * numY + j]*myDyy[j * 4 + 2]);
    }
}

void rollback_Control_6(
    int t,
    const uint outer,
    const uint numX, 
    const uint numY, 
    const host_vector<REAL> myTimeline,
    host_vector<REAL>& u,
    host_vector<REAL>& v,
    host_vector<REAL>& y
) {
    //cout << "test 6" << endl;
    for (int gidx = 0; gidx < outer * numX * numY; gidx++) {
        uint o = gidx / (numX * numY);
        uint plane_remain = gidx % (numX * numY);
        uint i = plane_remain / numY;
        uint j = plane_remain % numY;
        uint numZ = numX > numY ? numX : numY;
        REAL dtInv = 1.0/(myTimeline[t+1]-myTimeline[t]);
        y[((o * numZ) + i) * numZ + j] = dtInv*u[((o * numY) + j) * numX + i] - 0.5*v[((o * numX) + i) * numY + j];
    }
}

void rollback_Control_7(
    int t,
    int i,
    const uint outer, 
    const uint numX, 
    const uint numY,
    host_vector<REAL>& a,
    host_vector<REAL>& b,
    host_vector<REAL>& c,
    host_vector<REAL>& y,
    host_vector<REAL>& yy,
    host_vector<REAL>& myResult
) {
    for (int gidx = 0; gidx < outer; gidx++) {
        // here yy should have size [numY]
        uint numZ = numX > numY ? numX : numY;
        tridagPar(a,((gidx * numZ) + i) * numZ,b,((gidx * numZ) + i) * numZ,c,((gidx * numZ) + i) * numZ,y,((gidx * numZ) + i) * numZ,numY,myResult, (gidx * numX + i) * numY,yy,(gidx * numZ));
    }
}

void initGrid_Kernel(
    const int blocksize, 
    const REAL s0, 
    const REAL alpha, 
    const REAL nu,
    const REAL t, 
    const uint numX, 
    const uint numY, 
    const uint numT,
    device_vector<REAL>& myX, 
    device_vector<REAL>& myY, 
    device_vector<REAL>& myTimeline,
    uint& myXindex, 
    uint& myYindex
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

void initOperator_Kernel(
    const int blocksize, 
    const uint numZ, 
    device_vector<REAL>& myZ, 
    device_vector<REAL>& Dzz
) {
    REAL* myZ_p = raw_pointer_cast(&myZ[0]);
    REAL* Dzz_p = raw_pointer_cast(&Dzz[0]);
    uint num_blocks= (numZ * 4 + blocksize - 1) / blocksize;
    InitMyDzz<<<num_blocks, blocksize>>>(numZ, myZ_p, Dzz_p);
}

void updateParams_Kernel(
    const int blocksize, 
    const REAL alpha, 
    const REAL beta, 
    const REAL nu,
    const uint numX, 
    const uint numY, 
    const uint numT, 
    device_vector<REAL>& myX, 
    device_vector<REAL>& myY, 
    device_vector<REAL>& myTimeline,
    device_vector<REAL>& myVarX, 
    device_vector<REAL>& myVarY
){
    REAL* myX_p = raw_pointer_cast(&myX[0]);
    REAL* myY_p = raw_pointer_cast(&myY[0]);
    REAL* myTimeline_p = raw_pointer_cast(&myTimeline[0]);
    REAL* myVarX_p = raw_pointer_cast(&myVarX[0]);
    REAL* myVarY_p = raw_pointer_cast(&myVarY[0]);
    uint num_blocks= (numT * numX * numY + blocksize - 1) / blocksize;
    InitParams<<<num_blocks, blocksize>>>(numT, numX, numY, alpha, beta, nu, myX_p, myY_p, myTimeline_p, myVarX_p, myVarY_p);
}

void setPayoff_Kernel(
    const int blocksize, 
    device_vector<REAL>& myX, 
    const uint outer,
    const uint numX, 
    const uint numY,
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

        host_vector<REAL> a_h(a);
        host_vector<REAL> b_h(b);
        host_vector<REAL> c_h(c);
        host_vector<REAL> u_h(u);
        host_vector<REAL> yy_h(yy);
        for (int j = 0; j < numY; j++) {
            for (int o = 0; o < outer; o++) {
                tridagPar(
                    a_h, ((o * numZ) + j) * numZ,
                    b_h, ((o * numZ) + j) * numZ,
                    c_h, ((o * numZ) + j) * numZ,
                    u_h, ((o * numY) + j) * numX,
                    numX,
                    u_h, ((o * numY) + j) * numX,
                    yy_h, o * numZ
                );
            };
        }
        thrust::copy(a_h.begin(), a_h.end(), a.begin());
        thrust::copy(b_h.begin(), b_h.end(), b.begin());
        thrust::copy(c_h.begin(), c_h.end(), c.begin());
        thrust::copy(u_h.begin(), u_h.end(), u.begin());
        thrust::copy(yy_h.begin(), yy_h.end(), yy.begin());

        Rollback_5<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, myDyy_p, myVarY_p, u_p, v_p, a_p, b_p, c_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());

        Rollback_6<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, u_p, v_p, y_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());

        thrust::copy(a.begin(), a.end(), a_h.begin());
        thrust::copy(b.begin(), b.end(), b_h.begin());
        thrust::copy(c.begin(), c.end(), c_h.begin());
        host_vector<REAL> y_h(y);
        host_vector<REAL> myResult_h(myResult);
        thrust::copy(yy.begin(), yy.end(), yy_h.begin());
        for (int i = 0; i < numX; i++) {
            for (int o = 0; o < outer; o++) {
                tridagPar(
                    a_h, ((o * numZ) + i) * numZ, 
                    b_h, ((o * numZ) + i) * numZ, 
                    c_h, ((o * numZ) + i) * numZ,
                    y_h, ((o * numZ) + i) * numZ,   numY,
                    myResult_h, ((o * numX) + i) * numY,
                    yy_h, o * numZ
                );
            }
        } 
        thrust::copy(a_h.begin(), a_h.end(), a.begin());
        thrust::copy(b_h.begin(), b_h.end(), b.begin());
        thrust::copy(c_h.begin(), c_h.end(), c.begin());
        thrust::copy(y_h.begin(), y_h.end(), y.begin());
        thrust::copy(myResult_h.begin(), myResult_h.end(), myResult.begin());
        thrust::copy(yy_h.begin(), yy_h.end(), yy.begin());
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

//Do some individual tests of each kernel against sequential solution!
void rollback_Kernel_Test(
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

    host_vector<REAL> myTimeline_h = myTimeline;
    host_vector<REAL> myDxx_h      = myDxx;
    host_vector<REAL> myDyy_h      = myDyy;
    host_vector<REAL> myVarX_h     = myVarX;
    host_vector<REAL> myVarY_h     = myVarY;
    host_vector<REAL> u_h          = u;
    host_vector<REAL> v_h          = v;
    host_vector<REAL> a_h          = a;
    host_vector<REAL> b_h          = b;
    host_vector<REAL> c_h          = c;
    host_vector<REAL> y_h          = y;
    host_vector<REAL> yy_h         = yy;
    host_vector<REAL> myResult_h   = myResult;

    uint numZ = numX > numY ? numX : numY;

    for (int t = 0; t <= numT - 2; t++) {
        uint num_blocks = (outer * numX * numY + blocksize - 1) / blocksize;
        Rollback_1<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, myDxx_p, myVarX_p, u_p, myResult_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());
        rollback_Control_1(t, outer, numX, numY, myTimeline_h, myDxx_h, myVarX_h, u_h, myResult_h);
        host_vector<REAL> temp(u);
        for (int i = 0; i < outer * numY * numX; i++) {
            if (std::abs(temp[i] - u_h[i]) > 0.0001) {
                cout << "Rollback 1 index " << i << " failed to be equal, got " << u_h[i] << " expected " << temp[i];
                exit(0);
            }
        }


        Rollback_2<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, myDyy_p, myVarY_p, u_p, v_p, myResult_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());
        rollback_Control_2(t, outer, numX, numY, myTimeline_h, myDyy_h, myVarY_h, u_h, v_h, myResult_h);
        temp = u;
        host_vector<REAL> temp2(v);
        for (int i = 0; i < outer * numX * numY; i++) {
            if (std::abs(temp2[i] - v_h[i]) > 0.0001) {
                cout << "Rollback 2(v) index " << i << " failed to be equal, got " << v_h[i] << " expected " << temp2[i];
                exit(0);
            }
            if (std::abs(temp[i] - u_h[i]) > 0.0001) {
                cout << "Rollback 2(u) index " << i << " failed to be equal, got " << u_h[i] << " expected " << temp[i];
                exit(0);
            }
        }


        Rollback_3<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, myDxx_p, myVarX_p, a_p, b_p, c_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());
        rollback_Control_3(t, outer, numX, numY, myTimeline_h, myDxx_h, myVarX_h, a_h, b_h, c_h);
        temp = a;
        temp2 = b;
        host_vector<REAL> temp3(c);
        for (int i = 0; i < outer * numZ * numZ; i++) {
            if (std::abs(temp[i] - a_h[i]) > 0.0001) {
                cout << "Rollback 3(a) index " << i << " failed to be equal, got " << a_h[i] << " expected " << temp[i];
                exit(0);
            }
            if (std::abs(temp2[i] - b_h[i]) > 0.0001) {
                cout << "Rollback 3(b) index " << i << " failed to be equal, got " << b_h[i] << " expected " << temp2[i];
                exit(0);
            }
            if (std::abs(temp3[i] - c_h[i]) > 0.0001) {
                cout << "Rollback 3(c) index " << i << " failed to be equal, got " << c_h[i] << " expected " << temp3[i];
                exit(0);
            }
        }


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
            rollback_Control_4(t, j, outer, numX, numY, u_h, a_h, b_h, c_h, yy_h);
            temp = u;
            temp2 = yy;
            for (int i = 0; i < outer * numZ * numZ; i++) {
                if (std::abs(temp[i] - u_h[i]) > 0.0001) {
                    cout << "Rollback 2(u) index " << i << " failed to be equal, got " << u_h[i] << " expected " << temp[i];
                    exit(0);
                }
            }
            for (int i = 0; i < outer * numZ; i++) {
                if (std::abs(temp2[i] - yy_h[i]) > 0.0001) {
                    cout << "Rollback 2(yy) index " << i << " failed to be equal, got " << yy_h[i] << " expected " << temp2[i];
                    exit(0);
                }
            }
            //cudaDeviceSynchronize();
            //gpuErr(cudaPeekAtLastError());
        }

        Rollback_5<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, myDyy_p, myVarY_p, u_p, v_p, a_p, b_p, c_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());
        rollback_Control_5(t, outer, numX, numY, myTimeline_h, myDyy_h, myVarY_h, u_h, v_h, a_h, b_h, c_h);
        temp = a;
        temp2 = b;
        temp3 = c;
        for (int i = 0; i < outer * numZ * numZ; i++) {
            if (std::abs(temp[i] - a_h[i]) > 0.0001) {
                cout << "Rollback 5(a) index " << i << " failed to be equal, got " << a_h[i] << " expected " << temp[i];
                exit(0);
            }
            if (std::abs(temp2[i] - b_h[i]) > 0.0001) {
                cout << "Rollback 5(b) index " << i << " failed to be equal, got " << b_h[i] << " expected " << temp2[i];
                exit(0);
            }
            if (std::abs(temp3[i] - c_h[i]) > 0.0001) {
                cout << "Rollback 5(c) index " << i << " failed to be equal, got " << c_h[i] << " expected " << temp3[i];
                exit(0);
            }
        }

        Rollback_6<<<num_blocks, blocksize>>>(t, outer, numX, numY, myTimeline_p, u_p, v_p, y_p);
        cudaDeviceSynchronize();
        gpuErr(cudaPeekAtLastError());
        rollback_Control_6(t, outer, numX, numY, myTimeline_h, u_h, v_h, y_h);
        temp = y;
        for (int i = 0; i < outer * numZ * numZ; i++) {
            if (std::abs(temp[i] - y_h[i]) > 0.0001) {
                cout << "Rollback 6 index " << i << " failed to be equal, got " << y_h[i] << " expected " << temp[i];
                exit(0);
            }
        }

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
            rollback_Control_7(t, i, outer, numX, numY, a_h, b_h, c_h, y_h, yy_h, myResult_h);
            temp = myResult;
            for (int j = 0; j < outer * numX * numY; j++) {
                if (std::abs(temp[j] - myResult_h[j]) > 0.0001) {
                    cout << "Rollback 7 index " << j << " failed to be equal, got " << myResult_h[j] << " expected " << temp[j];
                    exit(0);
                }
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
    
    int sgm_size = 8;

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

    //cout << "Test1" << endl;
    initGrid_Kernel(blocksize, s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    //cout << "Test2" << endl;
    initOperator_Kernel(blocksize, numX, myX, myDxx);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    //cout << "Test3" << endl;
    initOperator_Kernel(blocksize, numY, myY, myDyy);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    //cout << "Test4" << endl;
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

    //cout << "Test5" << endl;
    updateParams_Kernel(blocksize, alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    //cout << "Test6" << endl;
    //rollback_Kernel_CPU(blocksize, outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, u, v, a, b, c, y, yy, myResult);
    rollback_Kernel_Test(blocksize, sgm_size, outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, u, v, a, b, c, y, yy, myResult);
	cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    host_vector<REAL> myResult_h(outer*numX*numY);
    thrust::copy(myResult.begin(), myResult.end(), myResult_h.begin());

    //cout << "Test7" << endl;
	for(uint i = 0; i < outer; i++) {
        res[i] = myResult_h[((i * numX) + myXindex) * numY + myYindex];
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

    host_vector<REAL> myResult_h(outer*numX*numY);
    thrust::copy(myResult.begin(), myResult.end(), myResult_h.begin());

    cout << "Test7" << endl;
	for(uint i = 0; i < outer; i++) {
        res[i] = myResult_h[((i * numX) + myXindex) * numY + myYindex];
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