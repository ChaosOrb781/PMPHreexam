#ifndef INTERCHANGED_ALGORITHM
#define INTERCHANGED_ALGORITHM

#include "OriginalAlgorithm.h"
#include <omp.h>

void initGrid_Alt(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
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

void initGrid_Alt_para(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
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

void initOperator_Alt(  const uint& numZ, const vector<REAL>& myZ, 
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

void initOperator_Alt_para(  const uint& numZ, const vector<REAL>& myZ, 
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
#pragma omp parallel for schedule(static)
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

void updateParams_Alt(const REAL alpha, const REAL beta, const REAL nu,
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

void updateParams_Alt_para(const REAL alpha, const REAL beta, const REAL nu,
    const uint numX, const uint numY, const uint numT, 
    const vector<REAL> myX, const vector<REAL> myY, const vector<REAL> myTimeline,
    vector<vector<vector<REAL> > >& myVarX, vector<vector<vector<REAL> > >& myVarY)
{
#pragma omp parallel for schedule(static) collapse(3)
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

void setPayoff_Alt(const vector<REAL> myX, const uint outer,
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

void setPayoff_Alt_para(const vector<REAL> myX, const uint outer,
    const uint numX, const uint numY,
    vector<vector< vector<REAL> > >& myResult)
{
#pragma omp parallel for schedule(static) collapse(2)
    for(uint i = 0; i < outer; i++) {
        for(uint j = 0; j < numX; j++)
        {
            REAL payoff = max(myX[j]-0.001*(REAL)i, (REAL)0.0);
            for(uint k = 0; k < numY; k++)
                myResult[i][j][k] = payoff;
        }
    }
}

void rollback_Alt(const uint outer, const uint numT, 
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

void rollback_Alt_para(const uint outer, const uint numT, 
    const uint numX, const uint numY, 
    const vector<REAL> myTimeline, 
    const vector<vector<REAL> > myDxx,
    const vector<vector<REAL> > myDyy,
    const vector<vector<vector<REAL> > > myVarX,
    const vector<vector<vector<REAL> > > myVarY,
    vector<vector< vector<REAL> > >& myResult 
) {
    for (int t = 0; t <= numT - 2; t++) {
#pragma omp parallel for schedule(static)
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

int   run_Interchanged(  
                const uint   outer,
                const uint   numX,
                const uint   numY,
                const uint   numT,
                const REAL   s0,
                const REAL   t, 
                const REAL   alpha, 
                const REAL   nu, 
                const REAL   beta,
                      REAL*  res   // [outer] RESULT
) {
	vector<PrivGlobs> globstastic;
	globstastic.resize(outer); //Generates list from default constructor
    
	for( unsigned i = 0; i < outer; ++ i ) {
		//Initialize each object as if called by the size constructor
		globstastic[i].Initialize(numX, numY, numT);
		initGrid(s0,alpha,nu,t, numX, numY, numT, globstastic[i]);
		initOperator(globstastic[i].myX,globstastic[i].myDxx);
		initOperator(globstastic[i].myY,globstastic[i].myDyy);
		REAL strike = 0.001*i;
		setPayoff(strike, globstastic[i]);
	}
	for(int j = 0;j<=numT-2;++j) {
		for( unsigned i = 0; i < outer; ++ i ) {
			{
				updateParams(j,alpha,beta,nu,globstastic[i]);
				rollback(j, globstastic[i]);
			}
		}
    }
	for( unsigned i = 0; i < outer; ++ i ) {
        res[i] = globstastic[i].myResult[globstastic[i].myXindex][globstastic[i].myYindex];
    }
    return 1;
}

int   run_InterchangedAlternative(  
                const uint   outer,
                const uint   numX,
                const uint   numY,
                const uint   numT,
                const REAL   s0,
                const REAL   t, 
                const REAL   alpha, 
                const REAL   nu, 
                const REAL   beta,
                      REAL*  res   // [outer] RESULT
) {
	vector<REAL>                   myX(numX);       // [numX]
    vector<REAL>                   myY(numY);       // [numY]
    vector<REAL>                   myTimeline(numT);// [numT]
    vector<vector<REAL> >          myDxx(numX);     // [numX][4]
    vector<vector<REAL> >          myDyy(numY);     // [numY][4]
    vector<vector<vector<REAL> > > myResult(outer); // [outer][numX][numY]
    vector<vector<vector<REAL> > > myVarX(numT);    // [numT][numX][numY]
    vector<vector<vector<REAL> > > myVarY(numT);    // [numT][numX][numY]

    uint myXindex = 0;
    uint myYindex = 0;

    for (int i = 0; i < numX; i++) {
        myDxx[i].resize(4);
    }
    for (int i = 0; i < numY; i++) {
        myDyy[i].resize(4);
    }
    for (int i = 0; i < outer; i++) {
        myResult[i].resize(numX);
        for (int j = 0; j < numX; j++) {
            myResult[i][j].resize(numY);
        }
    }
    for (int i = 0; i < numT; i++) {
        myVarX[i].resize(numX);
        myVarY[i].resize(numX);
        for (int j = 0; j < numX; j++) {
            myVarX[i][j].resize(numY);
            myVarY[i][j].resize(numY);
        }
    }

	initGrid_Alt(s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);
	initOperator_Alt(numX, myX, myDxx);
    initOperator_Alt(numY, myY, myDyy);
    setPayoff_Alt(myX, outer, numX, numY, myResult);

    updateParams_Alt(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);
	rollback_Alt(outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, myResult);
	
	for(uint i = 0; i < outer; i++) {
        res[i] = myResult[i][myXindex][myYindex];
    }
    return 1;
}



int   run_InterchangedParallel(  
                const uint   outer,
                const uint   numX,
                const uint   numY,
                const uint   numT,
                const REAL   s0,
                const REAL   t, 
                const REAL   alpha, 
                const REAL   nu, 
                const REAL   beta,
                      REAL*  res   // [outer] RESULT
) {
	int procs = 0;
	
	vector<PrivGlobs> globstastic;
	globstastic.resize(outer); //Generates list from default constructor
#pragma omp parallel for
	for( unsigned i = 0; i < outer; ++ i ) {
		//Initialize each object as if called by the size constructor
		globstastic[i].Initialize(numX, numY, numT);
		initGrid(s0,alpha,nu,t, numX, numY, numT, globstastic[i]);
		initOperator(globstastic[i].myX,globstastic[i].myDxx);
		initOperator(globstastic[i].myY,globstastic[i].myDyy);
		REAL strike = 0.001*i;
		setPayoff(strike, globstastic[i]);
	}
	for(int j = 0;j<=numT-2;++j) {
#pragma omp parallel for
		for( unsigned i = 0; i < outer; ++ i ) {
			updateParams(j,alpha,beta,nu,globstastic[i]);
			rollback(j, globstastic[i]);
		}
    }
#pragma omp parallel for
	for( unsigned i = 0; i < outer; ++ i ) {
		{
            int th_id = omp_get_thread_num();
            if(th_id == 0) { procs = omp_get_num_threads(); }
        }
        res[i] = globstastic[i].myResult[globstastic[i].myXindex][globstastic[i].myYindex];
    }
    return procs;
}


int   run_InterchangedParallelAlternative(  
                const uint   outer,
                const uint   numX,
                const uint   numY,
                const uint   numT,
                const REAL   s0,
                const REAL   t, 
                const REAL   alpha, 
                const REAL   nu, 
                const REAL   beta,
                      REAL*  res   // [outer] RESULT
) {
	int procs = 0;
	
		vector<REAL>                   myX(numX);       // [numX]
    vector<REAL>                   myY(numY);       // [numY]
    vector<REAL>                   myTimeline(numT);// [numT]
    vector<vector<REAL> >          myDxx(numX);     // [numX][4]
    vector<vector<REAL> >          myDyy(numY);     // [numY][4]
    vector<vector<vector<REAL> > > myResult(outer); // [outer][numX][numY]
    vector<vector<vector<REAL> > > myVarX(numT);    // [numT][numX][numY]
    vector<vector<vector<REAL> > > myVarY(numT);    // [numT][numX][numY]

    uint myXindex = 0;
    uint myYindex = 0;

    for (int i = 0; i < numX; i++) {
        myDxx[i].resize(4);
    }
    for (int i = 0; i < numY; i++) {
        myDyy[i].resize(4);
    }
    for (int i = 0; i < outer; i++) {
        myResult[i].resize(numX);
        for (int j = 0; j < numX; j++) {
            myResult[i][j].resize(numY);
        }
    }
    for (int i = 0; i < numT; i++) {
        myVarX[i].resize(numX);
        myVarY[i].resize(numX);
        for (int j = 0; j < numX; j++) {
            myVarX[i][j].resize(numY);
            myVarY[i][j].resize(numY);
        }
    }

	initGrid_Alt_para(s0, alpha, nu, t, numX, numY, numT, myX, myY, myTimeline, myXindex, myYindex);
	initOperator_Alt_para(numX, myX, myDxx);
    initOperator_Alt_para(numY, myY, myDyy);
    
    //left off from here!
    setPayoff_Alt_para(myX, outer, numX, numY, myResult);

    updateParams_Alt_para(alpha, beta, nu, numX, numY, numT, myX, myY, myTimeline, myVarX, myVarY);

	rollback_Alt_para(outer, numT, numX, numY, myTimeline, myDxx, myDyy, myVarX, myVarY, myResult);

#pragma omp parallel for schedule(static)
	for(uint i = 0; i < outer; ++ i ) {
		{
            int th_id = omp_get_thread_num();
            if(th_id == 0) { procs = omp_get_num_threads(); }
        }
        res[i] = myResult[i][myXindex][myYindex];
    }
    return procs;
}

#endif