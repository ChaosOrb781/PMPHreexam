#ifndef INTERCHANGED_ALGORITHM
#define INTERCHANGED_ALGORITHM

#include "OriginalAlgorithm.h"
#include <omp.h>

void setPayoff_Alt(const REAL strike, PrivGlobs& globs, vector< vector<REAL> >& myResult)
{
	for(unsigned i=0;i<globs.myX.size();++i)
	{
		REAL payoff = max(globs.myX[i]-strike, (REAL)0.0);
		for(unsigned j=0;j<globs.myY.size();++j)
			myResult[i][j] = payoff;
	}
}

void
rollback_Alt( const unsigned g, PrivGlobs& globs, vector< vector<REAL> >& myResult ) {
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();

    unsigned numZ = max(numX,numY);

    unsigned i, j;

    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
    vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
    vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
    vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

    //	explicit x
    for(i=0;i<numX;i++) {
        for(j=0;j<numY;j++) {
            u[j][i] = dtInv*globs.myResult[i][j];

            if(i > 0) { 
              u[j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][0] ) 
                            * myResult[i-1][j];
            }
            u[j][i]  +=  0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][1] )
                            * myResult[i][j];
            if(i < numX-1) {
              u[j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][2] )
                            * myResult[i+1][j];
            }
        }
    }

    //	explicit y
    for(j=0;j<numY;j++)
    {
        for(i=0;i<numX;i++) {
            v[i][j] = 0.0;

            if(j > 0) {
              v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][0] )
                         *  myResult[i][j-1];
            }
            v[i][j]  +=   ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][1] )
                         *  myResult[i][j];
            if(j < numY-1) {
              v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][2] )
                         *  myResult[i][j+1];
            }
            u[j][i] += v[i][j]; 
        }
    }

    //	implicit x
    for(j=0;j<numY;j++) {
        for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
            a[i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][0]);
            b[i] = dtInv - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][1]);
            c[i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][2]);
        }
        // here yy should have size [numX]
        tridagPar(a,b,c,u[j],numX,u[j],yy);
    }

    //	implicit y
    for(i=0;i<numX;i++) { 
        for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
            a[j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][0]);
            b[j] = dtInv - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][1]);
            c[j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][2]);
        }

        for(j=0;j<numY;j++)
            y[j] = dtInv*u[j][i] - 0.5*v[i][j];

        // here yy should have size [numY]
        tridagPar(a,b,c,y,numY,myResult[i],yy);
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
	PrivGlobs constantGlobs(numX, numY, numT);
	//Only non-constant throughout parallel operations for each sequential iteration
	vector < vector< vector<REAL> > > myResult;
	myResult.resize(outer);
	for (unsigned i = 0; i < outer; ++i ) {
		myResult[i].resize(numX);
		for (unsigned j = 0; j < numX; ++j) {
			myResult[i][j].resize(numY);
		}
	}

	initGrid(s0,alpha,nu,t, numX, numY, numT, constantGlobs);
	initOperator(constantGlobs.myX,constantGlobs.myDxx);
	initOperator(constantGlobs.myY,constantGlobs.myDyy);

	for ( unsigned i = 0; i < outer; ++ i ) {
		REAL strike = 0.001*i;
		setPayoff_Alt(strike, constantGlobs, myResult[i]);
	}
	for ( int j = 0; j <= numT-2; ++ j ) {
		updateParams(j,alpha,beta,nu,constantGlobs);
		for( unsigned i = 0; i < outer; ++ i ) {
			rollback_Alt(j, constantGlobs, myResult[i]);
		}
    }
	for( unsigned i = 0; i < outer; ++ i ) {
        res[i] = myResult[i][constantGlobs.myXindex][constantGlobs.myYindex];
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

/*
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
	
	PrivGlobs constantGlobs(numX, numY, numT);
	//Only non-constant throughout parallel operations for each sequential iteration
	vector<vector<REAL>> myResult;
	myResult.resize(numX);
	for (unsigned i = 0; i < numX; ++i) {
		myResult[i].resize(numY);
	}

	initGrid(s0,alpha,nu,t, numX, numY, numT, constantGlobs);
	initOperator(constantGlobs.myX,constantGlobs.myDxx);
	initOperator(constantGlobs.myY,constantGlobs.myDyy);
	
#pragma omp parallel for
	for ( unsigned i = 0; i < outer; ++ i ) {
		REAL strike = 0.001*i;
		setPayoff_Alt(strike, globstastic[i]);
	}
	for ( int j = 0; j <= numT-2; ++ j ) {
		updateParams(j,alpha,beta,nu,constantGlobs);
#pragma omp parallel for
		for( unsigned i = 0; i < outer; ++ i ) {
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
}*/

#endif