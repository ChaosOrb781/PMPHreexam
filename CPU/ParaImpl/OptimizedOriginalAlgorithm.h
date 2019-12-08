#include "OriginalAlgorithm.h"

/* Indifferent 
void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs& globs   
) {
    for(unsigned i=0;i<numT;++i)
        globs.myTimeline[i] = t*i/(numT-1);

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    globs.myXindex = static_cast<unsigned>(s0/dx) % numX;

    for(unsigned i=0;i<numX;++i)
        globs.myX[i] = i*dx - globs.myXindex*dx + s0;

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    globs.myYindex = static_cast<unsigned>(numY/2.0);

    for(unsigned i=0;i<numY;++i)
        globs.myY[i] = i*dy - globs.myYindex*dy + logAlpha;
}*/

void initOperator_opti(  const vector<REAL>& x, 
                    vector<vector<REAL> >& Dxx
) {
	const unsigned n = x.size();

	REAL dxl, dxu;

	//	lower boundary
	dxl		 =  0.0;
	dxu		 =  x[1] - x[0];
	
	Dxx[0][0] =  0.0;
	Dxx[0][1] =  0.0;
	Dxx[0][2] =  0.0;
    Dxx[0][3] =  0.0;
	
	//	standard case
	for(unsigned i=1;i<n-1;i++)
	{
		dxl      = x[i]   - x[i-1];
		dxu      = x[i+1] - x[i];

		Dxx[i][0] =  0.5*(dxl*(dxl+dxu));
		Dxx[i][1] =  (-2.0/dxl - 2.0/dxu)/(dxl+dxu);
		Dxx[i][2] =  0.5*(dxu*(dxl+dxu));
        Dxx[i][3] =  0.0; 
	}

	//	upper boundary
	dxl		   =  x[n-1] - x[n-2];
	dxu		   =  0.0;

	Dxx[n-1][0] = 0.0;
	Dxx[n-1][1] = 0.0;
	Dxx[n-1][2] = 0.0;
    Dxx[n-1][3] = 0.0;
}

void updateParams_opti(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{
    for(unsigned i=0;i<globs.myX.size();++i)
        for(unsigned j=0;j<globs.myY.size();++j) {
            globs.myVarX[i][j] = exp( 2.0 * beta*log(globs.myX[i])   
                                    + 2.0 * globs.myY[j])            
                                    - nu*nu*globs.myTimeline[g];
            globs.myVarY[i][j] = exp(2.0 * alpha*log(globs.myX[i])   
                                    + 2.0*globs.myY[j])             
                                    - nu*nu*globs.myTimeline[g];
        }
}

/* Indifferent
void setPayoff(const REAL strike, PrivGlobs& globs )
{
	for(unsigned i=0;i<globs.myX.size();++i)
	{
		REAL payoff = max(globs.myX[i]-strike, (REAL)0.0);
		for(unsigned j=0;j<globs.myY.size();++j)
			globs.myResult[i][j] = payoff;
	}
}*/

void rollback_opti( const unsigned g, PrivGlobs& globs ) {
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
              u[j][i] += 0.25*globs.myVarX[i][j]*globs.myDxx[i][0]
                            * globs.myResult[i-1][j];
            }
            u[j][i]  +=  0.25*globs.myVarX[i][j]*globs.myDxx[i][1]
                            * globs.myResult[i][j];
            if(i < numX-1) {
              u[j][i] += 0.25*globs.myVarX[i][j]*globs.myDxx[i][2]
                            * globs.myResult[i+1][j];
            }
        }
    }

    //	explicit y
    for(j=0;j<numY;j++)
    {
        for(i=0;i<numX;i++) {
            v[i][j] = 0.0;

            if(j > 0) {
              v[i][j] += 0.5*globs.myVarY[i][j]*globs.myDyy[j][0]
                         *  globs.myResult[i][j-1];
            }
            v[i][j]  +=  0.5*globs.myVarY[i][j]*globs.myDyy[j][1]
                         *  globs.myResult[i][j];
            if(j < numY-1) {
              v[i][j] += 0.5*globs.myVarY[i][j]*globs.myDyy[j][2]
                         *  globs.myResult[i][j+1];
            }
            u[j][i] += v[i][j]; 
        }
    }

    //	implicit x
    for(j=0;j<numY;j++) {
        for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
            a[i] =		 - 0.25*globs.myVarX[i][j]*globs.myDxx[i][0];
            b[i] = dtInv - 0.25*globs.myVarX[i][j]*globs.myDxx[i][1];
            c[i] =		 - 0.25*globs.myVarX[i][j]*globs.myDxx[i][2];
        }
        // here yy should have size [numX]
        tridagPar(a,b,c,u[j],numX,u[j],yy);
    }

    //	implicit y
    for(i=0;i<numX;i++) { 
        for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
            a[j] =		 - 0.25*globs.myVarY[i][j]*globs.myDyy[j][0];
            b[j] = dtInv - 0.25*globs.myVarY[i][j]*globs.myDyy[j][1];
            c[j] =		 - 0.25*globs.myVarY[i][j]*globs.myDyy[j][2];
        }

        for(j=0;j<numY;j++)
            y[j] = dtInv*u[j][i] - 0.5*v[i][j];

        // here yy should have size [numY]
        tridagPar(a,b,c,y,numY,globs.myResult[i],yy);
    }
}

REAL   value_opti(   PrivGlobs    globs,
                const REAL s0,
                const REAL strike, 
                const REAL t, 
                const REAL alpha, 
                const REAL nu, 
                const REAL beta,
                const unsigned int numX,
                const unsigned int numY,
                const unsigned int numT
) {	
    initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
    initOperator_opti(globs.myX,globs.myDxx);
    initOperator_opti(globs.myY,globs.myDyy);

    setPayoff(strike, globs);
    for(int i = globs.myTimeline.size()-2;i>=0;--i)
    {
        updateParams_opti(i,alpha,beta,nu,globs);
        rollback_opti(i, globs);
    }

    return globs.myResult[globs.myXindex][globs.myYindex];
}

int   run_OptimizedOriginal(  
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
    REAL strike;
    PrivGlobs    globs(numX, numY, numT);

    for( unsigned i = 0; i < outer; ++ i ) {
        strike = 0.001*i;
        res[i] = value_opti( globs, s0, strike, t,
                        alpha, nu,    beta,
                        numX,  numY,  numT );
    }
    return 1;
}

//#endif // PROJ_CORE_ORIG
