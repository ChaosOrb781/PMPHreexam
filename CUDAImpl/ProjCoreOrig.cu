#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <omp.h>

#include "./includes/ProjHelperFun.cu.h"
#include "./includes/Constants.cu.h"

#define DEBUG 0

void   run_OrigCPUExpand(  
    const unsigned int&   outer,
    const unsigned int&   numX,
    const unsigned int&   numY,
    const unsigned int&   numT,
    const REAL&           s0,
    const REAL&           t, 
    const REAL&           alpha, 
    const REAL&           nu, 
    const REAL&           beta,
          REAL*           res   // [outer] RESULT
) {
    for( unsigned i = 0; i < outer; ++ i ) {
        //Alternative to vectors, reused memory locations, part of tridag
        MyReal4* mats = (MyReal4*) malloc(max(numX, numY) * sizeof(MyReal4));
        MyReal2* lfuns = (MyReal2*) malloc(max(numX, numY) * sizeof(MyReal2));

        PrivGlobs* globs = new PrivGlobs(numX, numY, numT);
        REAL strike = 0.001*i;
        //value(globs, s0, strike, t, alpha, nu, beta, numX, numY, numT);
        //initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
        for(unsigned j=0;j<numT;++j) 
            globs->myTimeline[j] = t*j/(numT-1);

        const REAL stdX = 20.0*alpha*s0*sqrt(t);
        const REAL dx = stdX/numX;
        globs->myXindex = static_cast<unsigned>(s0/dx) % numX;

        for(unsigned j=0;j<numX;++j)
            globs->myX[j] = j*dx - globs->myXindex*dx + s0;

        const REAL stdY = 10.0*nu*sqrt(t);
        const REAL dy = stdY/numY;
        const REAL logAlpha = log(alpha);
        globs->myYindex = static_cast<unsigned>(numY/2.0);

        for(unsigned j=0;j<numY;++j)
            globs->myY[j] = j*dy - globs->myYindex*dy + logAlpha;

        //initOperator(globs->myX,globs->myDxx);
        REAL dl, du;

        //  lower boundary
        dl = 0.0;
        du = globs->myX[1] - globs->myX[0];

        globs->myDxx[0][0] =  0.0;
        globs->myDxx[0][1] =  0.0;
        globs->myDxx[0][2] =  0.0;
        globs->myDxx[0][3] =  0.0;

        //  standard case
        for(unsigned j=1;j<numX-1;j++)
        {
            dl = globs->myX[j]   - globs->myX[j-1];
            du = globs->myX[j+1] - globs->myX[j];

            globs->myDxx[j][0] =  2.0/dl/(dl+du);
            globs->myDxx[j][1] = -2.0*(1.0/dl + 1.0/du)/(dl+du);
            globs->myDxx[j][2] =  2.0/du/(dl+du);
            globs->myDxx[j][3] =  0.0; 
        }

        //  upper boundary
        dl = globs->myX[numX-1] - globs->myX[numX-2];
        du = 0.0;

        globs->myDxx[numX-1][0] = 0.0;
        globs->myDxx[numX-1][1] = 0.0;
        globs->myDxx[numX-1][2] = 0.0;
        globs->myDxx[numX-1][3] = 0.0;

        //initOperator(globs->myY,globs->myDyy);
        //  lower boundary
        dl =  0.0;
        du =  globs->myY[1] - globs->myY[0];

        globs->myDyy[0][0] =  0.0;
        globs->myDyy[0][1] =  0.0;
        globs->myDyy[0][2] =  0.0;
        globs->myDyy[0][3] =  0.0;

        //  standard case
        for(unsigned j=1;j<numY-1;j++)
        {
            dl = globs->myY[j]   - globs->myY[j-1];
            du = globs->myY[j+1] - globs->myY[j];

            globs->myDyy[j][0] =  2.0/dl/(dl+du);
            globs->myDyy[j][1] = -2.0*(1.0/dl + 1.0/du)/(dl+du);
            globs->myDyy[j][2] =  2.0/du/(dl+du);
            globs->myDyy[j][3] =  0.0; 
        }

        //  upper boundary
        dl = globs->myY[numY-1] - globs->myY[numY-2];
        du = 0.0;

        globs->myDyy[numY-1][0] = 0.0;
        globs->myDyy[numY-1][1] = 0.0;
        globs->myDyy[numY-1][2] = 0.0;
        globs->myDyy[numY-1][3] = 0.0;

        //setPayoff(strike, globs);
        for(unsigned j=0;j<numX;++j)
        {
            REAL payoff = max(globs->myX[j]-strike, (REAL)0.0);
            for(unsigned k=0;k<numY;++k)
                globs->myResult[j][k] = payoff;
        }

        for(int j = numT-2;j>=0;--j)
        {
            //updateParams(i,alpha,beta,nu,globs);
            for(unsigned k=0;k<numX;++k) {
                for(unsigned h=0;h<numY;++h) {
                    globs->myVarX[k][h] = exp(2.0*(  beta*log(globs->myX[k])   
                                                + globs->myY[h]             
                                                - 0.5*nu*nu*globs->myTimeline[j] )
                                            );
                    globs->myVarY[k][h] = exp(2.0*(  alpha*log(globs->myX[k])   
                                                + globs->myY[h]             
                                                - 0.5*nu*nu*globs->myTimeline[j] )
                                            ); // nu*nu
                }
            }
            //rollback(i, globs);

            unsigned numZ = max(numX,numY);

            REAL dtInv = 1.0/(globs->myTimeline[j+1]-globs->myTimeline[j]);

            vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
            vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
            vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
            vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

            //	explicit x
            for(unsigned k=0;k<numX;k++) {
                for(unsigned h=0;h<numY;h++) {
                    u[h][k] = dtInv*globs->myResult[k][h];

                    if(k > 0) { 
                        u[h][k] += 0.5*( 0.5*globs->myVarX[k][h]*globs->myDxx[k][0] ) 
                                    * globs->myResult[k-1][h];
                    }
                    u[h][k]  +=  0.5*( 0.5*globs->myVarX[k][h]*globs->myDxx[k][1] )
                                    * globs->myResult[k][h];
                    if(k < numX-1) {
                        u[h][k] += 0.5*( 0.5*globs->myVarX[k][h]*globs->myDxx[k][2] )
                                    * globs->myResult[k+1][h];
                    }
                }
            }

            //	explicit y
            for(unsigned k=0;k<numY;k++) {
                for(unsigned h=0;h<numX;h++) {
                    v[h][k] = 0.0;

                    if(k > 0) {
                        v[h][k] +=  ( 0.5*globs->myVarY[h][k]*globs->myDyy[k][0] )
                                *  globs->myResult[h][k-1];
                    }
                    v[h][k]  +=   ( 0.5*globs->myVarY[h][k]*globs->myDyy[k][1] )
                                *  globs->myResult[h][k];
                    if(k < numY-1) {
                        v[h][k] +=  ( 0.5*globs->myVarY[h][k]*globs->myDyy[k][2] )
                                *  globs->myResult[h][k+1];
                    }
                    u[k][h] += v[h][k]; 
                }
            }

            //	implicit x
            for(unsigned k=0;k<numY;k++) {
                for(unsigned h=0;h<numX;h++) {  // here a, b,c should have size [numX]
                    a[h] =		 - 0.5*(0.5*globs->myVarX[h][k]*globs->myDxx[h][0]);
                    b[h] = dtInv - 0.5*(0.5*globs->myVarX[h][k]*globs->myDxx[h][1]);
                    c[h] =		 - 0.5*(0.5*globs->myVarX[h][k]*globs->myDxx[h][2]);
                }

                // here yy should have size [numX]
                //tridagPar(a,b,c,u[k],numX,u[k],yy);

                //vector<MyReal4> scanres(n); // supposed to also be in shared memory and to reuse the space of mats
                //--------------------------------------------------
                // Recurrence 1: b[i] = b[i] - a[i]*c[i-1]/b[i-1] --
                //   solved by scan with 2x2 matrix mult operator --
                //--------------------------------------------------
                MyReal4* mats = (MyReal4*)malloc(numX*sizeof(MyReal4));    // supposed to be in shared memory!
                REAL b0 = b[0];
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    if (h==0) { mats[h].x = 1.0;  mats[h].y = 0.0;          mats[h].z = 0.0; mats[h].w = 1.0; }
                    else      { mats[h].x = b[h]; mats[h].y = -a[h]*c[h-1]; mats[h].z = 1.0; mats[h].w = 0.0; }
                }
                inplaceScanInc<MatMult2b2>(numX,mats);
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    yy[h] = (mats[h].x*b0 + mats[h].y) / (mats[h].z*b0 + mats[h].w);
                }
                // b -> uu
                //----------------------------------------------------
                // Recurrence 2: y[i] = y[i] - (a[i]/b[i-1])*y[i-1] --
                //   solved by scan with linear func comp operator  --
                //----------------------------------------------------
                MyReal2* lfuns = (MyReal2*)malloc(numX*sizeof(MyReal2));
                REAL y0 = u[k][0];
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    if (h==0) { lfuns[0].x = 0.0;     lfuns[0].y = 1.0;           }
                    else      { lfuns[h].x = u[k][h]; lfuns[h].y = -a[h]/yy[h-1]; }
                }
                inplaceScanInc<LinFunComp>(numX,lfuns);
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    u[k][h] = lfuns[h].x + y0*lfuns[h].y;
                }
                // y -> u

                //----------------------------------------------------
                // Recurrence 3: backward recurrence solved via     --
                //             scan with linear func comp operator  --
                //----------------------------------------------------
                REAL yn = u[k][numX-1]/yy[numX-1];
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    if (h==0) { lfuns[0].x = 0.0;  lfuns[0].y = 1.0;           }
                    else      { lfuns[h].x = u[k][numX-h-1]/yy[numX-h-1]; lfuns[h].y = -c[numX-h-1]/yy[numX-h-1]; }
                }
                inplaceScanInc<LinFunComp>(numX,lfuns);
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    u[k][numX-h-1] = lfuns[h].x + yn*lfuns[h].y;
                }
            }

            //	implicit y
            for(unsigned k=0;k<numX;k++) { 
                for(unsigned h=0;h<numY;h++) {  // here a, b, c should have size [numY]
                    a[h] =		 - 0.5*(0.5*globs->myVarY[k][h]*globs->myDyy[h][0]);
                    b[h] = dtInv - 0.5*(0.5*globs->myVarY[k][h]*globs->myDyy[h][1]);
                    c[h] =		 - 0.5*(0.5*globs->myVarY[k][h]*globs->myDyy[h][2]);
                }

                for(unsigned h=0;h<numY;h++)
                    y[h] = dtInv*u[h][k] - 0.5*v[k][h];

                // here yy should have size [numY]
                //tridagPar(a,b,c,y,numY,globs->myResult[j],yy);

                //vector<MyReal4> scanres(n); // supposed to also be in shared memory and to reuse the space of mats
                //--------------------------------------------------
                // Recurrence 1: b[i] = b[i] - a[i]*c[i-1]/b[i-1] --
                //   solved by scan with 2x2 matrix mult operator --
                //--------------------------------------------------
                MyReal4* mats = (MyReal4*)malloc(numY*sizeof(MyReal4));    // supposed to be in shared memory!
                REAL b0 = b[0];
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    if (h==0) { mats[h].x = 1.0;  mats[h].y = 0.0;          mats[h].z = 0.0; mats[h].w = 1.0; }
                    else      { mats[h].x = b[h]; mats[h].y = -a[h]*c[h-1]; mats[h].z = 1.0; mats[h].w = 0.0; }
                }
                inplaceScanInc<MatMult2b2>(numY,mats);
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    yy[h] = (mats[h].x*b0 + mats[h].y) / (mats[h].z*b0 + mats[h].w);
                }
                // b -> uu
                //----------------------------------------------------
                // Recurrence 2: y[i] = y[i] - (a[i]/b[i-1])*y[i-1] --
                //   solved by scan with linear func comp operator  --
                //----------------------------------------------------
                MyReal2* lfuns = (MyReal2*)malloc(numY*sizeof(MyReal2));
                REAL y0 = y[0];
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    if (h==0) { lfuns[0].x = 0.0;  lfuns[0].y = 1.0;           }
                    else      { lfuns[h].x = y[h]; lfuns[h].y = -a[h]/yy[h-1]; }
                }
                inplaceScanInc<LinFunComp>(numY,lfuns);
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    globs->myResult[k][h] = lfuns[h].x + y0*lfuns[h].y;
                }
                // y -> u

                //----------------------------------------------------
                // Recurrence 3: backward recurrence solved via     --
                //             scan with linear func comp operator  --
                //----------------------------------------------------
                REAL yn = globs->myResult[k][numY-1]/yy[numY-1];
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    if (h==0) { lfuns[0].x = 0.0;  lfuns[0].y = 1.0;           }
                    else      { lfuns[h].x = globs->myResult[k][numY-h-1]/yy[numY-h-1]; lfuns[h].y = -c[numY-h-1]/yy[numY-h-1]; }
                }
                inplaceScanInc<LinFunComp>(numY,lfuns);
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    globs->myResult[k][numY-h-1] = lfuns[h].x + yn*lfuns[h].y;
                }
            }
        }

        res[i] = globs->myResult[globs->myXindex][globs->myYindex];
        free(mats);
        free(lfuns);
    }
}

void   run_OrigCPUExpand1stOUTER(  
    const unsigned int&   outer,
    const unsigned int&   numX,
    const unsigned int&   numY,
    const unsigned int&   numT,
    const REAL&           s0,
    const REAL&           t, 
    const REAL&           alpha, 
    const REAL&           nu, 
    const REAL&           beta,
          REAL*           res   // [outer] RESULT
) {
    //Alternative to vectors, reused memory locations, part of tridag
    MyReal4* mats = (MyReal4*) malloc(max(numX, numY) * sizeof(MyReal4));
    MyReal2* lfuns = (MyReal2*) malloc(max(numX, numY) * sizeof(MyReal2));

    //Invariants for OUTER, never written to beyond initialization (read-only)
    //Therefore not expanded
    REAL* myX = (REAL*) malloc(numX * sizeof(REAL)); 
    REAL* myY = (REAL*) malloc(numY * sizeof(REAL));
    REAL* myTimeline = (REAL*) malloc(numT * sizeof(REAL));
    //Array expanded by OUTER due to distribution
    REAL* myResult = (REAL*) malloc(outer * numX * numY * sizeof(REAL));
    //Based on myX and myY which are all invariant to outer loop
    //Therefore no expansion needed
    REAL* myDxx = (REAL*) malloc(numX * 4 * sizeof(REAL));
    REAL* myDyy = (REAL*) malloc(numY * 4 * sizeof(REAL));
    REAL* myVarX = (REAL*) malloc(numT * numX * numY * sizeof(REAL));
    REAL* myVarY = (REAL*) malloc(numT * numX * numY * sizeof(REAL));
    //Tridag initialization values
    unsigned numZ = max(numX,numY);
    REAL* u = (REAL*) malloc(numY * numX * sizeof(REAL));
    REAL* v = (REAL*) malloc(numX * numY * sizeof(REAL));
    REAL* a = (REAL*) malloc(numZ * sizeof(REAL));
    REAL* b = (REAL*) malloc(numZ * sizeof(REAL));
    REAL* c = (REAL*) malloc(numZ * sizeof(REAL));
    REAL* y = (REAL*) malloc(numZ * sizeof(REAL));
    REAL* yy = (REAL*) malloc(numZ * sizeof(REAL));
    //Variables expanded to array
    REAL* dtInv = (REAL*) malloc(numT * sizeof(REAL));

    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S1, distributed, found to be invariant and no overrides, thereby not expanded
    for(unsigned j=0;j<numT;++j) {
        myTimeline[j] = t*j/(numT-1);
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    unsigned myXindex = static_cast<unsigned>(s0/dx) % numX;

    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S2, distributed, found to be invariant and no overrides, thereby not expanded
    for(unsigned j=0;j<numX;++j) {
        myX[j] = j*dx - myXindex*dx + s0;
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S3, distributed, found to be invariant and no overrides, thereby not expanded
    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    unsigned myYindex = static_cast<unsigned>(numY/2.0);

    for(unsigned j=0;j<numY;++j) {
        myY[j] = j*dy - myYindex*dy + logAlpha;
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S4, distributed, found to be invariant due to variable use and no overrides
    //thereby not expanded
    REAL dl, du;

    //  lower boundary
    dl = 0.0;
    du = myX[1] - myX[0];

    myDxx[0 * 4 + 0] =  0.0;
    myDxx[0 * 4 + 1] =  0.0;
    myDxx[0 * 4 + 2] =  0.0;
    myDxx[0 * 4 + 3] =  0.0;

    //  standard case
    for(unsigned j=1;j<numX-1;j++)
    {
        dl = myX[j]   - myX[j-1];
        du = myX[j+1] - myX[j];

        myDxx[j * 4 + 0] =  2.0/dl/(dl+du);
        myDxx[j * 4 + 1] = -2.0*(1.0/dl + 1.0/du)/(dl+du);
        myDxx[j * 4 + 2] =  2.0/du/(dl+du);
        myDxx[j * 4 + 3] =  0.0;
    }

    //  upper boundary
    dl = myX[numX-1] - myX[numX-2];
    du = 0.0;

    myDxx[(numX-1) * 4 + 0] = 0.0;
    myDxx[(numX-1) * 4 + 1] = 0.0;
    myDxx[(numX-1) * 4 + 2] = 0.0;
    myDxx[(numX-1) * 4 + 3] = 0.0;
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S5, distributed,  found to be invariant due to variable use and no overrides
    //thereby not expanded
    //initOperator(myY,myDyy);
    //  lower boundary
    dl =  0.0;
    du =  myY[1] - myY[0];

    myDyy[0 * 4 + 0] =  0.0;
    myDyy[0 * 4 + 1] =  0.0;
    myDyy[0 * 4 + 2] =  0.0;
    myDyy[0 * 4 + 3] =  0.0;

    //  standard case
    for(unsigned j=1;j<numY-1;j++)
    {
        dl = myY[j]   - myY[j-1];
        du = myY[j+1] - myY[j];

        myDyy[j * 4 + 0] =  2.0/dl/(dl+du);
        myDyy[j * 4 + 1] = -2.0*(1.0/dl + 1.0/du)/(dl+du);
        myDyy[j * 4 + 2] =  2.0/du/(dl+du);
        myDyy[j * 4 + 3] =  0.0;
    }

    //  upper boundary
    dl = myY[numY-1] - myY[numY-2];
    du = 0.0;

    myDyy[(numY-1) * 4 + 0] = 0.0;
    myDyy[(numY-1) * 4 + 1] = 0.0;
    myDyy[(numY-1) * 4 + 2] = 0.0;
    myDyy[(numY-1) * 4 + 3] = 0.0;
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S6, distributed and array expanded due to later write operations dependent on OUTER
    for( unsigned i = 0; i < outer; ++ i ) {
        REAL strike = 0.001*i;
        for(unsigned j=0;j<numX;++j)
        {
            REAL payoff = max(myX[j]-strike, (REAL)0.0);
            for(unsigned k=0;k<numY;++k) {
                myResult[(i * numX + j) * numY + k] = payoff;
            }
        }
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S7, distributed and array expanded due to dependency of j in myTimeline
    //Invariant to outer loop
    for(int j = numT-2;j>=0;--j) {
        //updateParams(i,alpha,beta,nu,globs);
        for(unsigned k=0;k<numX;++k) {
            for(unsigned h=0;h<numY;++h) {
                myVarX[(j * numX + k) * numY + h] = exp(2.0*(  beta*log(myX[k])   
                                            + myY[h]             
                                            - 0.5*nu*nu*myTimeline[j] )
                                        );
                myVarY[(j * numX + k) * numY + h] = exp(2.0*(  alpha*log(myX[k])   
                                            + myY[h]             
                                            - 0.5*nu*nu*myTimeline[j] )
                                        ); // nu*nu
            }
        }
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S8, distributed and array expanded due to several SCC requires value
    //But again invariant to the outer iteration, therefore we only expand by numT
    for(int j = numT-2;j>=0;--j) {
        dtInv[j] = 1.0/(myTimeline[j+1]-myTimeline[j]);
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

    //INTERCHANGE NEEDED: to make inner loop parallel instead of having to run both sequentially
    for( unsigned i = 0; i < outer; ++ i ) {
        for(int j = numT-2;j>=0;--j) {// required sequential due to myResult cross iteration RAW
            //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
            //S9, fully distributed and expanded due to depedencies of both outer loops
            //and later use in other SCC
            //	explicit x
            for(unsigned k=0;k<numX;k++) {
                for(unsigned h=0;h<numY;h++) {
                    u[h * numX + k] = dtInv[j]*myResult[(i * numX + k) * numY + h];
            
                    if(k > 0) { 
                        u[h * numX + k] += 0.5*( 0.5*myVarX[(j * numX + k) * numY + h]*myDxx[k * 4 + 0] ) 
                                    * myResult[(i * numX + (k-1)) * numY + h];
                    }
                    u[h * numX + k]  +=  0.5*( 0.5*myVarX[(j * numX + k) * numY + h]*myDxx[k * 4 + 1] )
                                    * myResult[(i * numX + k) * numY + h];
                    if(k < numX-1) {
                        u[h * numX + k] += 0.5*( 0.5*myVarX[(j * numX + k) * numY + h]*myDxx[k * 4 + 2] )
                                    * myResult[(i * numX + (k+1)) * numY + h];
                    }
                }
            }
            //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

            //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
            //S10, fully distributed and expanded due to depedencies of both outer loops
            //and later use in other SCC
            //	explicit y
            for(unsigned k=0;k<numY;k++) {
                for(unsigned h=0;h<numX;h++) {
                    v[h * numY + k] = 0.0;

                    if(k > 0) {
                        v[h * numY + k] +=  ( 0.5*myVarY[(j * numX + h) * numY + k]*myDyy[k * 4 + 0] )
                                *  myResult[(i * numX + h) * numY + k-1];
                    }
                    v[h * numY + k]  +=   ( 0.5*myVarY[(j * numX + h) * numY + k]*myDyy[k * 4 + 1] )
                                *  myResult[(i * numX + h) * numY + k];
                    if(k < numY-1) {
                        v[h * numY + k] +=  ( 0.5*myVarY[(j * numX + h) * numY + k]*myDyy[k * 4 + 2] )
                                *  myResult[(i * numX + h) * numY + k+1];
                    }
                    u[k * numX + h] += v[h * numY + k]; 
                }
            }
            //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

            //	implicit x
            for(unsigned k=0;k<numY;k++) {
                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S11, distribution and array expansion applied to j and k due to dependency
                //in other iterations, invarient to outer dimension and only ever read from
                //after this point (or entirely overwritten)
                for(unsigned h=0;h<numX;h++) {  // here a, b,c should have size [numX]
                    a[h] =		 - 0.5*(0.5*myVarX[(j * numX + h) * numY + k]*myDxx[h * 4 + 0]);
                    b[h] = dtInv[j] - 0.5*(0.5*myVarX[(j * numX + h) * numY + k]*myDxx[h * 4 + 1]);
                    c[h] =		 - 0.5*(0.5*myVarX[(j * numX + h) * numY + k]*myDxx[h * 4 + 2]);
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S11.1, distribution and array expansion applied to j and k due to dependency
                //in other iterations, invariant to the outer loop, therefore hoisted
                MyReal4* mats = (MyReal4*)malloc(numX*sizeof(MyReal4));    // supposed to be in shared memory!
                REAL b0 = b[0];
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    if (h==0) { mats[h].x = 1.0;  mats[h].y = 0.0;          mats[h].z = 0.0; mats[h].w = 1.0; }
                    else      { mats[h].x = b[h]; mats[h].y = -a[h]*c[h-1]; mats[h].z = 1.0; mats[h].w = 0.0; }
                }
                inplaceScanInc<MatMult2b2>(numX,mats);
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    yy[h] = (mats[h].x*b0 + mats[h].y) / (mats[h].z*b0 + mats[h].w);
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S11.2, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                MyReal2* lfuns = (MyReal2*)malloc(numX*sizeof(MyReal2));
                REAL y0 = u[k * numX + 0];
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    if (h==0) { lfuns[0].x = 0.0;     lfuns[0].y = 1.0;           }
                    else      { lfuns[h].x = u[k * numX + h]; lfuns[h].y = -a[h]/yy[h-1]; }
                }
                inplaceScanInc<LinFunComp>(numX,lfuns);
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    u[k * numX + h] = lfuns[h].x + y0*lfuns[h].y;
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S11.3, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                REAL yn = u[k * numX + numX-1]/yy[numX-1];
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    if (h==0) { lfuns[0].x = 0.0;  lfuns[0].y = 1.0;           }
                    else      { lfuns[h].x = u[k * numX + numX-h-1]/yy[numX-h-1]; lfuns[h].y = -c[numX-h-1]/yy[numX-h-1]; }
                }
                inplaceScanInc<LinFunComp>(numX,lfuns);
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    u[k * numX + numX-h-1] = lfuns[h].x + yn*lfuns[h].y;
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
            }

            //	implicit y
            for(unsigned k=0;k<numX;k++) { 
                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S12, distribution and array expansion applied to j and k due to dependency
                //in other iterations, invarient to outer dimension and only ever read from
                //after this point
                for(unsigned h=0;h<numY;h++) {  // here a, b, c should have size [numY]
                    a[h] =		 - 0.5*(0.5*myVarY[(j * numX + k) * numY + h]*myDyy[h * 4 + 0]);
                    b[h] = dtInv[j] - 0.5*(0.5*myVarY[(j * numX + k) * numY + h]*myDyy[h * 4 + 1]);
                    c[h] =		 - 0.5*(0.5*myVarY[(j * numX + k) * numY + h]*myDyy[h * 4 + 2]);
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S12.1, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                for(unsigned h=0;h<numY;h++) {
                    y[h] = dtInv[j]*u[h * numX + k] - 0.5*v[k * numY + h];
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S12.2, distribution and array expansion applied to j and k due to dependency
                //in other iterations, invariant to the outer loop, therefore hoisted
                MyReal4* mats = (MyReal4*)malloc(numY*sizeof(MyReal4));    // supposed to be in shared memory!
                REAL b0 = b[0];
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    if (h==0) { mats[h].x = 1.0;  mats[h].y = 0.0;          mats[h].z = 0.0; mats[h].w = 1.0; }
                    else      { mats[h].x = b[h]; mats[h].y = -a[h]*c[h-1]; mats[h].z = 1.0; mats[h].w = 0.0; }
                }
                inplaceScanInc<MatMult2b2>(numY,mats);
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    yy[h] = (mats[h].x*b0 + mats[h].y) / (mats[h].z*b0 + mats[h].w);
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
                
                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S12.3, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                MyReal2* lfuns = (MyReal2*)malloc(numY*sizeof(MyReal2));
                REAL y0 = y[0];
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    if (h==0) { lfuns[0].x = 0.0;  lfuns[0].y = 1.0;           }
                    else      { lfuns[h].x = y[h]; lfuns[h].y = -a[h]/yy[h-1]; }
                }
                inplaceScanInc<LinFunComp>(numY,lfuns);
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    myResult[(i * numX + k) * numY + h] = lfuns[h].x + y0*lfuns[h].y;
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
                
                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S12.4, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                REAL yn = myResult[(i * numX + k) * numY + numY-1]/yy[numY-1];
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    if (h==0) { lfuns[0].x = 0.0;  lfuns[0].y = 1.0;           }
                    else      { lfuns[h].x = myResult[(i * numX + k) * numY + numY-h-1]/yy[numY-h-1]; lfuns[h].y = -c[numY-h-1]/yy[numY-h-1]; }
                }
                inplaceScanInc<LinFunComp>(numY,lfuns);
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    myResult[(i * numX + k) * numY + numY-h-1] = lfuns[h].x + yn*lfuns[h].y;
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
            }
        }
        //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
        //S13, distribution and array expansion applied to j and k due to dependency
        //in other iterations, invariant to the outer loop, therefore hoisted
        res[i] = myResult[(i * numX + myXindex) * numY + myYindex];
        //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
    }
    free(mats);
    free(lfuns);
    free(myX       );
    free(myY       );
    free(myTimeline);
    free(myDxx     );
    free(myDyy     );
    free(myVarX    );
    free(myVarY    );
    free(a         );
    free(b         );
    free(c         );
    free(myResult  );
    free(u         );
    free(v         );
    free(y         );
    free(yy        );
    free(dtInv     );
}

void   run_OrigCPUExpand2ndOUTER(  
    const unsigned int&   outer,
    const unsigned int&   numX,
    const unsigned int&   numY,
    const unsigned int&   numT,
    const REAL&           s0,
    const REAL&           t, 
    const REAL&           alpha, 
    const REAL&           nu, 
    const REAL&           beta,
          REAL*           res   // [outer] RESULT
) {
    //Alternative to vectors, reused memory locations, part of tridag
    MyReal4* mats = (MyReal4*) malloc(max(numX, numY) * sizeof(MyReal4));
    MyReal2* lfuns = (MyReal2*) malloc(max(numX, numY) * sizeof(MyReal2));

    //Invariants for OUTER, never written to beyond initialization (read-only)
    //Therefore not expanded
    REAL* myX = (REAL*) malloc(numX * sizeof(REAL)); 
    REAL* myY = (REAL*) malloc(numY * sizeof(REAL));
    REAL* myTimeline = (REAL*) malloc(numT * sizeof(REAL));
    //Array expanded by OUTER due to distribution
    REAL* myResult = (REAL*) malloc(outer * numX * numY * sizeof(REAL));
    //Based on myX and myY which are all invariant to outer loop
    //Therefore no expansion needed
    REAL* myDxx = (REAL*) malloc(numX * 4 * sizeof(REAL));
    REAL* myDyy = (REAL*) malloc(numY * 4 * sizeof(REAL));
    REAL* myVarX = (REAL*) malloc(numT * numX * numY * sizeof(REAL));
    REAL* myVarY = (REAL*) malloc(numT * numX * numY * sizeof(REAL));
    //Tridag initialization values
    unsigned numZ = max(numX,numY);
    REAL* u = (REAL*) malloc(numY * numX * sizeof(REAL));
    REAL* v = (REAL*) malloc(numX * numY * sizeof(REAL));
    REAL* aX = (REAL*) malloc(numZ * numZ * sizeof(REAL));
    REAL* bX = (REAL*) malloc(numZ * numZ * sizeof(REAL));
    REAL* cX = (REAL*) malloc(numZ * numZ * sizeof(REAL));
    REAL* aY = (REAL*) malloc(numZ * numZ * sizeof(REAL));
    REAL* bY = (REAL*) malloc(numZ * numZ * sizeof(REAL));
    REAL* cY = (REAL*) malloc(numZ * numZ * sizeof(REAL));
    REAL* y = (REAL*) malloc(numZ * sizeof(REAL));
    REAL* yyX = (REAL*) malloc(numZ * numZ * sizeof(REAL));
    REAL* yyY = (REAL*) malloc(numZ * numZ * sizeof(REAL));
    //Variables expanded to array
    REAL* dtInv = (REAL*) malloc(numT * sizeof(REAL));

#if DEBUG
    cout << "S1" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S1, distributed, found to be invariant and no overrides, thereby not expanded
    for(unsigned j=0;j<numT;++j) {
        myTimeline[j] = t*j/(numT-1);
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    unsigned myXindex = static_cast<unsigned>(s0/dx) % numX;

#if DEBUG
    cout << "S2" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S2, distributed, found to be invariant and no overrides, thereby not expanded
    for(unsigned j=0;j<numX;++j) {
        myX[j] = j*dx - myXindex*dx + s0;
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
    cout << "S3" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S3, distributed, found to be invariant and no overrides, thereby not expanded
    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    unsigned myYindex = static_cast<unsigned>(numY/2.0);

    for(unsigned j=0;j<numY;++j) {
        myY[j] = j*dy - myYindex*dy + logAlpha;
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
    cout << "S4" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S4, distributed, found to be invariant due to variable use and no overrides
    //thereby not expanded
    REAL dl, du;

    //  lower boundary
    dl = 0.0;
    du = myX[1] - myX[0];

    myDxx[0 * 4 + 0] =  0.0;
    myDxx[0 * 4 + 1] =  0.0;
    myDxx[0 * 4 + 2] =  0.0;
    myDxx[0 * 4 + 3] =  0.0;

    //  standard case
    for(unsigned j=1;j<numX-1;j++)
    {
        dl = myX[j]   - myX[j-1];
        du = myX[j+1] - myX[j];

        myDxx[j * 4 + 0] =  2.0/dl/(dl+du);
        myDxx[j * 4 + 1] = -2.0*(1.0/dl + 1.0/du)/(dl+du);
        myDxx[j * 4 + 2] =  2.0/du/(dl+du);
        myDxx[j * 4 + 3] =  0.0;
    }

    //  upper boundary
    dl = myX[numX-1] - myX[numX-2];
    du = 0.0;

    myDxx[(numX-1) * 4 + 0] = 0.0;
    myDxx[(numX-1) * 4 + 1] = 0.0;
    myDxx[(numX-1) * 4 + 2] = 0.0;
    myDxx[(numX-1) * 4 + 3] = 0.0;
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
    cout << "S5" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S5, distributed,  found to be invariant due to variable use and no overrides
    //thereby not expanded
    //initOperator(myY,myDyy);
    //  lower boundary
    dl =  0.0;
    du =  myY[1] - myY[0];

    myDyy[0 * 4 + 0] =  0.0;
    myDyy[0 * 4 + 1] =  0.0;
    myDyy[0 * 4 + 2] =  0.0;
    myDyy[0 * 4 + 3] =  0.0;

    //  standard case
    for(unsigned j=1;j<numY-1;j++)
    {
        dl = myY[j]   - myY[j-1];
        du = myY[j+1] - myY[j];

        myDyy[j * 4 + 0] =  2.0/dl/(dl+du);
        myDyy[j * 4 + 1] = -2.0*(1.0/dl + 1.0/du)/(dl+du);
        myDyy[j * 4 + 2] =  2.0/du/(dl+du);
        myDyy[j * 4 + 3] =  0.0;
    }

    //  upper boundary
    dl = myY[numY-1] - myY[numY-2];
    du = 0.0;

    myDyy[(numY-1) * 4 + 0] = 0.0;
    myDyy[(numY-1) * 4 + 1] = 0.0;
    myDyy[(numY-1) * 4 + 2] = 0.0;
    myDyy[(numY-1) * 4 + 3] = 0.0;
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
    cout << "S6" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S6, distributed and array expanded due to later write operations dependent on OUTER
    for( unsigned i = 0; i < outer; ++ i ) {
        REAL strike = 0.001*i;
        for(unsigned j=0;j<numX;++j)
        {
            REAL payoff = max(myX[j]-strike, (REAL)0.0);
            for(unsigned k=0;k<numY;++k) {
                myResult[(i * numX + j) * numY + k] = payoff;
            }
        }
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
    cout << "S7" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S7, distributed and array expanded due to dependency of j in myTimeline
    //Invariant to outer loop
    for(int j = numT-2;j>=0;--j) {
        //updateParams(i,alpha,beta,nu,globs);
        for(unsigned k=0;k<numX;++k) {
            for(unsigned h=0;h<numY;++h) {
                myVarX[(j * numX + k) * numY + h] = exp(2.0*(  beta*log(myX[k])   
                                            + myY[h]             
                                            - 0.5*nu*nu*myTimeline[j] )
                                        );
                myVarY[(j * numX + k) * numY + h] = exp(2.0*(  alpha*log(myX[k])   
                                            + myY[h]             
                                            - 0.5*nu*nu*myTimeline[j] )
                                        ); // nu*nu
            }
        }
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
    cout << "S8" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S8, distributed and array expanded due to several SCC requires value
    //But again invariant to the outer iteration, therefore we only expand by numT
    for(int j = numT-2;j>=0;--j) {
        dtInv[j] = 1.0/(myTimeline[j+1]-myTimeline[j]);
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
    
    //INTERCHANGED to make it [<,=] instead of [=,<] due to myResult read/write dependency
    for(int j = numT-2;j>=0;--j) {// required sequential due to myResult cross iteration RAW

#if DEBUG
    cout << "S11 " << j << endl;
#endif 
        //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
        //S11, distribution and array expansion applied to j and k due to dependency
        //in other iterations, invarient to outer dimension and only ever read from
        //after this point (or entirely overwritten)
        for(unsigned k=0;k<numY;k++) {
            for(unsigned h=0;h<numX;h++) {  // here a, b,c should have size [numX]
                aX[k * numZ + h] =		    - 0.5*(0.5*myVarX[(j * numX + h) * numY + k]*myDxx[h * 4 + 0]);
                bX[k * numZ + h] = dtInv[j] - 0.5*(0.5*myVarX[(j * numX + h) * numY + k]*myDxx[h * 4 + 1]);
                cX[k * numZ + h] =		    - 0.5*(0.5*myVarX[(j * numX + h) * numY + k]*myDxx[h * 4 + 2]);
            }
        }
        //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
        cout << "S12 " << j << endl;
#endif 
        //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
        //S12, distribution and array expansion applied to j and k due to dependency
        //in other iterations, invarient to outer dimension and only ever read from
        //after this point
        for(unsigned k=0;k<numX;k++) {
            for(unsigned h=0;h<numY;h++) {  // here a, b, c should have size [numY]
                aY[k * numZ + h] =		    - 0.5*(0.5*myVarY[(j * numX + k) * numY + h]*myDyy[h * 4 + 0]);
                bY[k * numZ + h] = dtInv[j] - 0.5*(0.5*myVarY[(j * numX + k) * numY + h]*myDyy[h * 4 + 1]);
                cY[k * numZ + h] =		    - 0.5*(0.5*myVarY[(j * numX + k) * numY + h]*myDyy[h * 4 + 2]);
            }
        }
        //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
        cout << "S11.1 " << j << endl;
#endif 
        //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
        //S11.1, distribution and array expansion applied to j and k due to dependency
        //in other iterations, invariant to the outer loop, therefore hoisted
        for(unsigned k=0;k<numY;k++) {
            MyReal4* mats = (MyReal4*)malloc(numX*sizeof(MyReal4));    // supposed to be in shared memory!
            REAL b0 = bX[k * numZ + 0];
            for(int h=0; h<numX; h++) { //parallel, map-like semantics
                if (h==0) { 
                    mats[h].x = 1.0;  
                    mats[h].y = 0.0;          
                    mats[h].z = 0.0; 
                    mats[h].w = 1.0; 
                } else { 
                    mats[h].x = bX[k * numZ + h]; 
                    mats[h].y = -aX[k * numZ + h]*cX[k * numZ + h-1]; 
                    mats[h].z = 1.0; 
                    mats[h].w = 0.0; 
                }
            }
            inplaceScanInc<MatMult2b2>(numX,mats);
            for(int h=0; h<numX; h++) { //parallel, map-like semantics
                yyX[k * numZ + h] = (mats[h].x*b0 + mats[h].y) / (mats[h].z*b0 + mats[h].w);
            }
        }
        //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

        //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
        //S12.2, distribution and array expansion applied to j and k due to dependency
        //in other iterations, invariant to the outer loop, therefore hoisted
        for(unsigned k=0;k<numX;k++) {
            MyReal4* mats = (MyReal4*)malloc(numY*sizeof(MyReal4));    // supposed to be in shared memory!
            REAL b0 = bY[k * numZ + 0];
            for(int h=0; h<numY; h++) { //parallel, map-like semantics
                if (h==0) { 
                    mats[h].x = 1.0;  
                    mats[h].y = 0.0;          
                    mats[h].z = 0.0; 
                    mats[h].w = 1.0; 
                } else { 
                    mats[h].x = bY[k * numZ + h]; 
                    mats[h].y = -aY[k * numZ + h]*cY[k * numZ + h-1]; 
                    mats[h].z = 1.0; 
                    mats[h].w = 0.0; 
                }
            }
            inplaceScanInc<MatMult2b2>(numY,mats);
            for(int h=0; h<numY; h++) { //parallel, map-like semantics
                yyY[k * numZ + h] = (mats[h].x*b0 + mats[h].y) / (mats[h].z*b0 + mats[h].w);
            }
        }
        //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

        for( unsigned i = 0; i < outer; ++ i ) {
#if DEBUG
        cout << "S9 " << j << " " << i << endl;
#endif 
            //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
            //S9, fully distributed and expanded due to depedencies of both outer loops
            //and later use in other SCC
            //	explicit x
            for(unsigned k=0;k<numX;k++) {
                for(unsigned h=0;h<numY;h++) {
                    u[h * numX + k] = dtInv[j]*myResult[(i * numX + k) * numY + h];
            
                    if(k > 0) { 
                        u[h * numX + k] += 0.5*( 0.5*myVarX[(j * numX + k) * numY + h]*myDxx[k * 4 + 0] ) 
                                    * myResult[(i * numX + (k-1)) * numY + h];
                    }
                    u[h * numX + k]  +=  0.5*( 0.5*myVarX[(j * numX + k) * numY + h]*myDxx[k * 4 + 1] )
                                    * myResult[(i * numX + k) * numY + h];
                    if(k < numX-1) {
                        u[h * numX + k] += 0.5*( 0.5*myVarX[(j * numX + k) * numY + h]*myDxx[k * 4 + 2] )
                                    * myResult[(i * numX + (k+1)) * numY + h];
                    }
                }
            }
            //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
            cout << "S10 " << j << " " << i << endl;
#endif 
            //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
            //S10, fully distributed and expanded due to depedencies of both outer loops
            //and later use in other SCC
            //	explicit y
            for(unsigned k=0;k<numY;k++) {
                for(unsigned h=0;h<numX;h++) {
                    v[h * numY + k] = 0.0;

                    if(k > 0) {
                        v[h * numY + k] +=  ( 0.5*myVarY[(j * numX + h) * numY + k]*myDyy[k * 4 + 0] )
                                *  myResult[(i * numX + h) * numY + k-1];
                    }
                    v[h * numY + k]  +=   ( 0.5*myVarY[(j * numX + h) * numY + k]*myDyy[k * 4 + 1] )
                                *  myResult[(i * numX + h) * numY + k];
                    if(k < numY-1) {
                        v[h * numY + k] +=  ( 0.5*myVarY[(j * numX + h) * numY + k]*myDyy[k * 4 + 2] )
                                *  myResult[(i * numX + h) * numY + k+1];
                    }
                    u[k * numX + h] += v[h * numY + k]; 
                }
            }
            //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
            cout << "S11.2-3 " << j << " " << i << endl;
#endif 
            //	implicit x
            for(unsigned k=0;k<numY;k++) {
                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S11.2, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                MyReal2* lfuns = (MyReal2*)malloc(numX*sizeof(MyReal2));
                REAL y0 = u[k * numX + 0];
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    if (h==0) { 
                        lfuns[0].x = 0.0;     
                        lfuns[0].y = 1.0;           
                    } else { 
                        lfuns[h].x = u[k * numX + h]; 
                        lfuns[h].y = -aX[k * numZ + h]/yyX[k * numZ + h-1]; }
                }
                inplaceScanInc<LinFunComp>(numX,lfuns);
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    u[k * numX + h] = lfuns[h].x + y0*lfuns[h].y;
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S11.3, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                REAL yn = u[k * numX + numX-1]/yyX[k * numZ + numX-1];
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    if (h==0) { 
                        lfuns[0].x = 0.0;  
                        lfuns[0].y = 1.0;           
                    } else { 
                        lfuns[h].x = u[k * numX + numX-h-1]/yyX[k * numZ + numX-h-1]; 
                        lfuns[h].y = -cX[k * numZ + numX-h-1]/yyX[k * numZ + numX-h-1]; }
                }
                inplaceScanInc<LinFunComp>(numX,lfuns);
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    u[k * numX + numX-h-1] = lfuns[h].x + yn*lfuns[h].y;
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
            }

#if DEBUG
            cout << "S12.1-4 " << j << " " << i << endl;
#endif 
            //	implicit y
            for(unsigned k=0;k<numX;k++) {
                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S12.1, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                for(unsigned h=0;h<numY;h++) {
                    y[h] = dtInv[j]*u[h * numX + k] - 0.5*v[k * numY + h];
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
                
                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S12.3, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                MyReal2* lfuns = (MyReal2*)malloc(numY*sizeof(MyReal2));
                REAL y0 = y[0];
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    if (h==0) { 
                        lfuns[0].x = 0.0;  
                        lfuns[0].y = 1.0;           
                    } else { 
                        lfuns[h].x = y[h]; 
                        lfuns[h].y = -aY[k * numZ + h]/yyY[k * numZ + h-1]; 
                    }
                }
                inplaceScanInc<LinFunComp>(numY,lfuns);
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    myResult[(i * numX + k) * numY + h] = lfuns[h].x + y0*lfuns[h].y;
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
                
                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S12.4, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                REAL yn = myResult[(i * numX + k) * numY + numY-1]/yyY[k * numZ + numY-1];
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    if (h==0) { 
                        lfuns[0].x = 0.0;  
                        lfuns[0].y = 1.0;           
                    } else { 
                        lfuns[h].x = myResult[(i * numX + k) * numY + numY-h-1]/yyY[k * numZ + numY-h-1]; 
                        lfuns[h].y = -cY[k * numZ + numY-h-1]/yyY[k * numZ + numY-h-1]; 
                    }
                }
                inplaceScanInc<LinFunComp>(numY,lfuns);
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    myResult[(i * numX + k) * numY + numY-h-1] = lfuns[h].x + yn*lfuns[h].y;
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
            }
#if DEBUG
            cout << "S13 " << j << " " << i << endl;
#endif 
            //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
            //S13, distribution and array expansion applied to j and k due to dependency
            //in other iterations, invariant to the outer loop, therefore hoisted
            res[i] = myResult[(i * numX + myXindex) * numY + myYindex];
            //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
        }
    }
    free(mats);
    free(lfuns);
    free(myX       );
    free(myY       );
    free(myTimeline);
    free(myDxx     );
    free(myDyy     );
    free(myVarX    );
    free(myVarY    );
    free(aX        );
    free(bX        );
    free(cX        );
    free(aY        );
    free(bY        );
    free(cY        );
    free(myResult  );
    free(u         );
    free(v         );
    free(y         );
    free(yyX       );
    free(yyY       );
    free(dtInv     );
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
    /* Debugging plane transposition
    for (unsigned i = 0; i < planes; i++) {
        for (unsigned j = 0; j < rowsA; j++) {
            for (unsigned k = 0; k < colsA; k++) {
                if (A[(i * rowsA + j) * colsA + k] - trA[(i * colsA + k) * rowsA + j] > 0.0001 ||
                    A[(i * rowsA + j) * colsA + k] - trA[(i * colsA + k) * rowsA + j] < -0.0001) {
                    cout << "Error was not transposed correctly!" << endl;
                }
            }
        }   
    }*/
}

void   run_OrigCPUExpandKernelPrep(  
    const unsigned int&   outer,
    const unsigned int&   numX,
    const unsigned int&   numY,
    const unsigned int&   numT,
    const unsigned int&   B,  //blocksize simulated
    const REAL&           s0,
    const REAL&           t, 
    const REAL&           alpha, 
    const REAL&           nu, 
    const REAL&           beta,
          REAL*           res,   // [outer] RESULT
    DataCenter*           dc
) {
    //Alternative to vectors, reused memory locations, part of tridag
    MyReal4* mats = (MyReal4*) malloc(max(numX, numY) * sizeof(MyReal4));
    MyReal2* lfuns = (MyReal2*) malloc(max(numX, numY) * sizeof(MyReal2));

    /* Normally how we allocate, but using dc
    unsigned numZ = max(numX,numY);
    //Invariants for OUTER, never written to beyond initialization (read-only)
    //Therefore not expanded
    REAL* myX           = (REAL*) malloc(numX * sizeof(REAL)); 
    REAL* myY           = (REAL*) malloc(numY * sizeof(REAL));
    REAL* myTimeline    = (REAL*) malloc(numT * sizeof(REAL));
    REAL* myDxx         = (REAL*) malloc(numX * 4 * sizeof(REAL));
    REAL* myDyy         = (REAL*) malloc(numY * 4 * sizeof(REAL));
    REAL* trMyDxx       = (REAL*) malloc(4 * numX * sizeof(REAL));
    REAL* trMyDyy       = (REAL*) malloc(4 * numY * sizeof(REAL));
    //Expanded due to distribution over 2nd outer loop (numT)
    REAL* myVarX        = (REAL*) malloc(numT * numX * numY * sizeof(REAL));
    REAL* myVarY        = (REAL*) malloc(numT * numX * numY * sizeof(REAL));
    REAL* trMyVarX      = (REAL*) malloc(numT * numY * numX * sizeof(REAL));
    REAL* aX            = (REAL*) malloc(numT * numY * numX * sizeof(REAL));
    REAL* bX            = (REAL*) malloc(numT * numY * numX * sizeof(REAL));
    REAL* cX            = (REAL*) malloc(numT * numY * numX * sizeof(REAL));
    REAL* aY            = (REAL*) malloc(numT * numX * numY * sizeof(REAL));
    REAL* bY            = (REAL*) malloc(numT * numX * numY * sizeof(REAL));
    REAL* cY            = (REAL*) malloc(numT * numX * numY * sizeof(REAL));

    //Expanded after interchange and distribution of outer loops
    REAL* myResult      = (REAL*) malloc(outer * numX * numY * sizeof(REAL));
    REAL* trMyResult    = (REAL*) malloc(outer * numY * numX * sizeof(REAL));
    //Tridag initialization values
    REAL* u             = (REAL*) malloc(outer * numY * numX * sizeof(REAL));
    REAL* trU           = (REAL*) malloc(outer * numX * numY * sizeof(REAL));
    REAL* v             = (REAL*) malloc(outer * numX * numY * sizeof(REAL));
    REAL* trV           = (REAL*) malloc(outer * numY * numX * sizeof(REAL));
    //Tridag temporaries, thereby not expanded
    REAL* y             = (REAL*) malloc(outer * numX * numY * sizeof(REAL));
    REAL* yy            = (REAL*) malloc(numZ * sizeof(REAL));

    //Variable expanded to array
    REAL* dtInv         = (REAL*) malloc(numT * sizeof(REAL));
    REAL* dl            = (REAL*) malloc(numZ * sizeof(REAL));
    REAL* du            = (REAL*) malloc(numZ * sizeof(REAL));
    */

#if DEBUG
    cout << "S1" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S1, distributed, found to be invariant and no overrides, thereby not expanded
    //KERNEL NOTE: [=], already coalesced, no cross iteration dependencies
    for(unsigned block_off = 0; block_off < numT; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx < numT) {
                dc->myTimeline[gidx] = t*gidx/(numT-1);
            }
        }
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    unsigned myXindex = static_cast<unsigned>(s0/dx) % numX;

#if DEBUG
    cout << "S2" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S2, distributed, found to be invariant and no overrides, thereby not expanded
    //KERNEL NOTE: [=], already coalesced, no cross iteration dependencies
    for(unsigned block_off = 0; block_off < numX; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx < numX) {
                dc->myX[gidx] = gidx*dx - myXindex*dx + s0;
            }
        }
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    unsigned myYindex = static_cast<unsigned>(numY/2.0);

#if DEBUG
    cout << "S3" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S3, distributed, found to be invariant and no overrides, thereby not expanded
    //KERNEL NOTE: [=], already coalesced, no cross iteration dependencies
    for(unsigned block_off = 0; block_off < numY; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx < numY) {
                dc->myY[gidx] = gidx*dy - myYindex*dy + logAlpha;
            }
        }
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//


#if DEBUG
    cout << "S4" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S4, distributed, found to be invariant due to variable use and no overrides
    //since it is based on myX, thereby not expanded
    //KERNEL NOTE: extended dl and du into two kernels each to get coalesced access to myX
    //Transpose Dxx to get coalesced access through j columns instead of 4 columns

    //dl kernel 1, coalesced, no cross iteration dependencies
    for(unsigned block_off = 0; block_off < numX; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx > 0 && gidx < numY - 1) {
                dc->dl[gidx] = dc->myX[gidx];
            }
        }
    }
    //dl kernel 2, coalesced, no cross iteration dependencies
    for(unsigned block_off = 0; block_off < numX; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx > 0 && gidx < numY - 1) {
                dc->dl[gidx] -= dc->myX[gidx-1];
            }
        }
    }

    //du kernel 1, coalesced, no cross iteration dependencies
    for(unsigned block_off = 0; block_off < numX; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx > 0 && gidx < numY - 1) {
                dc->du[gidx] = dc->myX[gidx+1];
            }
        }
    }

    //du kernel 2, coalesced, no cross iteration dependencies
    for(unsigned block_off = 0; block_off < numX; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx > 0 && gidx < numY - 1) {
                dc->du[gidx] -= dc->myX[gidx];
            }
        }
    }

    //Unnecessary since myDxx is currently still empty, therefore fill
    //transformed myDxx and then transform into myDxx
    //matTranspose<REAL>(myDxx, trMyDxx, numX, 4);

    //kernel for Dxx transpose
    for(unsigned block_off = 0; block_off < 4 * numX; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            unsigned row = gidx / numX;
            unsigned col = gidx % numX;
            if (gidx < 4 * numX) {
                if (col > 0 && col < numX-1) {
                    dc->trMyDxx[gidx] = 
                        (row == 0) ?  2.0/dc->dl[col]/(dc->dl[col]+dc->du[col]) :
                    ((row == 1) ? -2.0*(1.0/dc->dl[col] + 1.0/dc->du[col])/(dc->dl[col]+dc->du[col]) :
                    ((row == 2) ?  2.0/dc->du[col]/(dc->dl[col]+dc->du[col]) :
                        0.0)); //row == 4 -> 0.0
                } else {
                    dc->trMyDxx[gidx] =  0.0;
                }
            }
        }
    }
    
    matTranspose<REAL>(dc->trMyDxx, dc->myDxx, 4, numX);
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
    cout << "S5" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S5, distributed, found to be invariant due to variable use and no overrides
    //since it is based on myY, thereby not expanded
    //KERNEL NOTE: extended dl and du into two kernels each to get coalesced access to myX
    //Transpose Dxx to get coalesced access through j columns instead of 4 columns
    
    //REAL* dl = (REAL*) malloc(numY * sizeof(REAL));
    //REAL* du = (REAL*) malloc(numY * sizeof(REAL));

    //dl kernel 1, coalesced, no cross iteration dependencies
    for(unsigned block_off = 0; block_off < numY; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx > 0 && gidx < numY - 1) {
                dc->dl[gidx] = dc->myY[gidx];
            }
        }
    }
    //dl kernel 2, coalesced, no cross iteration dependencies
    for(unsigned block_off = 0; block_off < numY; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx > 0 && gidx < numY - 1) {
                dc->dl[gidx] -= dc->myY[gidx-1];
            }
        }
    }

    //du kernel 1, coalesced, no cross iteration dependencies
    for(unsigned block_off = 0; block_off < numY; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx > 0 && gidx < numY - 1) {
                dc->du[gidx] = dc->myY[gidx+1];
            }
        }
    }

    //du kernel 2, coalesced, no cross iteration dependencies
    for(unsigned block_off = 0; block_off < numY; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx > 0 && gidx < numY - 1) {
                dc->du[gidx] -= dc->myY[gidx];
            }
        }
    }

    //Unnecessary since myDxx is currently still empty, therefore fill
    //transformed myDxx and then transform into myDxx
    //matTranspose<REAL>(myDxx, trMyDxx, numX, 4);

    //kernel for Dyy transpose
    for(unsigned block_off = 0; block_off < 4 * numY; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            unsigned row = gidx / numY;
            unsigned col = gidx % numY;
            if (gidx < 4 * numY) {
                if (col > 0 && col < numY-1) {
                    dc->trMyDyy[gidx] = 
                        (row == 0) ?  2.0/dc->dl[col]/(dc->dl[col]+dc->du[col]) :
                    ((row == 1) ? -2.0*(1.0/dc->dl[col] + 1.0/dc->du[col])/(dc->dl[col]+dc->du[col]) :
                    ((row == 2) ?  2.0/dc->du[col]/(dc->dl[col]+dc->du[col]) :
                        0.0)); //row == 4 -> 0.0
                } else {
                    dc->trMyDyy[gidx] =  0.0;
                }
            }
        }
    }

    //Separate kernel call
    matTranspose<REAL>(dc->trMyDyy, dc->myDyy, 4, numY);
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//


#if DEBUG
    cout << "S6" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S6, distributed and array expanded due to later write operations dependent on OUTER
    //KERNEL NOTE: flattened outer loop to allow 1 dimensional kernel
    for(unsigned block_off = 0; block_off < outer * numX * numY; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            unsigned row = gidx / (numX * numY);
            unsigned row_remain = gidx % (numX * numY);
            unsigned col = row_remain / numY;
            //Shared memory for myX
            if (gidx < outer * numX * numY) {
                REAL strike = 0.001*row;
                REAL payoff = max(dc->myX[col]-strike, (REAL)0.0);
                dc->myResult[gidx] = payoff;
            }
        }
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
    cout << "S7" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S7, distributed and array expanded due to dependency of j in myTimeline
    //Invariant to outer loop
    //KERNEL NOTE: flattened outer loop to allow 1 dimensional kernel and
    //adding each chunk of memory acceses separatly
    
    //VarX/Y kernel 1
    for(unsigned block_off = 0; block_off < numT * numX * numY; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            unsigned row_remain = gidx % (numX * numY);
            unsigned col = row_remain / numY;
            //Shared memory for myX
            if (gidx < numT * numX * numY) {
                dc->myVarX[gidx] = beta*log(dc->myX[col]);
                dc->myVarY[gidx] = alpha*log(dc->myX[col]);
            }
        }
    }

    //VarX/Y kernel 2
    for(unsigned block_off = 0; block_off < numT * numX * numY; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            unsigned row_remain = gidx % (numX * numY);
            unsigned depth = row_remain % numY;
            if (gidx < numT * numX * numY) {
                dc->myVarX[gidx] += dc->myY[depth];
                dc->myVarY[gidx] += dc->myY[depth];
            }
        }
    }

    //VarX/Y kernel 3
    for(unsigned block_off = 0; block_off < numT * numX * numY; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            unsigned row = gidx / (numX * numY);
            //Shared memory for myTimeline
            if (gidx < numT * numX * numY) {
                dc->myVarX[gidx] = exp(2.0*(dc->myVarX[gidx] - 0.5*nu*nu*dc->myTimeline[row]));
                dc->myVarY[gidx] = exp(2.0*(dc->myVarY[gidx] - 0.5*nu*nu*dc->myTimeline[row]));// nu*nu
            }
        }
    }

    matTransposePlane<REAL>(dc->myVarX, dc->trMyVarX, numT, numX, numY);

    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
    cout << "S8" << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S8, distributed and array expanded due to several SCC requires value
    //But again invariant to the outer iteration, therefore we only expand by numT
    //KERNEL NOTE: Separated into kernel chunks to allow coalesced access
    
    //dtInv kernel 1
    for(unsigned block_off = 0; block_off < numT; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx < numT - 1) {
                dc->dtInv[gidx] = dc->myTimeline[gidx+1];
            }
        }
    }

    //dtInv kernel 2
    for(unsigned block_off = 0; block_off < numT; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx < numT - 1) {
                dc->dtInv[gidx] = 1.0/(dc->dtInv[gidx]-dc->myTimeline[gidx]);
            }
        }
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
    

#if DEBUG
    cout << "S11 " << j << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S11, distribution and array expansion applied to j and k due to dependency
    //in other iterations, invarient to outer dimension and only ever read from
    //after this point (or entirely overwritten)
    //KERNEL NOTE: flattened outer loop to allow 1 dimensional kernel and 
    //separated into kernel chunks to allow coalesced access

    for(unsigned block_off = 0; block_off < numT * numY * numX; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            unsigned row = gidx / (numX * numY);
            unsigned depth = gidx % numX;
            //Shared memory for dtInv
            if (gidx < numT * numY * numX) {
                dc->aX[gidx] =		          - 0.5*(0.5*dc->trMyVarX[gidx]*dc->trMyDxx[0 * numX + depth]);
                dc->bX[gidx] = dc->dtInv[row] - 0.5*(0.5*dc->trMyVarX[gidx]*dc->trMyDxx[1 * numX + depth]);
                dc->cX[gidx] =		          - 0.5*(0.5*dc->trMyVarX[gidx]*dc->trMyDxx[2 * numX + depth]);
            }
        }
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
        cout << "S12 " << j << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S13, distribution and array expansion applied to j and k due to dependency
    //in other iterations, invarient to outer dimension and only ever read from
    //after this point
    for(unsigned block_off = 0; block_off < numT * numX * numY; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            unsigned row = gidx / (numX * numY);
            unsigned depth = gidx % numY;
            //Shared memory for dtInv
            if (gidx < numT * numX * numY) {
                dc->aY[gidx] =		          - 0.5*(0.5*dc->myVarY[gidx]*dc->trMyDyy[0 * numY + depth]);
                dc->bY[gidx] = dc->dtInv[row] - 0.5*(0.5*dc->myVarY[gidx]*dc->trMyDyy[1 * numY + depth]);
                dc->cY[gidx] =		          - 0.5*(0.5*dc->myVarY[gidx]*dc->trMyDyy[2 * numY + depth]);
            }
        }
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

    for(int j = numT-2;j>=0;--j) {// required sequential due to myResult cross iteration RAW
#if DEBUG
        cout << "S9 " << j << " " << i << endl;
#endif 
        //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
        //S9, fully distributed and expanded due to depedencies of both outer loops
        //and later use in other SCC
        //	explicit x

        matTransposePlane<REAL>(dc->myResult, dc->trMyResult, outer, numX, numY);

        //interchange k, h

        //u kernel 1
        for(unsigned block_off = 0; block_off < outer * numY * numX; block_off += B) {
            for (unsigned tid = 0; tid < B; tid ++) {
                unsigned gidx = block_off + tid;
                unsigned row_remain = gidx % (numX * numY);
                unsigned col = row_remain / numX;
                unsigned depth = row_remain % numX;

                //Input argument: j -> VarX plane
                unsigned VarXPlane = j;

                //pass dtInv as argument
                if (gidx < outer * numY * numX) {
                    dc->u[gidx] = dc->dtInv[VarXPlane]*dc->trMyResult[gidx];
            
                    if(depth > 0) { 
                        dc->u[gidx] += 0.5*( 0.5*dc->trMyVarX[(VarXPlane * numY + col) * numX + depth]*dc->trMyDxx[0 * numX + depth] ) 
                                    * dc->trMyResult[gidx-1];
                    }
                    dc->u[gidx]     +=  0.5*( 0.5*dc->trMyVarX[(VarXPlane * numY + col) * numX + depth]*dc->trMyDxx[1 * numX + depth] )
                                    * dc->trMyResult[gidx];
                    if(depth < numX-1) {
                        dc->u[gidx] += 0.5*( 0.5*dc->trMyVarX[(VarXPlane * numY + col) * numX + depth]*dc->trMyDxx[2 * numX + depth] )
                                    * dc->trMyResult[gidx+1];
                    }
                }
            }
        }
        //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
        cout << "S10 " << j << " " << i << endl;
#endif 
        //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
        //S10, fully distributed and expanded due to depedencies of both outer loops
        //and later use in other SCC
        //	explicit y

        for(unsigned block_off = 0; block_off < outer * numX * numY; block_off += B) {
            for (unsigned tid = 0; tid < B; tid ++) {
                unsigned gidx = block_off + tid;
                unsigned row_remain = gidx % (numX * numY);
                unsigned col = row_remain / numY;
                unsigned depth = row_remain % numY;
                
                //Input argument: j -> VarX plane
                unsigned VarYPlane = j;

                if (gidx < outer * numX * numY) {
                    dc->v[gidx] = 0.0;

                    if(depth > 0) {
                        dc->v[gidx] +=  ( 0.5*dc->myVarY[(VarYPlane * numX + col) * numY + depth]*dc->trMyDyy[0 * numY + depth] )
                                *  dc->myResult[gidx-1];
                    }
                    dc->v[gidx]     +=   ( 0.5*dc->myVarY[(VarYPlane * numX + col) * numY + depth]*dc->trMyDyy[1 * numY + depth] )
                                *  dc->myResult[gidx];
                    if(depth < numY-1) {
                        dc->v[gidx] +=  ( 0.5*dc->myVarY[(VarYPlane * numX + col) * numY + depth]*dc->trMyDyy[2 * numY + depth] )
                                *  dc->myResult[gidx+1];
                    } 
                }
            }
        }

        //transform v's inner dimensions
        matTransposePlane<REAL>(dc->v, dc->trV, outer, numX, numY);

        for(unsigned block_off = 0; block_off < outer * numY * numX; block_off += B) {
            for (unsigned tid = 0; tid < B; tid ++) {
                unsigned gidx = block_off + tid;
                if (gidx < outer * numX * numY) {
                    dc->u[gidx] += dc->trV[gidx];
                }
            }
        }

        //!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//
        //STOP kernalization since rest is tridag
        //!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//!//

        for( unsigned i = 0; i < outer; ++ i ) {
#if DEBUG
            cout << "S11.1-3 " << j << " " << i << endl;
#endif 
            //	implicit x
            for(unsigned k=0;k<numY;k++) {
                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S11.1, distribution and array expansion applied to j and k due to dependency
                //in other iterations, invariant to the outer loop, therefore hoisted
                MyReal4* mats = (MyReal4*)malloc(numX*sizeof(MyReal4));    // supposed to be in shared memory!
                REAL b0 = dc->bX[(j * numY + k) * numX + 0];
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    if (h==0) { 
                        mats[h].x = 1.0;  
                        mats[h].y = 0.0;          
                        mats[h].z = 0.0; 
                        mats[h].w = 1.0; 
                    } else { 
                        mats[h].x = dc->bX[(j * numY + k) * numX + h]; 
                        mats[h].y = -dc->aX[(j * numY + k) * numX + h]*dc->cX[(j * numY + k) * numX + h-1]; 
                        mats[h].z = 1.0; 
                        mats[h].w = 0.0; 
                    }
                }
                inplaceScanInc<MatMult2b2>(numX,mats);
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    dc->yy[h] = (mats[h].x*b0 + mats[h].y) / (mats[h].z*b0 + mats[h].w);
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S11.2, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                MyReal2* lfuns = (MyReal2*)malloc(numX*sizeof(MyReal2));
                REAL y0 = dc->u[(i * numY + k) * numX + 0];
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    if (h==0) { 
                        lfuns[0].x = 0.0;     
                        lfuns[0].y = 1.0;           
                    } else { 
                        lfuns[h].x = dc->u[(i * numY + k) * numX + h]; 
                        lfuns[h].y = -dc->aX[(j * numY + k) * numX + h]/dc->yy[h-1]; }
                }
                inplaceScanInc<LinFunComp>(numX,lfuns);
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    dc->u[(i * numY + k) * numX + h] = lfuns[h].x + y0*lfuns[h].y;
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S11.3, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                REAL yn = dc->u[(i * numY + k) * numX + numX-1]/dc->yy[numX-1];
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    if (h==0) { 
                        lfuns[0].x = 0.0;  
                        lfuns[0].y = 1.0;           
                    } else { 
                        lfuns[h].x = dc->u[(i * numY + k) * numX + numX-h-1]/dc->yy[numX-h-1]; 
                        lfuns[h].y = -dc->cX[(j * numY + k) * numX + numX-h-1]/dc->yy[numX-h-1]; }
                }
                inplaceScanInc<LinFunComp>(numX,lfuns);
                for(int h=0; h<numX; h++) { //parallel, map-like semantics
                    dc->u[(i * numY + k) * numX + numX-h-1] = lfuns[h].x + yn*lfuns[h].y;
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
            }
        }

        matTransposePlane<REAL>(dc->u, dc->trU, outer, numY, numX);

        //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
        //S12
        for(unsigned block_off = 0; block_off < outer * numX * numY; block_off += B) {
            for (unsigned tid = 0; tid < B; tid ++) {
                unsigned gidx = block_off + tid;
                
                //Input argument: j -> plane of y
                unsigned plane = j;

                if (gidx < outer * numX * numY) {
                    dc->y[gidx] = dc->dtInv[plane]*dc->trU[gidx] - 0.5*dc->v[gidx];
                }
            }
        }

        //return; ///////////RETURNING TO GET FIRST ITERATION u, v AND y

        //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

#if DEBUG
            cout << "S13.1-3 " << j << " " << i << endl;
#endif 
        for( unsigned i = 0; i < outer; ++ i ) {
            //	implicit y
            for(unsigned k=0;k<numX;k++) {
                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S13.1, distribution and array expansion applied to j and k due to dependency
                //in other iterations, invariant to the outer loop, therefore hoisted
                MyReal4* mats = (MyReal4*)malloc(numY*sizeof(MyReal4));   // supposed to be in shared memory!
                REAL b0 = dc->bY[(j * numX + k) * numY + 0];
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    if (h==0) { 
                        mats[h].x = 1.0;  
                        mats[h].y = 0.0;          
                        mats[h].z = 0.0; 
                        mats[h].w = 1.0; 
                    } else { 
                        mats[h].x = dc->bY[(j * numX + k) * numY + h]; 
                        mats[h].y = -dc->aY[(j * numX + k) * numY + h]*dc->cY[(j * numX + k) * numY + h-1]; 
                        mats[h].z = 1.0; 
                        mats[h].w = 0.0; 
                    }
                }
                inplaceScanInc<MatMult2b2>(numY,mats);
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    dc->yy[h] = (mats[h].x*b0 + mats[h].y) / (mats[h].z*b0 + mats[h].w);
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S13.2, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                MyReal2* lfuns = (MyReal2*)malloc(numY*sizeof(MyReal2));
                REAL y0 = dc->y[(i * numX + k) * numY + 0];
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    if (h==0) { 
                        lfuns[0].x = 0.0;  
                        lfuns[0].y = 1.0;           
                    } else { 
                        lfuns[h].x = dc->y[(i * numX + k) * numY + h]; 
                        lfuns[h].y = -dc->aY[(j * numX + k) * numY + h]/dc->yy[h-1]; 
                    }
                }
                inplaceScanInc<LinFunComp>(numY,lfuns);
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    dc->myResult[(i * numX + k) * numY + h] = lfuns[h].x + y0*lfuns[h].y;
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
                
                //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
                //S13.3, distribution and array expansion applied to i, j and k due to dependency
                //in other iterations
                REAL yn = dc->myResult[(i * numX + k) * numY + numY-1]/dc->yy[numY-1];
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    if (h==0) { 
                        lfuns[0].x = 0.0;  
                        lfuns[0].y = 1.0;           
                    } else { 
                        lfuns[h].x = dc->myResult[(i * numX + k) * numY + numY-h-1]/dc->yy[numY-h-1]; 
                        lfuns[h].y = -dc->cY[(j * numX + k) * numY + numY-h-1]/dc->yy[numY-h-1]; 
                    }
                }
                inplaceScanInc<LinFunComp>(numY,lfuns);
                for(int h=0; h<numY; h++) { //parallel, map-like semantics
                    dc->myResult[(i * numX + k) * numY + numY-h-1] = lfuns[h].x + yn*lfuns[h].y;
                }
                //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
            }
        }
    }
#if DEBUG
    cout << "S14 " << j << " " << i << endl;
#endif 
    //+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//+//
    //S14, questionable if we need to transpose as it weighs a lot compared to iterations
    for(unsigned block_off = 0; block_off < outer; block_off += B) {
        for (unsigned tid = 0; tid < B; tid ++) {
            unsigned gidx = block_off + tid;
            if (gidx < outer) {
                res[gidx] = dc->myResult[(gidx * numX + myXindex) * numY + myYindex];
            }
        }
    }
    //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
    free(mats);
    free(lfuns);
}


//#endif // PROJ_CORE_ORIG
