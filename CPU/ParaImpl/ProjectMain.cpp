#include "OpenmpUtil.h"
#include "ParseInput.h"

//#include "ProjHelperFun.h"

typedef unsigned int uint;

bool compare_validate(REAL* result, REAL* expected, uint size) {
    bool isvalid = true;
    for (int i = 0; i < size; i++) {
        float err = fabs(result[i] - expected[i]);
        //d != d -> nan check
        if ( result[i] != result[i] || err > 0.00001 ) {
            cout << "Error at index [" << i << "] expected: " << expected[i] << " got: " << result[i] << endl;
            isvalid = false;
        }
    }
    return isvalid;
}

int main()
{
    unsigned int outer, numX, numY, numT; 
	const REAL s0 = 0.03, strike = 0.03, t = 5.0, alpha = 0.2, nu = 0.6, beta = 0.5;

    readDataSet( outer, numX, numY, numT ); 

    const int Ps = get_CPU_num_threads();
    REAL* res = (REAL*)malloc(outer*sizeof(REAL));
    unsigned long int origTime = OriginalProgram(res, outer, numX, numY, numT, s0, t, alpha, nu, beta);
    
    // Initial validation, rest is based on this result as validate gets a segmentation fault if repeated calls
    bool is_valid = validate ( res, outer );
    writeStatsAndResult( is_valid, res, outer, numX, numY, numT, false, 1/*Ps*/, origTime );

    // If initial original program is correct, run rest
    if (is_valid) {

    }

    return 0;
}

unsigned long int OriginalProgram(REAL* res, uint outer, uint numX, uint numY, uint numT, REAL s0, REAL t, REAL alpha, REAL nu, REAL beta) {
    cout<<"\n// Running Original, Sequential Project Program"<<endl;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    run_OrigCPU( outer, numX, numY, numT, s0, t, alpha, nu, beta, res );

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    return t_diff.tv_sec*1e6+t_diff.tv_usec;
}

