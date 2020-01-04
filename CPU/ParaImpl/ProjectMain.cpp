#include "OpenmpUtil.h"
#include "ParseInput.h"
#include "OriginalAlgorithm.h"
#include "SimpleParallelAlgorithm.h"
#include "InterchangedAlgorithm.h"

#define RUN_ALL false //will enable all and also experimental tests
#define RUN_CPU_EXPERIMENTAL true

typedef int (*fun)(const uint, const uint, const uint, const uint, const REAL, const REAL, const REAL, const REAL, const REAL, REAL*);

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

ReturnStat* RunStatsOnProgram(const char* name, fun f, 
    REAL* res, const uint outer, const uint numX, const uint numY, const uint numT, 
    const REAL s0, const REAL t, const REAL alpha, const REAL nu, const REAL beta) 
    {
    //61 characters long
    printf("\n[Running %-15s, outer: %3d, X: %3d, Y: %3d, T: %3d]\n", name, outer, numX, numY, numT);
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    int procs = f(outer, numX, numY, numT, s0, t, alpha, nu, beta, res);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    return new ReturnStat(t_diff.tv_sec*1e6+t_diff.tv_usec, procs);
}

void RunTestOnProgram(const char* title, fun f, REAL* expected, ReturnStat* expectedStats, const uint outer, const uint numX, const uint numY, const uint numT,
	const REAL s0, const REAL t, const REAL alpha, const REAL nu, const REAL beta) {
	REAL* res = (REAL*)malloc(outer * sizeof(REAL));
	ReturnStat* returnstatus = RunStatsOnProgram(title, f, res, outer, numX, numY, numT, s0, t, alpha, nu, beta);
	bool is_valid = compare_validate(res, expected, outer);
	writeStatsAndResult(is_valid, res, outer, false, returnstatus, expectedStats);
}


int main()
{
    unsigned int outer, numX, numY, numT; 
	const REAL s0 = 0.03, strike = 0.03, t = 5.0, alpha = 0.2, nu = 0.6, beta = 0.5;

    readDataSet( outer, numX, numY, numT ); 

    REAL* res_original = (REAL*)malloc(outer*sizeof(REAL));
    ReturnStat* originalStat = RunStatsOnProgram("Original", run_Original, res_original, outer, numX, numY, numT, s0, t, alpha, nu, beta);
    // Initial validation, rest is based on this result as validate gets a segmentation fault if repeated calls
    bool is_valid = validate ( res_original, outer );
    writeStatsAndResult( is_valid, res_original, outer, false, originalStat, originalStat );

    // If initial original program is correct, run rest
    if (is_valid) {
        //Simple parallized programs
        RunTestOnProgram("Simple Parallel", run_SimpleParallel, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);

#if RUN_CPU_EXPERIMENTAL || RUN_ALL
        RunTestOnProgram("Simple Parallel Static", run_SimpleParallelStatic, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
		RunTestOnProgram("Simple Parallel Dynamic", run_SimpleParallelDynamic, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
#endif

#if RUN_CPU_EXPERIMENTAL || RUN_ALL
		RunTestOnProgram("Interchanged", run_Interchanged, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
		RunTestOnProgram("Parallel Interchanged", run_InterchangedParallel, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
#endif  
    }

    return 0;
}
