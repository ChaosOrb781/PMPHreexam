#include "OpenmpUtil.h"
#include "ParseInput.h"
#include "SimpleParallelAlgorithm.h"

#define RUN_ALL false //will enable all and also experimental tests
#define RUN_CPU_EXPERIMENTAL false

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
        REAL* res_simpleParallel = (REAL*)malloc(outer*sizeof(REAL));
        ReturnStat* simpelParallelStat = RunStatsOnProgram("SimpleParallel", run_SimpleParallel, res_simpleParallel, outer, numX, numY, numT, s0, t, alpha, nu, beta);
        is_valid = compare_validate ( res_simpleParallel, res_original, outer );
        writeStatsAndResult( is_valid, res_simpleParallel, outer, false, simpelParallelStat, originalStat );

#if RUN_CPU_EXPERIMENTAL || RUN_ALL
        REAL* res_simpleParallelStatic = (REAL*)malloc(outer*sizeof(REAL));
        ReturnStat* simpelParallelStaticStat = RunStatsOnProgram("SimpleParallelStatic", run_SimpleParallelStatic, res_simpleParallelStatic, outer, numX, numY, numT, s0, t, alpha, nu, beta);
        is_valid = compare_validate ( res_simpleParallel, res_original, outer );
        writeStatsAndResult( is_valid, res_simpleParallelStatic, outer, false, simpelParallelStaticStat, originalStat );

        REAL* res_simpleParallelDynamic = (REAL*)malloc(outer*sizeof(REAL));
        ReturnStat* simpelParallelDynamicStat = RunStatsOnProgram("SimpleParallelDynamic", run_SimpleParallelDynamic, res_simpleParallelDynamic, outer, numX, numY, numT, s0, t, alpha, nu, beta);
        is_valid = compare_validate ( res_simpleParallel, res_original, outer );
        writeStatsAndResult( is_valid, res_simpleParallelDynamic, outer, false, simpelParallelDynamicStat, originalStat );
#endif

        
    }

    return 0;
}
