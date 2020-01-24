#include "OpenmpUtil.h"
#include "ParseInput.h"
#include "OriginalAlgorithm.h"
#include "SimpleParallelAlgorithm.h"
#include "InterchangedAlgorithm.h"
#include "KernelizedAlgorithm.h"

#include "cuda.h"
#include "cuda_runtime.h"

#define RUN_ALL false //will enable all and also experimental tests
#define RUN_CPU_EXPERIMENTAL true

typedef int (*funCPU)(const uint, const uint, const uint, const uint, const REAL, const REAL, const REAL, const REAL, const REAL, REAL*);
typedef int (*funGPU)(const uint, const uint, const uint, const uint, const REAL, const REAL, const REAL, const REAL, const REAL, const uint, REAL*);

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

template <typename funType>
ReturnStat* RunStatsOnProgram(const char* name, void* f, 
    REAL* res, const uint outer, const uint numX, const uint numY, const uint numT, 
    const REAL s0, const REAL t, const REAL alpha, const REAL nu, const REAL beta, const uint blocksize = 0) 
    {
    //61 characters long
    printf("\n[Running %-15s, outer: %3d, X: %3d, Y: %3d, T: %3d]\n", name, outer, numX, numY, numT);
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    
    int procs = 0;

    try
    {
        funCPU* funCPU = dynamic_cast<funType&>(f);
        if (funCPU) {
            procs = *funCPU(outer, numX, numY, numT, s0, t, alpha, nu, beta, res);
        } else {
            funGPU* funGPU = dynamic_cast<funType&>(f);
            procs = *funGPU(outer, numX, numY, numT, s0, t, alpha, nu, beta, blocksize, res);
        }
    }
    catch(const std::bad_cast& ex)
    {
        std::cout << "["<<ex.what()<<"]" << std::endl;
    }

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    return new ReturnStat(t_diff.tv_sec*1e6+t_diff.tv_usec, procs);
}

template <typename funType>
void RunTestOnProgram(const char* title, void* f, REAL* expected, ReturnStat* expectedStats, const uint outer, const uint numX, const uint numY, const uint numT,
	const REAL s0, const REAL t, const REAL alpha, const REAL nu, const REAL beta, const uint blocksize = 0) {
	REAL* res = (REAL*)malloc(outer * sizeof(REAL));
	ReturnStat* returnstatus = RunStatsOnProgram(title, f, res, outer, numX, numY, numT, s0, t, alpha, nu, beta, blocksize);
	bool is_valid = compare_validate(res, expected, outer);
	writeStatsAndResult(is_valid, res, outer, false, returnstatus, expectedStats);
}


int main()
{
    unsigned int outer, numX, numY, numT; 
	const REAL s0 = 0.03, strike = 0.03, t = 5.0, alpha = 0.2, nu = 0.6, beta = 0.5;

    readDataSet( outer, numX, numY, numT ); 

    REAL* res_original = (REAL*)malloc(outer*sizeof(REAL));
    ReturnStat* originalStat = RunStatsOnProgram<funCPU>("Original", (void*)&run_Original, res_original, outer, numX, numY, numT, s0, t, alpha, nu, beta);
    // Initial validation, rest is based on this result as validate gets a segmentation fault if repeated calls
    bool is_valid = validate ( res_original, outer );
    writeStatsAndResult( is_valid, res_original, outer, false, originalStat, originalStat );

    // If initial original program is correct, run rest
    if (is_valid) {
        //Simple parallized programs
        RunTestOnProgram<funCPU>("Simple Parallel", (void*)&run_SimpleParallel, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);

#if RUN_CPU_EXPERIMENTAL || RUN_ALL
        RunTestOnProgram<funCPU>("Simple Parallel Static", (void*)&run_SimpleParallelStatic, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
		RunTestOnProgram<funCPU>("Simple Parallel Dynamic", (void*)&run_SimpleParallelDynamic, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
#endif


		RunTestOnProgram<funCPU>("Interchanged", (void*)&run_Interchanged, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
#if RUN_CPU_EXPERIMENTAL || RUN_ALL
        RunTestOnProgram<funCPU>("Interchanged Optimized", (void*)&run_InterchangedAlternative, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
#endif  
		RunTestOnProgram<funCPU>("Parallel Interchanged", (void*)&run_InterchangedParallel, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
#if RUN_CPU_EXPERIMENTAL || RUN_ALL
        RunTestOnProgram<funCPU>("Parallel Interchanged Optimized", (void*)&run_InterchangedParallelAlternative, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
#endif  

        RunTestOnProgram<funGPU>("Kernelized", (void*)&run_Kernelized, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
    }

    return 0;
}
