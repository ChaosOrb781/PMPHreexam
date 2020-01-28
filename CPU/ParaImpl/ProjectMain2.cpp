#include "OpenmpUtil.h"
#include "ParseInput.h"
#include "OriginalAlgorithm.h"
#include "DistributedAlgorithm.h"
#include "KernelAlgorithm.h"

#include "cuda.h"
#include "cuda_runtime.h"

#define RUN_ALL false //will enable all and also experimental tests
#define RUN_CPU_EXPERIMENTAL true

#define Block 256

typedef int (*fun)();
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
ReturnStat* RunStatsOnProgram(const char* name, fun f, 
    REAL* res, const uint outer, const uint numX, const uint numY, const uint numT, 
    const REAL s0, const REAL t, const REAL alpha, const REAL nu, const REAL beta, const uint blocksize = 1) 
    {
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    
    int procs = 0;
    if (is_same<funType, funCPU>::value)
        procs = ((funCPU) f)(outer, numX, numY, numT, s0, t, alpha, nu, beta, res);
    else if (is_same<funType, funGPU>::value)
        procs = ((funGPU) f)(outer, numX, numY, numT, s0, t, alpha, nu, beta, blocksize, res);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    return new ReturnStat(t_diff.tv_sec*1e6+t_diff.tv_usec, procs);
}

template <typename funType>
void RunTestOnProgram(const char* title, fun f, REAL* expected, ReturnStat* expectedStats, const uint outer, const uint numX, const uint numY, const uint numT,
	const REAL s0, const REAL t, const REAL alpha, const REAL nu, const REAL beta, const uint blocksize = 1) {
	REAL* res = (REAL*)malloc(outer * sizeof(REAL));
	ReturnStat* returnstatus = RunStatsOnProgram<funType>(title, f, res, outer, numX, numY, numT, s0, t, alpha, nu, beta, blocksize);
	bool is_valid = compare_validate(res, expected, outer);

    char buffer[256];
    if (blocksize > 1) {
        sprintf(buffer, "%s (%d)", title, Block);
    } else {
        sprintf(buffer, "%s", title);
    }
	writeStatsAndResult(buffer, is_valid, res, outer, false, returnstatus, expectedStats);
}


int main()
{
    unsigned int outer, numX, numY, numT; 
	const REAL s0 = 0.03, strike = 0.03, t = 5.0, alpha = 0.2, nu = 0.6, beta = 0.5;

    readDataSet( outer, numX, numY, numT ); 

    //printf("\n[Running programs with: outer: %3d, X: %3d, Y: %3d, T: %3d, B: %3d]\n", outer, numX, numY, numT, Block);
    //printf("|_%s_________________________|_%s_|_%s_|_%s_|_%s_|\n", "Program", "VALIDITY", "# OF THREADS", "TIME TAKEN", "SPEEDUP");

    REAL* res_original = (REAL*)malloc(outer*sizeof(REAL));
    ReturnStat* originalStat = RunStatsOnProgram<funCPU>("Original", (fun)run_Original, res_original, outer, numX, numY, numT, s0, t, alpha, nu, beta);
    // Initial validation, rest is based on this result as validate gets a segmentation fault if repeated calls
    bool is_valid = validate ( res_original, outer );
    writeStatsAndResult("Original", is_valid, res_original, outer, false, originalStat, originalStat );

    // If initial original program is correct, run rest
    if (is_valid) {
        //Simple parallized programs
        //RunTestOnProgram<funCPU>("Simple Parallel", (fun)run_SimpleParallel, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
#if RUN_CPU_EXPERIMENTAL || RUN_ALL
        //RunTestOnProgram<funCPU>("Simple Parallel Static", (fun)run_SimpleParallelStatic, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
		//RunTestOnProgram<funCPU>("Simple Parallel Dynamic", (fun)run_SimpleParallelDynamic, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
#endif
		//RunTestOnProgram<funCPU>("Interchanged", (fun)run_Interchanged, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
        //RunTestOnProgram<funCPU>("Interchanged Optimized", (fun)run_InterchangedAlternative, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
		//RunTestOnProgram<funCPU>("Parallel Interchanged", (fun)run_InterchangedParallel, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
        //RunTestOnProgram<funCPU>("Parallel Interchanged Optimized", (fun)run_InterchangedParallelAlternative, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta);
        //RunTestOnProgram<funGPU>("Kernelized", (fun)run_SimpleKernelized, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta, Block);
        //RunTestOnProgram<funGPU>("Kernelized Parallel", (fun)run_SimpleKernelized_Parallel, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta, Block);
        //RunTestOnProgram<funGPU>("Kernelized Flat", (fun)run_Kernelized_Rollback, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta, Block);
        //RunTestOnProgram<funGPU>("Kernelized Flat Parallel", (fun)run_Kernelized_Rollback_Parallel, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta, Block);
        //RunTestOnProgram<funGPU>("Kernelized Dist", (fun)run_Kernelized_Rollback_Dist, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta, Block);
        //RunTestOnProgram<funGPU>("Kernelized Dist Parallel", (fun)run_Kernelized_Rollback_Dist_Parallel, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta, Block);
        //RunTestOnProgram<funGPU>("Kernelized Dist 2", (fun)run_Kernelized_Rollback_Dist_Alt, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta, Block);
        //RunTestOnProgram<funGPU>("Kernelized Dist 2 Parallel", (fun)run_Kernelized_Rollback_Dist_Alt_Parallel, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta, Block);
        RunTestOnProgram<funGPU>("Distributed Rollback", (fun)run_Distributed, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta, Block);
        RunTestOnProgram<funGPU>("Distributed Parallel", (fun)run_Distributed_Parallel, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta, Block);
        RunTestOnProgram<funGPU>("Simple Kernel", (fun)run_SimpleKernel, res_original, originalStat, outer, numX, numY, numT, s0, t, alpha, nu, beta, Block);
    }

    return 0;
}
