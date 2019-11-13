#include "./includes/ParseInput.cu.h"
#include "./includes/ProjHelperFun.cu.h"
#include "./kernels/CUDAKernels.cu.h"
#include "./kernels/TridagKernel.cu.h"
#include "./TridagPar.cu.h"

#define TILE_myDzz 4
#define TILE       16
#define BLOCK      128
#define TIME_INDIVIDUAL false
#define DEBUGCUDA  false
#define FLOWCUDA   false
#define RUN_CPU    false
#define RUN_GPU    true

#define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      cout << "GPUassert: " << cudaGetErrorString(code) << " in " << file << " : "<< line << endl;
      if (abort) exit(code);
   }
}

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
}

int main()
{
    unsigned int OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T; 
	const REAL s0 = 0.03, /*strike = 0.03,*/ t = 5.0, alpha = 0.2, nu = 0.6, beta = 0.5;

    readDataSet( OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T ); 

    //const int Ps = get_CPU_num_threads();
    REAL* res_cpu1 = (REAL*)malloc(OUTER_LOOP_COUNT*sizeof(REAL));
    REAL* res_cpu2 = (REAL*)malloc(OUTER_LOOP_COUNT*sizeof(REAL));
    REAL* res_cpu3 = (REAL*)malloc(OUTER_LOOP_COUNT*sizeof(REAL));
    REAL* res_cpu4 = (REAL*)malloc(OUTER_LOOP_COUNT*sizeof(REAL));
    REAL* h_res_gpu = (REAL*)malloc(OUTER_LOOP_COUNT*sizeof(REAL));

    DataCenter cpu_arrs(OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T);

    {   // Original Program (Sequential CPU Execution)
        cout<<"\n// Running run_OrigCPUExpand, Sequential Project Program"<<endl;

        unsigned long int elapsed = 0;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        run_OrigCPUExpand( OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T, s0, t, alpha, nu, beta, res_cpu1 );

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;

        // validation and writeback of the result
        bool is_valid = validate   ( res_cpu1, OUTER_LOOP_COUNT );
        writeStatsAndResult( is_valid, res_cpu1, OUTER_LOOP_COUNT, 
                             NUM_X, NUM_Y, NUM_T, false, 1/*Ps*/, elapsed );        
    }
#if RUN_CPU
    {
        cout<<"\n// Running run_OrigCPUExpand1stOUTER, Sequential Project Program"<<endl;

        unsigned long int elapsed = 0;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        run_OrigCPUExpand1stOUTER( OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T, s0, t, alpha, nu, beta, res_cpu2 );

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;

        // validation and writeback of the result
        bool is_valid = compare_validate ( res_cpu2, res_cpu1, OUTER_LOOP_COUNT );
        writeStatsAndResult( is_valid, res_cpu2, OUTER_LOOP_COUNT, 
            NUM_X, NUM_Y, NUM_T, false, 1, elapsed );       
    }

    {
        cout<<"\n// Running run_OrigCPUExpand2ndOUTER, Sequential Project Program"<<endl;

        cout<<"Starting..."<<endl;

        unsigned long int elapsed = 0;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        run_OrigCPUExpand2ndOUTER( OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T, s0, t, alpha, nu, beta, res_cpu3 );

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;
        cout<<"Ended..."<<endl;

        // validation and writeback of the result
        bool is_valid = compare_validate ( res_cpu3, res_cpu1, OUTER_LOOP_COUNT );
        writeStatsAndResult( is_valid, res_cpu3, OUTER_LOOP_COUNT, 
            NUM_X, NUM_Y, NUM_T, false, 1, elapsed );       
    }
#endif
#if RUN_GPU
    {
        cout<<"\n// Running run_OrigCPUExpandKernelPrep, Sequential Project Program"<<endl;

        cout<<"Starting..."<<endl;

        unsigned long int elapsed = 0;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        run_OrigCPUExpandKernelPrep( OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T, BLOCK, s0, t, alpha, nu, beta, res_cpu4, &cpu_arrs );

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;
        cout<<"Ended..."<<endl;

        // validation and writeback of the result
        bool is_valid = compare_validate ( res_cpu4, res_cpu1, OUTER_LOOP_COUNT );
        writeStatsAndResult( is_valid, res_cpu4, OUTER_LOOP_COUNT, 
            NUM_X, NUM_Y, NUM_T, false, 1, elapsed );       
    }

    {
        cout<<"\n// Running CUDA (outer loops), Parallel Project Program"<<endl;

#if DEBUGCUDA
        cout << "Starting memory allocation..." << endl;
#endif
        
        unsigned numZ = max(NUM_X,NUM_Y);
        //Invariants for OUTER, never written to beyond initialization (read-only)
        //Therefore not expanded
        REAL* d_myX       ; cudaMalloc((void**) &d_myX         , NUM_X * sizeof(REAL));
        REAL* d_myY       ; cudaMalloc((void**) &d_myY         , NUM_Y * sizeof(REAL));
        REAL* d_myTimeline; cudaMalloc((void**) &d_myTimeline  , NUM_T * sizeof(REAL));
        REAL* d_myDxx     ; cudaMalloc((void**) &d_myDxx       , NUM_X * 4 * sizeof(REAL));
        REAL* d_myDyy     ; cudaMalloc((void**) &d_myDyy       , NUM_Y * 4 * sizeof(REAL));
        REAL* d_trMyDxx   ; cudaMalloc((void**) &d_trMyDxx     , 4 * NUM_X * sizeof(REAL));
        REAL* d_trMyDyy   ; cudaMalloc((void**) &d_trMyDyy     , 4 * NUM_Y * sizeof(REAL));
        //Expanded due to distribution over 2nd outer loop
        REAL* d_myVarX    ; cudaMalloc((void**) &d_myVarX      , NUM_T * NUM_X * NUM_Y * sizeof(REAL));
        REAL* d_myVarY    ; cudaMalloc((void**) &d_myVarY      , NUM_T * NUM_X * NUM_Y * sizeof(REAL));
        REAL* d_trMyVarX  ; cudaMalloc((void**) &d_trMyVarX    , NUM_T * NUM_Y * NUM_X * sizeof(REAL));
        REAL* d_aX        ; cudaMalloc((void**) &d_aX          , NUM_T * NUM_Y * NUM_X * sizeof(REAL));
        REAL* d_bX        ; cudaMalloc((void**) &d_bX          , NUM_T * NUM_Y * NUM_X * sizeof(REAL));
        REAL* d_cX        ; cudaMalloc((void**) &d_cX          , NUM_T * NUM_Y * NUM_X * sizeof(REAL));
        REAL* d_aY        ; cudaMalloc((void**) &d_aY          , NUM_T * NUM_X * NUM_Y * sizeof(REAL));
        REAL* d_bY        ; cudaMalloc((void**) &d_bY          , NUM_T * NUM_X * NUM_Y * sizeof(REAL));
        REAL* d_cY        ; cudaMalloc((void**) &d_cY          , NUM_T * NUM_X * NUM_Y * sizeof(REAL));

        //Expanded after interchange and distribution of outer loop
        REAL* d_myResult  ; cudaMalloc((void**) &d_myResult    , OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL));
        REAL* d_trMyResult; cudaMalloc((void**) &d_trMyResult  , OUTER_LOOP_COUNT * NUM_Y * NUM_X * sizeof(REAL));
        //Tridag initialization value)
        REAL* d_u         ; cudaMalloc((void**) &d_u           , OUTER_LOOP_COUNT * NUM_Y * NUM_X * sizeof(REAL));
        REAL* d_trU       ; cudaMalloc((void**) &d_trU         , OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL));
        REAL* d_v         ; cudaMalloc((void**) &d_v           , OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL));
        REAL* d_trV       ; cudaMalloc((void**) &d_trV         , OUTER_LOOP_COUNT * NUM_Y * NUM_X * sizeof(REAL));
        //Tridag temporaries, thereby not expand
        REAL* d_y         ; cudaMalloc((void**) &d_y           , OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL));
        REAL* d_yy        ; cudaMalloc((void**) &d_yy          , numZ * sizeof(REAL));

        //Variable expanded to array
        REAL* d_dtInv     ; cudaMalloc((void**) &d_dtInv       , NUM_T * sizeof(REAL));
        REAL* h_dtInv     = (REAL*) malloc(NUM_T * sizeof(REAL));
        REAL* d_dl        ; cudaMalloc((void**) &d_dl          , numZ * sizeof(REAL));
        REAL* d_du        ; cudaMalloc((void**) &d_du          , numZ * sizeof(REAL));

        //Return array:
        REAL* d_res_gpu   ; cudaMalloc((void**) &d_res_gpu     , OUTER_LOOP_COUNT*sizeof(REAL));

        //Host memory used for tridag
        REAL* h_aX        = (REAL*) malloc(NUM_T * NUM_Y * NUM_X * sizeof(REAL));
        REAL* h_bX        = (REAL*) malloc(NUM_T * NUM_Y * NUM_X * sizeof(REAL));
        REAL* h_cX        = (REAL*) malloc(NUM_T * NUM_Y * NUM_X * sizeof(REAL));
        REAL* h_aY        = (REAL*) malloc(NUM_T * NUM_Y * NUM_X * sizeof(REAL));
        REAL* h_bY        = (REAL*) malloc(NUM_T * NUM_Y * NUM_X * sizeof(REAL));
        REAL* h_cY        = (REAL*) malloc(NUM_T * NUM_Y * NUM_X * sizeof(REAL));
        REAL* h_y         = (REAL*) malloc(OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL));
        REAL* h_yy        = (REAL*) malloc(numZ * sizeof(REAL));
        REAL* h_myResult  = (REAL*) malloc(OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL));
        REAL* h_u         = (REAL*) malloc(OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL));

        //Temporary used for debuggin'/testing purposes
        REAL* temp = (REAL*) malloc(NUM_T * NUM_X * NUM_Y * sizeof(REAL));
        REAL* tempT = (REAL*) malloc(NUM_T * NUM_X * NUM_Y * sizeof(REAL));

#if DEBUGCUDA || FLOWCUDA
        cout << "Ended allocation, starting on constant calculation..." << endl;
#endif

        REAL dx = (20.0*alpha*s0*sqrt(t)) / NUM_X;
        uint myXindex = static_cast<uint>(s0/((20.0*alpha*s0*sqrt(t))/NUM_X));

        REAL dy = (10.0*nu*sqrt(t)) / NUM_Y;
        REAL logAlpha = log(alpha);
        uint myYindex = static_cast<uint>(NUM_Y / 2.0);

#if DEBUGCUDA
        cout << "/////////////////RUNTIME KERNEL RESULTS.../////////////////" << endl;
#endif

        unsigned long int elapsed = 0;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        //Initialization kernel S1, S2, S3 (runs independently)
#if FLOWCUDA
        cout << "    MyTimeline..." << endl;
#endif
        MyTimeline <<< ((BLOCK - 1 + NUM_T) / BLOCK), BLOCK >>> (d_myTimeline, t, NUM_T); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if DEBUGCUDA
        gpuErr ( cudaMemcpy(temp, d_myTimeline, NUM_T * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myTimeline", cpu_arrs.myTimeline, temp, NUM_T, true, false)) {
            cout << "-> d_myTimeline is correct" << endl;
        }
#endif
#if FLOWCUDA
        cout << "    MyX..." << endl;
#endif
        MyX        <<< ((BLOCK - 1 + NUM_X) / BLOCK), BLOCK >>> (d_myX, s0, dx, myXindex, NUM_X); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if DEBUGCUDA
        gpuErr ( cudaMemcpy(temp, d_myX, NUM_X * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myX", cpu_arrs.myX, temp, NUM_X, true, false)) {
            cout << "-> d_myX is correct" << endl;
        }
#endif
#if FLOWCUDA
        cout << "    MyY..." << endl;
#endif
        MyY        <<< ((BLOCK - 1 + NUM_Y) / BLOCK), BLOCK >>> (d_myY, logAlpha, dy, myYindex, NUM_Y); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if DEBUGCUDA
        gpuErr ( cudaMemcpy(temp, d_myY, NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myY", cpu_arrs.myY, temp, NUM_Y, true, false)) {
            cout << "-> d_myY is correct" << endl;
        }
#endif
#if FLOWCUDA
        cout << "   MyVarXY1..." << endl;
#endif
        //MyVarXY1 <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK, (NUM_X + 1) * sizeof(REAL) >>> (d_myVarX, d_myVarY, d_myX, beta, alpha, NUM_T, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
        MyVarXY1_dif <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_myVarX, d_myVarY, d_myX, beta, alpha, NUM_T, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if FLOWCUDA
        cout << "   DtInv1..." << endl;
#endif
        DtInv1   <<< ((BLOCK - 1 + NUM_T) / BLOCK), BLOCK >>> (d_dtInv, d_myTimeline, NUM_T); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if FLOWCUDA
        cout << "   Dl1..." << endl;
#endif
        Dl1      <<< ((BLOCK - 1 + NUM_X) / BLOCK), BLOCK >>> (d_myX, d_dl, NUM_X); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if FLOWCUDA
        cout << "   Du1..." << endl;
#endif
        Du1      <<< ((BLOCK - 1 + NUM_X) / BLOCK), BLOCK >>> (d_myX, d_du, NUM_X); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();

#if FLOWCUDA
        cout << "   DtInv2..." << endl;
#endif
        DtInv2   <<< ((BLOCK - 1 + NUM_T) / BLOCK), BLOCK >>> (d_dtInv, d_myTimeline, NUM_T); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if DEBUGCUDA
        gpuErr ( cudaMemcpy(temp, d_dtInv, NUM_T * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_dtInv", cpu_arrs.dtInv, temp, NUM_T, true, false)) {
            cout << "-> d_dtInv is correct" << endl;
        }
#endif
#if FLOWCUDA
        cout << "   MyVarXY2..." << endl;
#endif
        MyVarXY2 <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_myVarX, d_myVarY, d_myY, NUM_T, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if FLOWCUDA
        cout << "   Dl2..." << endl;
#endif
        Dl2      <<< ((BLOCK - 1 + NUM_X) / BLOCK), BLOCK >>> (d_myX, d_dl, NUM_X); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();

#if FLOWCUDA
        cout << "   Du2..." << endl;
#endif
        Du2      <<< ((BLOCK - 1 + NUM_X) / BLOCK), BLOCK >>> (d_myX, d_du, NUM_X); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();

        gpuErr ( cudaMemcpy(h_dtInv, d_dtInv, NUM_T * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    
#if FLOWCUDA
        cout << "   MyVarXY3..." << endl;
#endif
        //MyVarXY3 <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK, (NUM_T + 1) * sizeof(REAL) >>> (d_myVarX, d_myVarY, d_myTimeline, nu, NUM_T, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
        MyVarXY3_dif <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_myVarX, d_myVarY, d_myTimeline, nu, NUM_T, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if DEBUGCUDA
        gpuErr ( cudaMemcpy(temp, d_myVarX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myVarX", cpu_arrs.myVarX, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "-> d_myVarX is correct" << endl;
        }
        gpuErr ( cudaMemcpy(temp, d_myVarY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myVarY", cpu_arrs.myVarY, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "-> d_myVarY is correct" << endl;
        }
#endif
        //MyVarXY <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_myVarX, d_myVarY, d_myX, d_myY, d_myTimeline, beta, alpha, nu, NUM_T, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
        //MyVarX <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_myVarX, d_myX, d_myY, d_myTimeline, beta, nu, NUM_T, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
        //MyVarY <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_myVarY, d_myX, d_myY, d_myTimeline, alpha, nu, NUM_T, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
#if FLOWCUDA
        cout << "   TrMyDxx..." << endl;
#endif
        TrMyDzz  <<< ((BLOCK - 1 + 4 * NUM_X) / BLOCK), BLOCK >>> (d_trMyDxx, d_dl, d_du, NUM_X); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if DEBUGCUDA
        gpuErr ( cudaMemcpy(temp, d_trMyDxx, NUM_X * 4 * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_trMyDxx", cpu_arrs.trMyDxx, temp, NUM_X * 4, true, false)) {
            cout << "-> d_trMyDxx is correct" << endl;
        }
#endif
#if FLOWCUDA
        cout << "   MyResult..." << endl;
#endif
        //MyResult <<< ((BLOCK - 1 + OUTER_LOOP_COUNT * NUM_X * NUM_Y) / BLOCK), BLOCK, (NUM_X + 1) * sizeof(REAL) >>> (d_myResult, d_myX, OUTER_LOOP_COUNT, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
        MyResult_dif <<< ((BLOCK - 1 + OUTER_LOOP_COUNT * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_myResult, d_myX, OUTER_LOOP_COUNT, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if DEBUGCUDA
        REAL* tempResultCPU = (REAL*) malloc(OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL));
        for(unsigned block_off = 0; block_off < OUTER_LOOP_COUNT * NUM_X * NUM_Y; block_off += BLOCK) {
            for (unsigned tid = 0; tid < BLOCK; tid ++) {
                unsigned gidx = block_off + tid;
                unsigned row = gidx / (NUM_X * NUM_Y);
                unsigned row_remain = gidx % (NUM_X * NUM_Y);
                unsigned col = row_remain / NUM_Y;
                //Shared memory for myX
                if (gidx < OUTER_LOOP_COUNT * NUM_X * NUM_Y) {
                    REAL strike = 0.001*row;
                    REAL payoff = max(cpu_arrs.myX[col]-strike, (REAL)0.0);
                    tempResultCPU[gidx] = payoff;
                }
            }
        }
        gpuErr ( cudaMemcpy(temp, d_myResult, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myResult", tempResultCPU, temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y, true, false)) {
            cout << "-> myResult initialized correctly" << endl;
        }
        free(tempResultCPU);
#endif
#if FLOWCUDA
        cout << "   trMyVarX..." << endl;
#endif
        for (int i = 0; i < NUM_T; i++) {
            transpose_nosync<REAL, TILE> (d_myVarX + i * NUM_X * NUM_Y, d_trMyVarX + i * NUM_X * NUM_Y, NUM_X, NUM_Y);
        }
        cudaDeviceSynchronize();
#if DEBUGCUDA
        gpuErr ( cudaMemcpy(temp, d_trMyVarX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_trMyVarX", cpu_arrs.trMyVarX, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "-> d_trMyVarX is correct" << endl;
        }
#endif
#if FLOWCUDA
        cout << "   aX, bX, cX..." << endl;
#endif
        //ABCX <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK, (NUM_T + 1) * sizeof(REAL) >>> (d_aX, d_bX, d_cX, d_dtInv, d_trMyVarX, d_trMyDxx, NUM_T, NUM_Y, NUM_X); gpuErr(cudaPeekAtLastError());
        ABCX_dif <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_aX, d_bX, d_cX, d_dtInv, d_trMyVarX, d_trMyDxx, NUM_T, NUM_Y, NUM_X); gpuErr(cudaPeekAtLastError());
        //AX_dif <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_aX, d_dtInv, d_trMyVarX, d_trMyDxx, NUM_T, NUM_Y, NUM_X); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if DEBUGCUDA
        //Check d_aX
        gpuErr ( cudaMemcpy(temp, d_aX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_aX", cpu_arrs.aX, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "-> d_aX is correct" << endl;
        }
        
        //Check d_bX
        gpuErr ( cudaMemcpy(temp, d_bX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_bX", cpu_arrs.bX, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "-> d_bX is correct" << endl;
        }

        //Check d_cX
        gpuErr ( cudaMemcpy(temp, d_aX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_cX", cpu_arrs.cX, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "-> d_cX is correct" << endl;
        }
#endif

#if FLOWCUDA
        cout << "   myDxx..." << endl;
#endif
        transpose<REAL, TILE> (d_trMyDxx, d_myDxx, 4, NUM_X);
#if DEBUGCUDA
        gpuErr ( cudaMemcpy(temp, d_myDxx, NUM_X * 4 * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myDxx", cpu_arrs.myDxx, temp, NUM_X * 4, true, false)) {
            cout << "-> d_myDxx is correct" << endl;
        }
#endif
#if FLOWCUDA
        cout << "   Dl1..." << endl;
#endif
        Dl1 <<< ((BLOCK - 1 + NUM_Y) / BLOCK), BLOCK >>> (d_myY, d_dl, NUM_Y); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if FLOWCUDA
        cout << "   Du1..." << endl;
#endif
        Du1 <<< ((BLOCK - 1 + NUM_Y) / BLOCK), BLOCK >>> (d_myY, d_du, NUM_Y); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();

#if FLOWCUDA
        cout << "   Dl2..." << endl;
#endif
        Dl2 <<< ((BLOCK - 1 + NUM_Y) / BLOCK), BLOCK >>> (d_myY, d_dl, NUM_Y); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if FLOWCUDA
        cout << "   Du2..." << endl;
#endif
        Du2 <<< ((BLOCK - 1 + NUM_Y) / BLOCK), BLOCK >>> (d_myY, d_du, NUM_Y); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();

#if FLOWCUDA
        cout << "   TrMyDyy..." << endl;
#endif
        TrMyDzz <<< ((BLOCK - 1 + 4 * NUM_Y) / BLOCK), BLOCK >>> (d_trMyDyy, d_dl, d_du, NUM_Y); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if DEBUGCUDA
        gpuErr ( cudaMemcpy(temp, d_trMyDyy, NUM_Y * 4 * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_trMyDyy", cpu_arrs.trMyDyy, temp, NUM_Y * 4, true, false)) {
            cout << "-> d_trMyDyy is correct" << endl;
        }
#endif
#if FLOWCUDA
        cout << "   myDyy..." << endl;
#endif
        transpose<REAL, TILE> (d_trMyDyy, d_myDyy, 4, NUM_Y);
#if DEBUGCUDA
        gpuErr ( cudaMemcpy(temp, d_myDyy, NUM_Y * 4 * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myDyy", cpu_arrs.myDyy, temp, NUM_Y * 4, true, false)) {
            cout << "-> d_myDyy is correct" << endl;
        }
#endif
#if FLOWCUDA
        cout << "   aY, bY, cY..." << endl;
#endif
        //ABCY <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK, (NUM_T + 1) * sizeof(REAL) >>> (d_aX, d_bX, d_cX, d_dtInv, d_myVarY, d_trMyDyy, NUM_T, NUM_Y, NUM_X); gpuErr(cudaPeekAtLastError());
        ABCY_dif <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_aY, d_bY, d_cY, d_dtInv, d_myVarY, d_trMyDyy, NUM_T, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
#if DEBUGCUDA
        //Check d_aY
        gpuErr ( cudaMemcpy(temp, d_aY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_aY", cpu_arrs.aY, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "-> d_aY is correct" << endl;
        }

        //Check d_bY
        gpuErr ( cudaMemcpy(temp, d_bY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_bY", cpu_arrs.bY, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "-> d_bY is correct" << endl;
        }

        //Check d_cY
        gpuErr ( cudaMemcpy(temp, d_cY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_cY", cpu_arrs.cY, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "-> d_cY is correct" << endl;
        }
#endif

        //gpuErr ( cudaMemcpy(h_aX, d_aX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        //gpuErr ( cudaMemcpy(h_bX, d_bX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        //gpuErr ( cudaMemcpy(h_cX, d_cX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        //gpuErr ( cudaMemcpy(h_aY, d_aY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        //gpuErr ( cudaMemcpy(h_bY, d_bY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        //gpuErr ( cudaMemcpy(h_cY, d_cY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        //cudaDeviceSynchronize();

#if DEBUGCUDA
        unsigned lower = NUM_T - 4;
        unsigned higher = NUM_T - 2;
#endif
        for(int j = NUM_T-2; j >= 0; --j) {
#if FLOWCUDA
            cout << "   trMyResult... " << j << endl;
#endif
            for (int i = 0; i < OUTER_LOOP_COUNT; i++) {
                transpose_nosync<REAL, TILE> (d_myResult + i * NUM_X * NUM_Y, d_trMyResult + i * NUM_X * NUM_Y, NUM_X, NUM_Y);
            }
            cudaDeviceSynchronize();
#if DEBUGCUDA
            if (j <= higher && j >= lower) {
                gpuErr ( cudaMemcpy(temp, d_myResult, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
                if (cpu_arrs.NaNExists(temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y)) {
                    cout << "->-> d_myResult contains NaN, terminating at index: " << j << endl;
                    break;
                } else if (higher - lower < 4) {
                    cout << "->-> d_myResult still valid" << endl;
                }
                gpuErr ( cudaMemcpy(temp, d_trMyResult, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
                if (cpu_arrs.NaNExists(temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y)) {
                    cout << "->-> d_trMyResult contains NaN, terminating at index: " << j << endl;
                    break;
                } else if (higher - lower < 4) {
                    cout << "->-> d_trMyResult still valid" << endl;
                }
            }
            bool valid = true;
            gpuErr ( cudaMemcpy(temp, d_myResult, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
            REAL* tempT = (REAL*) malloc(NUM_T * NUM_X * NUM_Y * sizeof(REAL));
            gpuErr ( cudaMemcpy(tempT, d_trMyResult, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
            for (int i = 0; i < OUTER_LOOP_COUNT && valid; i++) {
                for (int k = 0; k < NUM_X && valid; k++) {
                    for (int h = 0; h < NUM_Y && valid; h++) {
                        if (temp[(i * NUM_X + k) * NUM_Y + h] != tempT[(i * NUM_Y + h) * NUM_X + k]) {
                            cout << "Error, did not transpose correctly at index [" << i << "][" << k << "][" << h << "] was " << tempT[(i * NUM_Y + h) * NUM_X + k] << " expected " << temp[(i * NUM_X + k) * NUM_Y + h] << endl;
                            valid = false;
                        }
                    }
                }
            }
            if (valid) {
                cout << "Valid transposition (" << j << ")" << endl;
            }
            
#endif

#if FLOWCUDA
            cout << "   U1... " << j << endl;
#endif
            U1 <<< ((BLOCK - 1 + OUTER_LOOP_COUNT * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_u, d_trMyVarX, d_trMyDxx, d_trMyResult, h_dtInv[j], j, OUTER_LOOP_COUNT, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
            cudaDeviceSynchronize();
#if DEBUGCUDA
            if (j <= higher && j >= lower) {
                gpuErr ( cudaMemcpy(temp, d_u, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
                if (cpu_arrs.NaNExists(temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y)) {
                    cout << "->-> d_u(U1) contains NaN, terminating at index: " << j << endl;
                    break;
                } else if (higher - lower < 4) {
                    cout << "->-> d_u(U1) still valid" << endl;
                }
            }
#endif
            
#if FLOWCUDA
            cout << "   V... " << j << endl;
#endif
            V <<< ((BLOCK - 1 + OUTER_LOOP_COUNT * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_v, d_myVarY, d_trMyDyy, d_myResult, j, OUTER_LOOP_COUNT, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
            cudaDeviceSynchronize();
#if DEBUGCUDA
            //if (j == NUM_T - 2) {
            //    gpuErr ( cudaMemcpy(temp, d_v, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
            //    if (cpu_arrs.compareArrays("d_v", cpu_arrs.v, temp, NUM_X * NUM_Y, true, false)) {
            //        cout << "->-> v is correct on first iteration" << endl;
            //    }
            //}
            if (j <= higher && j >= lower) {
                gpuErr ( cudaMemcpy(temp, d_v, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
                if (cpu_arrs.NaNExists(temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y)) {
                    cout << "->-> d_v contains NaN, terminating at index: " << j << endl;
                    break;
                } else if (higher - lower < 4) {
                    cout << "->-> d_v still valid" << endl;
                }
            }
#endif

#if FLOWCUDA
            cout << "   trV... " << j << endl;
#endif
            for (int i = 0; i < OUTER_LOOP_COUNT; i++) {
                transpose_nosync<REAL, TILE> (d_v + i * NUM_X * NUM_Y, d_trV + i * NUM_X * NUM_Y, NUM_Y, NUM_X);
            }
            cudaDeviceSynchronize();
#if DEBUGCUDA
            if (j <= higher && j >= lower) {
                gpuErr ( cudaMemcpy(temp, d_trV, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
                if (cpu_arrs.NaNExists(temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y)) {
                    cout << "->-> d_trV contains NaN, terminating at index: " << j << endl;
                    break;
                } else if (higher - lower < 4) {
                    cout << "->-> d_trV still valid" << endl;
                }
            }
#endif

#if FLOWCUDA
            cout << "   U2... " << j << endl;
#endif
            U2 <<< ((BLOCK - 1 + OUTER_LOOP_COUNT * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_u, d_trV, OUTER_LOOP_COUNT, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
            cudaDeviceSynchronize();
#if DEBUGCUDA
            //if (j == NUM_T - 2) {
            //    gpuErr ( cudaMemcpy(temp, d_u, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
            //    if (cpu_arrs.compareArrays("d_u", cpu_arrs.u, temp, NUM_X * NUM_Y, true, false)) {
            //        cout << "->-> u is correct on first iteration" << endl;
            //    }
            //}
            if (j <= higher && j >= lower) {
                gpuErr ( cudaMemcpy(temp, d_u, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
                if (cpu_arrs.NaNExists(temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y)) {
                    cout << "->-> d_u(U2) contains NaN, terminating at index: " << j << endl;
                    break;
                } else if (higher - lower < 4) {
                    cout << "->-> d_u(U2) still valid" << endl;
                }
            }
#endif

#if FLOWCUDA
            cout << "   implicit x... " << j << endl;
#endif
            //gpuErr ( cudaMemcpy(h_u, d_u, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
            for( unsigned i = 0; i < OUTER_LOOP_COUNT; ++ i ) {
                for(unsigned k = 0; k < NUM_Y; k++) {
                    //Launch sequential kernel (1 block, utilizing 1 warp thread)
                    SeqTridagPar1 <<< 1, 32, NUM_X * sizeof(MyReal4) + NUM_X * sizeof(MyReal2) >>> (d_aX, d_bX, d_cX, d_u, i, j, k, NUM_X, NUM_Y, d_yy);
                    cudaDeviceSynchronize();
                    //tridagPar(&h_aX[(j * NUM_Y + k) * NUM_X],
                    //          &h_bX[(j * NUM_Y + k) * NUM_X],
                    //          &h_cX[(j * NUM_Y + k) * NUM_X],
                    //          &h_u[(i * NUM_Y + k) * NUM_X],
                    //          NUM_X,
                    //          &h_u[(i * NUM_Y + k) * NUM_X],
                    //          h_yy
                    //);
                }
            }
            //gpuErr ( cudaMemcpy(d_u, h_u, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyHostToDevice) ); gpuErr(cudaPeekAtLastError());

#if FLOWCUDA
            cout << "   trU... " << j << endl;
#endif
            for (int i = 0; i < OUTER_LOOP_COUNT; i++) {
                transpose_nosync<REAL, TILE> (d_u + i * NUM_X * NUM_Y, d_trU + i * NUM_X * NUM_Y, NUM_X, NUM_Y);
            }
            cudaDeviceSynchronize();
#if DEBUGCUDA
            if (j <= higher && j >= lower) {
                gpuErr ( cudaMemcpy(temp, d_trU, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
                if (cpu_arrs.NaNExists(temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y)) {
                    cout << "->-> d_trU contains NaN, terminating at index: " << j << endl;
                    break;
                } else if (higher - lower < 4) {
                    cout << "->-> d_trU still valid" << endl;
                }
            }
#endif

#if FLOWCUDA
            cout << "   Y... " << j << endl;
#endif
            Y <<< ((BLOCK - 1 + OUTER_LOOP_COUNT * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_y, d_trU, d_v, h_dtInv[j], OUTER_LOOP_COUNT, NUM_X, NUM_Y);
            cudaDeviceSynchronize();

#if FLOWCUDA
            cout << "   implicit y... " << j << endl;
#endif
            //gpuErr ( cudaMemcpy(h_y, d_y, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
            //cudaDeviceSynchronize();
#if FLOWCUDA
            cout << "   implicit y load 1... " << j << endl;
#endif
            //gpuErr ( cudaMemcpy(h_myResult, d_myResult, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
            //cudaDeviceSynchronize();
#if FLOWCUDA
            cout << "   implicit y load 2... " << j << endl;
#endif
            for( unsigned i = 0; i < OUTER_LOOP_COUNT; ++ i ) {
                for(unsigned k = 0; k < NUM_X; k++) {
                    SeqTridagPar2 <<< 1, 32, NUM_Y * sizeof(MyReal4) + NUM_Y * sizeof(MyReal2) >>> (d_aY, d_bY, d_cY, d_y, i, j, k, NUM_Y, NUM_X, d_myResult, d_yy);
                    cudaDeviceSynchronize();
                    //tridagPar(&h_aY[(j * NUM_X + k) * NUM_Y],
                    //          &h_bY[(j * NUM_X + k) * NUM_Y],
                    //          &h_cY[(j * NUM_X + k) * NUM_Y],
                    //          &h_y[(i * NUM_X + k) * NUM_Y],
                    //          NUM_Y,
                    //          &h_myResult[(i * NUM_X + k) * NUM_Y],
                    //          h_yy
                    //);
                }
            }
#if FLOWCUDA
            cout << "   implicit y store 1... " << j << endl;
#endif
            //gpuErr ( cudaMemcpy(d_myResult, h_myResult, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyHostToDevice) ); gpuErr(cudaPeekAtLastError());
            //cudaDeviceSynchronize();
#if FLOWCUDA
            cout << "   iteration " << j << " done... " << endl;
#endif
#if DEBUGCUDA            
            gpuErr ( cudaMemcpy(temp, d_myResult, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
            cout << "->-> myResultIndex: " << temp[(0 * NUM_X + myXindex) * NUM_Y + myYindex] << endl; 
#endif
        }
        cudaDeviceSynchronize();

#if FLOWCUDA
        cout << "   res... " << endl;
#endif
        Res <<< ((BLOCK - 1 + OUTER_LOOP_COUNT) / BLOCK), BLOCK >>>(d_res_gpu, d_myResult, myXindex, myYindex, OUTER_LOOP_COUNT, NUM_X, NUM_Y);
        cudaDeviceSynchronize();

#if FLOWCUDA
        cout << "   wrote back to res... " << endl;
#endif
        gpuErr ( cudaMemcpy(h_res_gpu, d_res_gpu, OUTER_LOOP_COUNT * sizeof(REAL), cudaMemcpyDeviceToHost) );

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;

        // Apparently crashes when called twice, therefore compare to other
        bool is_valid = compare_validate ( h_res_gpu, res_cpu1, OUTER_LOOP_COUNT );
        writeStatsAndResult( is_valid, h_res_gpu, OUTER_LOOP_COUNT, 
                             NUM_X, NUM_Y, NUM_T, is_valid, BLOCK, elapsed );     

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//       TEST SECTION
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        cout << "/////////////////POST KERNEL RESULTS IN COMPARISON TO CPU.../////////////////" << endl;
        //cpu_arrs.printArray("myX", 3, cpu_arrs.myX, NUM_X);

        //Check d_myX
        gpuErr ( cudaMemcpy(temp, d_myX, NUM_X * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myX", cpu_arrs.myX, temp, NUM_X, true, false)) {
            cout << "d_myX is correct" << endl;
        }

        //Check d_myY
        gpuErr ( cudaMemcpy(temp, d_myY, NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myY", cpu_arrs.myY, temp, NUM_Y, true, false)) {
            cout << "d_myY is correct" << endl;
        }

        //Check d_myTimeline
        gpuErr ( cudaMemcpy(temp, d_myTimeline, NUM_T * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myTimeline", cpu_arrs.myTimeline, temp, NUM_T, true, false)) {
            cout << "d_myTimeline is correct" << endl;
        }

        //Check d_myDxx
        gpuErr ( cudaMemcpy(temp, d_myDxx, NUM_X * 4 * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myDxx", cpu_arrs.myDxx, temp, NUM_X * 4, true, false)) {
            cout << "d_myDxx is correct" << endl;
        }

        //Check d_myDyy
        gpuErr ( cudaMemcpy(temp, d_myDyy, NUM_Y * 4 * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myDyy", cpu_arrs.myDyy, temp, NUM_Y * 4, true, false)) {
            cout << "d_myDyy is correct" << endl;
        }

        //Check d_trMyDxx
        gpuErr ( cudaMemcpy(temp, d_trMyDxx, NUM_X * 4 * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_trMyDxx", cpu_arrs.trMyDxx, temp, NUM_X * 4, true, false)) {
            cout << "d_trMyDxx is correct" << endl;
        }

        //Check d_trMyDyy
        gpuErr ( cudaMemcpy(temp, d_trMyDyy, NUM_Y * 4 * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_trMyDyy", cpu_arrs.trMyDyy, temp, NUM_Y * 4, true, false)) {
            cout << "d_trMyDyy is correct" << endl;
        }

        //Check d_myVarX
        gpuErr ( cudaMemcpy(temp, d_myVarX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myVarX", cpu_arrs.myVarX, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_myVarX is correct" << endl;
        }

        //Check d_myVarY
        gpuErr ( cudaMemcpy(temp, d_myVarY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myVarY", cpu_arrs.myVarY, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_myVarY is correct" << endl;
        }

        //Check d_trMyVarX
        gpuErr ( cudaMemcpy(temp, d_trMyVarX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_trMyVarX", cpu_arrs.trMyVarX, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_trMyVarX is correct" << endl;
        }

        //Check dtInv
        gpuErr ( cudaMemcpy(temp, d_dtInv, NUM_T * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_dtInv", cpu_arrs.dtInv, temp, NUM_T, true, false)) {
            cout << "d_dtInv is correct" << endl;
        }

        
        //Check d_aX
        gpuErr ( cudaMemcpy(temp, d_aX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_aX", cpu_arrs.aX, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_aX is correct" << endl;
        }
        
        //Check d_bX
        gpuErr ( cudaMemcpy(temp, d_bX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_bX", cpu_arrs.bX, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_bX is correct" << endl;
        }

        //Check d_cX
        gpuErr ( cudaMemcpy(temp, d_aX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_cX", cpu_arrs.cX, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_cX is correct" << endl;
        }
        
        //Check d_aY
        gpuErr ( cudaMemcpy(temp, d_aY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_aY", cpu_arrs.aY, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_aY is correct" << endl;
        }

        //Check d_bY
        gpuErr ( cudaMemcpy(temp, d_bY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_bY", cpu_arrs.bY, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_bY is correct" << endl;
        }

        //Check d_cY
        gpuErr ( cudaMemcpy(temp, d_cY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_cY", cpu_arrs.cY, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_cY is correct" << endl;
        }

        //Check d_myResult
        gpuErr ( cudaMemcpy(temp, d_myResult, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myResult", cpu_arrs.myResult, temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y, true, false)) {
            cout << "d_myResult is correct" << endl;
        }

        //Check d_trMyResult
        gpuErr ( cudaMemcpy(temp, d_trMyResult, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_trMyResult", cpu_arrs.trMyResult, temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y, true, false)) {
            cout << "d_trMyResult is correct" << endl;
        }

        //Check d_u
        gpuErr ( cudaMemcpy(temp, d_u, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_u", cpu_arrs.u, temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y, true, false)) {
            cout << "d_u is correct" << endl;
        }

        //Check d_trU
        gpuErr ( cudaMemcpy(temp, d_trU, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_trU", cpu_arrs.trU, temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y, true, false)) {
            cout << "d_trU is correct" << endl;
        }

        //Check d_v
        gpuErr ( cudaMemcpy(temp, d_v, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_v", cpu_arrs.v, temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y, true, false)) {
            cout << "d_v is correct" << endl;
        }

        //Check d_trV
        gpuErr ( cudaMemcpy(temp, d_cY, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_trV", cpu_arrs.trV, temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y, true, false)) {
            cout << "d_trV is correct" << endl;
        }

        //Check d_y
        gpuErr ( cudaMemcpy(temp, d_y, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_y", cpu_arrs.y, temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y, true, false)) {
            cout << "d_y is correct" << endl;
        }

        /*
        //Check d_yy
        gpuErr ( cudaMemcpy(temp, d_cY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_cY", cpu_arrs.cY, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_cY is correct" << endl;
        }
        */

        //gpuErr ( cudaMemcpy(temp, d_myX, NUM_X * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        //cpu_arrs.printTwoArray("myX", cpu_arrs.myX, temp, 0, NUM_X - 1);

        MyResult_dif <<< ((BLOCK - 1 + OUTER_LOOP_COUNT * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_myResult, d_myX, OUTER_LOOP_COUNT, NUM_X, NUM_Y); gpuErr(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        for(unsigned block_off = 0; block_off < OUTER_LOOP_COUNT * NUM_X * NUM_Y; block_off += BLOCK) {
            for (unsigned tid = 0; tid < BLOCK; tid ++) {
                unsigned gidx = block_off + tid;
                unsigned row = gidx / (NUM_X * NUM_Y);
                unsigned row_remain = gidx % (NUM_X * NUM_Y);
                unsigned col = row_remain / NUM_Y;
                //Shared memory for myX
                if (gidx < OUTER_LOOP_COUNT * NUM_X * NUM_Y) {
                    REAL strike = 0.001*row;
                    REAL payoff = max(cpu_arrs.myX[col]-strike, (REAL)0.0);
                    cpu_arrs.myResult[gidx] = payoff;
                }
            }
        }
        matTransposePlane<REAL>(cpu_arrs.myResult, cpu_arrs.trMyResult, OUTER_LOOP_COUNT, NUM_X, NUM_Y);
        gpuErr ( cudaMemcpy(temp, d_myResult, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_myResult", cpu_arrs.myResult, temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y, true, false)) {
            cout << "myResult initialized correctly" << endl;
        }

        for (int i = 0; i < OUTER_LOOP_COUNT; i++) {
            transpose_nosync<REAL, TILE> (d_myResult + i * NUM_X * NUM_Y, d_trMyResult + i * NUM_X * NUM_Y, NUM_X, NUM_Y);
        }
        cudaDeviceSynchronize();

        
        gpuErr ( cudaMemcpy(temp, d_trMyResult, OUTER_LOOP_COUNT * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_trMyResult", cpu_arrs.trMyResult, temp, OUTER_LOOP_COUNT * NUM_X * NUM_Y, true, false)) {
            cout << "trMyResult initialized correctly" << endl;
        }


        //ABCX_dif <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_aX, d_bX, d_cX, d_dtInv, d_trMyVarX, d_trMyDxx, NUM_T, NUM_Y, NUM_X); gpuErr(cudaPeekAtLastError());
        //cudaDeviceSynchronize();

        //AX_dif <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_aX, d_dtInv, d_trMyVarX, d_trMyDxx, NUM_T, NUM_Y, NUM_X); gpuErr(cudaPeekAtLastError());
        //cudaDeviceSynchronize();

        gpuErr ( cudaMemcpy(temp, d_aX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_aX", cpu_arrs.aX, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_aX initialized correctly" << endl;
        }


        //BX_dif <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_bX, d_dtInv, d_trMyVarX, d_trMyDxx, NUM_T, NUM_Y, NUM_X); gpuErr(cudaPeekAtLastError());
        //cudaDeviceSynchronize();

        gpuErr ( cudaMemcpy(temp, d_bX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_bX", cpu_arrs.bX, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_bX initialized correctly" << endl;
        }


        //CX_dif <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_cX, d_dtInv, d_trMyVarX, d_trMyDxx, NUM_T, NUM_Y, NUM_X); gpuErr(cudaPeekAtLastError());
        //cudaDeviceSynchronize();

        gpuErr ( cudaMemcpy(temp, d_cX, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_cX", cpu_arrs.cX, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_cX initialized correctly" << endl;
        }

        //ABCY_dif <<< ((BLOCK - 1 + NUM_T * NUM_X * NUM_Y) / BLOCK), BLOCK >>> (d_aX, d_bX, d_cX, d_dtInv, d_myVarY, d_trMyDyy, NUM_T, NUM_Y, NUM_X); gpuErr(cudaPeekAtLastError());
        //cudaDeviceSynchronize();

        gpuErr ( cudaMemcpy(temp, d_aY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_aY", cpu_arrs.aY, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_aY initialized correctly" << endl;
        }

        gpuErr ( cudaMemcpy(temp, d_bY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_bY", cpu_arrs.bY, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_bY initialized correctly" << endl;
        }

        gpuErr ( cudaMemcpy(temp, d_cY, NUM_T * NUM_X * NUM_Y * sizeof(REAL), cudaMemcpyDeviceToHost) ); gpuErr(cudaPeekAtLastError());
        if (cpu_arrs.compareArrays("d_cY", cpu_arrs.cY, temp, NUM_T * NUM_X * NUM_Y, true, false)) {
            cout << "d_cY initialized correctly" << endl;
        }



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                             
        cudaFree(d_myX       );
        cudaFree(d_myY       );
        cudaFree(d_myTimeline);
        cudaFree(d_myDxx     );
        cudaFree(d_myDyy     );
        cudaFree(d_trMyDxx   );
        cudaFree(d_trMyDyy   );

        cudaFree(d_myVarX    );
        cudaFree(d_myVarY    );
        cudaFree(d_trMyVarX  );
        cudaFree(d_aX        );
        cudaFree(d_bX        );
        cudaFree(d_cX        );
        cudaFree(d_aY        );
        cudaFree(d_bY        );
        cudaFree(d_cY        );

        cudaFree(d_myResult  );
        cudaFree(d_trMyResult);

        cudaFree(d_u         );
        cudaFree(d_trU       );
        cudaFree(d_v         );
        cudaFree(d_trV       );
        cudaFree(d_y         );
        cudaFree(d_yy        );

        cudaFree(d_dtInv     );
        cudaFree(d_dl        );
        cudaFree(d_du        );

        cudaFree(d_res_gpu   );

        free(h_dtInv         );
        free(h_aX            );
        free(h_bX            );
        free(h_cX            );
        free(h_aY            );
        free(h_bY            );
        free(h_cY            );
        free(h_yy            );
        free(h_myResult      );
        free(h_u             );
    }
#endif
    //Free memory



    free(res_cpu1);
    free(res_cpu2);
    free(res_cpu3);
    free(res_cpu4);
    free(h_res_gpu);
    
    
    cout<<"\n// Done"<<endl;
    return 0;
}

