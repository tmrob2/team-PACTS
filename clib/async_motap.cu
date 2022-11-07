#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

/*
First we need to check that the device can perform upstream and downstream copies
simultaneously.
*/

/*
#######################################################################
#                              CUDA                                   #
#######################################################################
*/

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

extern "C" {

// Do C interface stuff
int check_device_properties(void) {
    cudaDeviceProp GPUProp;

    CHECK_CUDA(cudaGetDeviceProperties(&GPUProp, 0));

    printf("Simultaneous transfer = %s\n", GPUProp.deviceOverlap ? "YES" : "NO");
    return 0;
}

// Test a function  which starts a stream and then turns a the
// three input arrays into a CSR array on the stream
int test_stream_csr(
    int *pi, 
    int *pj, 
    float* px, 
    int* pm, 
    int* pn, 
    int* pnz,
    int fsizem,
    int fsizen,
    int fnz
    ) {
    cudaEvent_t     start, stop;
    float           elaspsedTime;
    // Start the timers
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    // initialise the stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Allocate the memory to the GPU for the steam size
    int *dPCsrRowPtr, *dPCsrColPtr;
    float *dPCsrValPtr;

    // allocate device memory to store the sparse CSR 
    CHECK_CUDA(cudaMalloc((void **)&dPCsrValPtr, sizeof(float) * pnz[0]));
    CHECK_CUDA(cudaMalloc((void **)&dPCsrRowPtr, sizeof(int) * pm[0]));
    CHECK_CUDA(cudaMalloc((void **)&dPCsrColPtr, sizeof(int) * pnz[0]));

    // Allocate the page-locked memory used in the stream
    //CHECK_CUDA(cudaHostAlloc((void**)&host_a, fsizem * sizeof(int), cudaHostAllocDefault));
    //CHECK_CUDA(cudaHostAlloc((void**)&host_))
    //CHECK_CUDA(cudaHostAlloc((void**)&host_))

    return 0;
}

}