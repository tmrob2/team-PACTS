#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "cublas_v2.h"
#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

/*
#######################################################################
#                           KERNELS                                   #
#######################################################################
*/


extern "C" {

__global__ void abs_diff(float *a, float *b, float *c, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // HANDLE THE DATA AT THIS INDEX
    if (tid < m) {
        // compute the absolute diff between two elems
        float temp = fabsf(a[tid] - b[tid]);
        c[tid] = temp;
    } 
}


/*
#######################################################################
#                           C HELPER                                  #
#######################################################################
*/

struct CSparse {
    int nz;
    int m;
    int n;
    int *p;
    int *i;
    float *x;
};

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

/*
#######################################################################
#                              CUDA                                   #
#######################################################################
*/


void test_csr_spmv(
    int *csr_row, 
    int *csr_col, 
    float *csr_vals, 
    float *x,
    float *y,
    int nnz, 
    int sizeof_row, 
    int m, 
    int n
    ) {
    // sizeof_row is the size of csr_row
    // sizeof_col is the size of csr_col
    // m number of rows in the matrix
    // n number of cols in the matrix
    // nnz is the size of the csr_vals
    // create a sparse handle
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t descrC = NULL;
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;

    //allocate dCsrRowPtr, dCsrColPtr, dCsrValPtr
    int *dCsrRowPtr, *dCsrColPtr;
    float *dCsrValPtr;

    // allocate device memory to store the sparse CSR
    cudaMalloc((void **)&dCsrValPtr, sizeof(float) * nnz);
    cudaMalloc((void **)&dCsrColPtr, sizeof(int) * nnz);
    cudaMalloc((void **)&dCsrRowPtr, sizeof(int) * m);

    // Free the device memory allocated to the coo ptrs once they
    // the conversion from coo to csr has been completed
    cudaMemcpy(dCsrValPtr, csr_vals, sizeof(float) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dCsrColPtr, csr_col, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dCsrRowPtr, csr_row, sizeof(int) * m, cudaMemcpyHostToDevice);

    // create the sparse CSR matrix in device memory
    status = cusparseCreateCsr(
        &descrC, // MATRIX DESCRIPTION
        m, // NUMBER OF ROWS
        n, // NUMBER OF COLS
        nnz, // NUMBER OF NON ZERO VALUES
        dCsrRowPtr, // ROWS OFFSETS
        dCsrColPtr, // COL INDICES
        dCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    );

    float alpha = 1.0;
    float beta = 1.0;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecX, vecY;
    float *dX, *dY;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    cudaMalloc((void**)&dX, n * sizeof(float));
    cudaMalloc((void**)&dY, m * sizeof(float));

    // copy the vector from host memory to device memory
    cudaMemcpy(dX, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, y, m * sizeof(float), cudaMemcpyHostToDevice);

    // create a dense vector on device memory
    cusparseCreateDnVec(&vecX, n, dX, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, m, dY, CUDA_R_32F);

    cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descrC, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &alpha, descrC, vecX, &beta, vecY, CUDA_R_32F, 
        CUSPARSE_MV_ALG_DEFAULT, dBuffer);

    // Any algorithms get inserted here

    cudaMemcpy(y, dY, m *sizeof(float), cudaMemcpyDeviceToHost);

    //destroy the vector descriptors
    cusparseDestroySpMat(descrC);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    // Free the device memory
    cudaFree(dCsrColPtr);
    cudaFree(dCsrRowPtr);
    cudaFree(dCsrValPtr);
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dBuffer);

}

void test_csr_create(
    int m,
    int n,
    int nnz,
    int *i,
    int *j,
    int *x
) {
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);

    cusparseSpMatDescr_t descrP = NULL;
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;

    // allocated the device memory for the COO matrix
    int *dCOORowPtr, *dCOOColPtr;
    float *dCOOValPtr;

    // allocate device memory to store the sparse CSR
    cudaMalloc((void **)&dCOOValPtr, sizeof(float) * nnz);
    cudaMalloc((void **)&dCOORowPtr, sizeof(int) * nnz);
    cudaMalloc((void **)&dCOOColPtr, sizeof(int) * nnz);

    cudaMemcpy(dCOOValPtr, i, sizeof(float) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dCOOColPtr, j, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dCOORowPtr, x, sizeof(int) * (m + 1), cudaMemcpyHostToDevice);

    status = cusparseCreateCsr(
        &descrP,
        m,
        n,
        nnz,
        dCOORowPtr,
        dCOOColPtr,
        dCOOValPtr,
        CUSPARSE_INDEX_32I, // ROW OFFSET
        CUSPARSE_INDEX_32I, // COL IND
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    );

    // convert the trans output to device memory

    cusparseDestroySpMat(descrP);
    cusparseDestroy(handle);

    // Free the device memory
    
    cudaFree(dCOORowPtr);
    cudaFree(dCOOColPtr);
    cudaFree(dCOOValPtr);
}

int test_initial_policy_value(
    float *value,
    int pm,
    int pn,
    int pnz,
    int * pi,
    int * pj,
    float * px,
    int rm,
    int rn,
    int rnz,
    int *ri,
    int *rj,
    float *rx,
    float *x,
    float *y,
    float *w,
    float *rmv
    ) {
    /* 
    this test is to understand moving data onto CUDA so that
    a spmv can be performed with cublas, cusparse
    then a resulting sum ax + by

    Get the COO matrix into sparsescoo fmt

    Then multiply the COO by the initial value vector

    The rewards matrix is also sparse so it will need a sparse matrix descr
    as well. Multiply R by a repeated weight vector in the number 
    of prods and actions

    Finally sum the result

    This should happen in a loop until convergence

    I also want to do some wall timing to see some statistics on 
    the GPU 
    */
    //int *trans_output, *reward_output;
    //trans_output = (int *)malloc(block_size * max_state_space * sizeof(int));
    //reward_output = (int *)malloc(block_size * num_objectives * sizeof(int));

    // lets build the sparse transition matrix first

    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    cudaError_t cudaStat;
    cublasHandle_t blashandle;
    cublasCreate(&blashandle);


    cusparseSpMatDescr_t descrP = NULL;
    cusparseSpMatDescr_t descrR = NULL;
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    cublasStatus_t blas_status = CUBLAS_STATUS_SUCCESS;

    // allocated the device memory for the COO matrix

    // ----------------------------------------------------------------
    //                       Transition Matrix
    // ----------------------------------------------------------------

    //allocate dCsrRowPtr, dCsrColPtr, dCsrValPtr
    int *dPCsrRowPtr, *dPCsrColPtr;
    float *dPCsrValPtr;

    // allocate device memory to store the sparse CSR 
    cudaMalloc((void **)&dPCsrValPtr, sizeof(float) * pnz);
    cudaMalloc((void **)&dPCsrRowPtr, sizeof(int) * pm);
    cudaMalloc((void **)&dPCsrColPtr, sizeof(int) * pnz);

    cudaMemcpy(dPCsrValPtr, px, sizeof(float) * pnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dPCsrColPtr, pj, sizeof(int) * pnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dPCsrRowPtr, pi, sizeof(int) * pm, cudaMemcpyHostToDevice);
    
    // create the sparse CSR matrix in device memory
    status = cusparseCreateCsr(
        &descrP, // MATRIX DESCRIPTION
        pm, // NUMBER OF ROWS
        pn, // NUMBER OF COLS
        pnz, // NUMBER OF NON ZERO VALUES
        dPCsrRowPtr, // ROWS OFFSETS
        dPCsrColPtr, // COL INDICES
        dPCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    );
    
    // ----------------------------------------------------------------
    //                       Rewards Matrix
    // ----------------------------------------------------------------
    
    int *dRCsrRowPtr, *dRCsrColPtr;
    float *dRCsrValPtr;

    // allocate device memory to store the sparse CSR 
    cudaMalloc((void **)&dRCsrValPtr, sizeof(float) * rnz);
    cudaMalloc((void **)&dRCsrRowPtr, sizeof(int) * rm);
    cudaMalloc((void **)&dRCsrColPtr, sizeof(int) * rnz);
    printf("PRINTING COPIED REWARDS DATA\n");
    for (int k = 0; k < rnz; k ++) {
        printf("%f, ", rx[k]);
    }
    printf("\n");

    cudaMemcpy(dRCsrValPtr, rx, sizeof(float) * rnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dRCsrColPtr, rj, sizeof(int) * rnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dRCsrRowPtr, ri, sizeof(int) * rm, cudaMemcpyHostToDevice);

    // create the sparse CSR matrix in device memory
    printf("ROWS: %i, COLS: %i, NNZ: %i\n", rm, rn, rnz);
    status = cusparseCreateCsr(
        &descrR, // MATRIX DESCRIPTION
        rm, // NUMBER OF ROWS
        rn, // NUMBER OF COLS
        rnz, // NUMBER OF NON ZERO VALUES
        dRCsrRowPtr, // ROWS OFFSETS
        dRCsrColPtr, // COL INDICES
        dRCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    );

    // ----------------------------------------------------------------
    //                      Start of VI
    // ----------------------------------------------------------------

    // --------------TRANSITION MATRIX MULTIPLICATION SETUP------------
    
    float alpha = 1.0;
    float beta = 1.0;
    float *epsilon = (float*) malloc(pm * sizeof(float));
    int iepsilon;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecX, vecY;
    int *d_arg_epsilon;
    float *dX, *dY, *d_tmp, *dZ, *dStaticY, *dOutput;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    cudaMalloc((void**)&dX, pm * sizeof(float));
    cudaMalloc((void**)&dOutput, pm * sizeof(float));
    cudaMalloc((void**)&dY, pm * sizeof(float));
    cudaMalloc((void**)&dZ, pm * sizeof(float));
    cudaMalloc((void**)&dStaticY, pm * sizeof(float));
    cudaMalloc((void**)&d_tmp, pm * sizeof(float));
    cudaMalloc((void**)&d_arg_epsilon, sizeof(int));

    // create a initial Y vector
    float *static_y = (float*) calloc(pm, sizeof(float));
    
    // copy the vector from host memory to device memory
    cudaMemcpy(dX, x, pn * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(dStaticY, static_y, pm * sizeof(float), cudaMemcpyHostToDevice);

    // create a dense vector on device memory
    cusparseCreateDnVec(&vecX, pn, dX, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, pm, dY, CUDA_R_32F);

    cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descrP, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    
    // --------------REWARDS MATRIX MULTIPLICATION SETUP---------------

    float alphaR = 1.0;
    float betaR = 1.0;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecW, vecRMv;
    float *dRw, *dRMv, *dRstaticMx;
    void* dBufferR = NULL;
    size_t bufferSizeR = 0;

    //float *rmv = (float*) calloc(rm, sizeof(float));

    cudaMalloc((void**)&dRw, rn * sizeof(float));
    cudaMalloc((void**)&dRMv, rm * sizeof(float));
    cudaMalloc((void**)&dRstaticMx, rm * sizeof(float));

    // copy the vector from host memory to device memory
    cudaMemcpy(dRw, w, rn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dRMv, rmv, rm * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(dRstaticMx, rmv, rm * sizeof(float), cudaMemcpyHostToDevice);

    // create a dense vector on device memory
    cusparseCreateDnVec(&vecW, rn, dRw, CUDA_R_32F);
    cusparseCreateDnVec(&vecRMv, rm, dRMv, CUDA_R_32F);

    cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alphaR, descrR, vecW, &betaR, vecRMv, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSizeR);
    cudaMalloc(&dBufferR, bufferSizeR);

    // ALGORITHM LOOP

    // Copy the zero vector to initialise Y -> captures A.x result 
    // for transition matrix
    //csparseDnVecSetValues(vecY, dY);
    //cublasScopy(blashandle, pm, dYStatic, 1, dY, 1);
    // copy the static Y vector to initialise Y
    CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &alphaR, descrR, vecW, &betaR, vecRMv, CUDA_R_32F, 
        CUSPARSE_MV_ALG_DEFAULT, dBufferR));

    CHECK_CUDA(cudaMemcpy(y, dRMv, pm *sizeof(float), cudaMemcpyDeviceToHost));
    printf("PRINTING REWARDS VECTOR AFTER MxV\n");
    for (int k = 0; k < pm; k++) {
        printf("%f, ", y[k]);
    }
    printf("\n");

    for (int algo_i = 0; algo_i < 10; algo_i ++) {

        CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &alpha, descrP, vecX, &beta, vecY, CUDA_R_32F, 
        CUSPARSE_MV_ALG_DEFAULT, dBuffer));

        // push this into the algorithm loop


        // ---------------------SUM DENSE VECTORS-------------------------

        /* 
        The gpu memory shoulf already be allocated, i.e. we are summing
        dY + dRMv
        */
        cublasSaxpy(blashandle, pm, &alpha, dRMv, 1, dY, 1);
        
        // ---------------------COMPUTE EPSILON---------------------------

        // what is the difference between dY and dX

        // EPSILON COMPUTATION
        abs_diff<<<pm,1>>>(dX, dY, dZ, pm);
        CHECK_CUDA(cudaMemcpy(y, dY, pm *sizeof(float), cudaMemcpyDeviceToHost));
        /*
        for (int k = 0; k < pm; k++) {
            printf("%.1f, ", y[k]);
        }
        printf("\n");
        */
        CHECK_CUBLAS(cublasIsamax(blashandle, pm, dZ, 1, &iepsilon));
        CHECK_CUDA(cudaMemcpy(epsilon, dZ, pm *sizeof(float), cudaMemcpyDeviceToHost));
        //thrust::device_ptr<float> dev_ptr(dZ);
        //epsilon = dev_ptr[iepsilon];
        //epsilon = y[iepsilon];
        CHECK_CUBLAS(cublasScopy(blashandle, pm, dY, 1, dX, 1));
        // RESET Y
        CHECK_CUBLAS(cublasScopy(blashandle, pm, dStaticY, 1, dY, 1));
        // RESET RMV
        
        //CHECK_CUSPARSE(cusparseDnVecSetValues(vecX, dX));
        //CHECK_CUSPARSE(cusparseDnVecSetValues(vecY, dY));
    }
    
    
    //cudaMemcpy(rmv, dRMv, rm *sizeof(float), cudaMemcpyDeviceToHost);
    //destroy the vector descriptors
    cusparseDestroySpMat(descrP);
    cusparseDestroySpMat(descrR);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroyDnVec(vecRMv);
    cusparseDestroyDnVec(vecW);
    cusparseDestroy(handle);
    cublasDestroy(blashandle);

    // Free the device memory
    cudaFree(dPCsrColPtr);
    cudaFree(dPCsrRowPtr);
    cudaFree(dPCsrValPtr);
    cudaFree(dRCsrColPtr);
    cudaFree(dRCsrRowPtr);
    cudaFree(dRCsrValPtr);
    cudaFree(d_arg_epsilon);
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dStaticY);
    cudaFree(dZ);
    cudaFree(dRw);
    cudaFree(dRMv);
    cudaFree(dRstaticMx);
    cudaFree(dBuffer);
    cudaFree(dBufferR);
    free(epsilon);
    
}

}