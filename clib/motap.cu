#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "cublas_v2.h"
#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

int MAX_ITERATIONS = 10;

/*
#######################################################################
#                           KERNELS                                   #
#######################################################################
*/

__global__ void max_value(
    float *y,
    float *x,
    float *xnew,
    float *eps,
    int *pi,
    int *pinew,
    int prod_block, 
    int nact
    ) {
    // The purpose of this kernel is to do effective row-wise comparison of values
    // to determine the new policy and the new value vector without copy of 
    // data from the GPU to CPU
    //
    // It is recognised that this code will be slow due to memory segmentation
    // and cache access, but this should in theory be faster then sending data
    // back and forth between the GPU and CPU
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < prod_block) {
        eps[tid] = y[tid] - x[tid];
        pinew[tid] = 0;
        if (y[tid] > x[tid]) {
            xnew[tid] = y[tid];
        } else {
            xnew[tid] = x[tid];
        }
        for (int k = 1; k < nact; k++) {
            if (eps[tid] < y[k * prod_block + tid] - x[tid]) {
                pinew[tid] = k;
                xnew[tid] = y[k * prod_block + tid];
            }
        }

        if (eps[tid] > 0.001) {
            pi[tid] = pinew[tid];
        }

        for (int k = 1; k < nact; k++) {
            xnew[k * prod_block + tid] = xnew[tid];
        }
    }
}

__global__ void abs_diff(float *a, float *b, float *c, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // HANDLE THE DATA AT THIS INDEX
    if (tid < m) {
        // compute the absolute diff between two elems
        float temp = b[tid] - a[tid];
        c[tid] = temp;
    } 
}

extern "C" {

void max_value_launcher(float *y, float *x, float* xnew, float * eps, 
    int* pi, int *pinew, int prod_block, int nact) {
    int blockSize;    // The launch configurator returned block size
    int minGridSize;  // The maximum grid size needed to achieve max
                      // maximum occupancy
    int gridSize;     // The grid size needed, based on the input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, max_value, 0, 0);

    // Round up according to array size
    gridSize = (prod_block + blockSize - 1) / blockSize;

    max_value<<<gridSize, blockSize>>>(y, x, xnew, eps, pi, pinew, prod_block, nact);
}

void abs_diff_launcher(float *a, float *b, float* c, int m) {
    int blockSize;    // The launch configurator returned block size
    int minGridSize;  // The maximum grid size needed to achieve max
                      // maximum occupancy
    int gridSize;     // The grid size needed, based on the input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, abs_diff, 0, 0);

    // Round up according to array size
    gridSize = (m + blockSize - 1) / blockSize;

    abs_diff<<<gridSize, blockSize>>>(a, b, c, m);
}

/*
#######################################################################
#                           C HELPER                                  #
#######################################################################
*/

/// @brief Function which takes a total value vector for all actions and
///        compares the value vector at particular indices corresponding
///        to rows. 
/// @param a value vector
/// @param b output policy
/// @param prod_block the size of the col major stride (rows in matrix)
/// @param num_actions the number of actions in an environment
void create_policy(float *a, int*b, int prod_block, int num_actions) {
    for (int k = 0; k < prod_block; k ++) {
        float tmp_max = 0;
        int tmp_action = -1;
        int current_action = b[k];
        int value_of_current_action = a[b[k] * prod_block + k];
        for (int i = 0; i < num_actions; i++) {
            if (a[i * prod_block + k] > tmp_max) {
                tmp_max = a[i * prod_block + k];
                tmp_action = i;
            }
        }
        if (tmp_max > value_of_current_action && tmp_action >= 0) {
            b[k] = tmp_action;
        }
    }
}
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

int initial_policy_value(
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
    float *rmv,
    float eps
    ) {
    /* 
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
    // build the sparse transition matrix first

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
    cudaMemcpy(dRCsrValPtr, rx, sizeof(float) * rnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dRCsrColPtr, rj, sizeof(int) * rnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dRCsrRowPtr, ri, sizeof(int) * rm, cudaMemcpyHostToDevice);

    // create the sparse CSR matrix in device memory
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
    //float d_eps;
    //float h_eps; 
    float *dX, *dY, *d_tmp, *dZ, *dStaticY, *dOutput;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    cudaMalloc((void**)&dX, pm * sizeof(float));
    cudaMalloc((void**)&dOutput, pm * sizeof(float));
    cudaMalloc((void**)&dY, pm * sizeof(float));
    cudaMalloc((void**)&dZ, pm * sizeof(float));
    cudaMalloc((void**)&dStaticY, pm * sizeof(float));
    cudaMalloc((void**)&d_tmp, pm * sizeof(float));
    //cudaMalloc((void**)&d_eps, sizeof(float));

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

    float maxeps;
    maxeps = 0.0f;

    for (int algo_i = 0; algo_i < MAX_ITERATIONS; algo_i ++) {

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
        abs_diff_launcher(dX, dY, dZ, pm);
        //CHECK_CUBLAS(cublasIsamax(blashandle, pm, dZ, 1, &iepsilon));

        thrust::device_ptr<float> dev_ptr(dZ);
        maxeps = *thrust::max_element(thrust::device, dev_ptr, dev_ptr + pm);
        
        CHECK_CUBLAS(cublasScopy(blashandle, pm, dY, 1, dX, 1));
        // RESET Y
        CHECK_CUBLAS(cublasScopy(blashandle, pm, dStaticY, 1, dY, 1));
        if (maxeps < eps) {
            printf("EPS TOL REACHED in %i ITERATIONS\n", algo_i);
            break;
        }
    }
    
    CHECK_CUDA(cudaMemcpy(y, dX, pm *sizeof(float), cudaMemcpyDeviceToHost));
    
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
    //cudaFree(d_eps);
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

int policy_optimisation(
    float *init_value,
    int *Pi, // SIZE OF THE INIT POLICY WILL BE P.M
    int pm,    // TRANSITION COL NUMBER
    int pn,    // TRANSITION ROW NUMBER 
    int pnz,   // TRANSITION NON ZERO VALUES
    int *pi,   // TRANSITION ROW PTR CSR
    int *pj,   // TRANSITION COL VECTOR CSR
    float *px, // TRANSITION VALUE VECTOR
    int rm,    // REWARDS VALUE ROW NUMBER
    int rn,    // REWARDS VALUE COLS NUMBER
    int rnz,   // REWARDS NON ZERO VALUES
    int *ri,   // REWARDS MATRIX ROW PTR CSR
    int *rj,   // REWARDS MATRIX COL VECTOR CSR
    float *rx, // REWARDS MATRIX VALUE VECTOR
    float *x,  // ACC VALUE VECTOR
    float *y,  // TMP ACC VALUE VECTOR
    float *rmv, // initial R vec
    float *w,  // REPEATED WEIGHT VECTOR
    float eps,  // THRESHOLD
    int block_size,
    int nact
){
    /*
    This function is the second part of the value iteration implementation
    */
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    cudaError_t cudaStat;
    cublasHandle_t blashandle;
    cublasCreate(&blashandle);


    cusparseSpMatDescr_t descrP = NULL;
    cusparseSpMatDescr_t descrR = NULL;
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    cublasStatus_t blas_status = CUBLAS_STATUS_SUCCESS;

    // ----------------------------------------------------------------
    //                             POLICY
    // ----------------------------------------------------------------

    int *PI, *PI_new;
    CHECK_CUDA(cudaMalloc((void**)&PI, block_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&PI_new, block_size * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(PI, Pi, block_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(PI_new, Pi, block_size * sizeof(int), cudaMemcpyHostToDevice));

    printf("\n");
    for (int k = 0; k<block_size; k++) {
        if (k == 0) {
            printf("Pi: %i, ", Pi[k]);
        } else {
            printf("%i, ", Pi[k]);
        }
    }
    printf("\n");

    // allocated the device memory for the COO matrix

    // ----------------------------------------------------------------
    //                       Transition Matrix
    // ----------------------------------------------------------------

    //allocate dCsrRowPtr, dCsrColPtr, dCsrValPtr
    int *dPCsrRowPtr, *dPCsrColPtr;
    float *dPCsrValPtr;

    // allocate device memory to store the sparse CSR 
    CHECK_CUDA(cudaMalloc((void **)&dPCsrValPtr, sizeof(float) * pnz));
    CHECK_CUDA(cudaMalloc((void **)&dPCsrRowPtr, sizeof(int) * (pm + 1)));
    CHECK_CUDA(cudaMalloc((void **)&dPCsrColPtr, sizeof(int) * pnz));

    CHECK_CUDA(cudaMemcpy(dPCsrValPtr, px, sizeof(float) * pnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dPCsrColPtr, pj, sizeof(int) * pnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dPCsrRowPtr, pi, sizeof(int) * (pm + 1), cudaMemcpyHostToDevice));
    
    // create the sparse CSR matrix in device memory
    CHECK_CUSPARSE(cusparseCreateCsr(
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
    ));

    // ----------------------------------------------------------------
    //                       Rewards Matrix
    // ----------------------------------------------------------------
    
    int *dRCsrRowPtr, *dRCsrColPtr;
    float *dRCsrValPtr;

    // allocate device memory to store the sparse CSR 
    CHECK_CUDA(cudaMalloc((void **)&dRCsrValPtr, sizeof(float) * rnz));
    CHECK_CUDA(cudaMalloc((void **)&dRCsrRowPtr, sizeof(int) * (rm + 1)));
    CHECK_CUDA(cudaMalloc((void **)&dRCsrColPtr, sizeof(int) * rnz));
    CHECK_CUDA(cudaMemcpy(dRCsrValPtr, rx, sizeof(float) * rnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRCsrColPtr, rj, sizeof(int) * rnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRCsrRowPtr, ri, sizeof(int) * (rm + 1), cudaMemcpyHostToDevice)); 

    // create the sparse CSR matrix in device memory
    CHECK_CUSPARSE(cusparseCreateCsr(
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
    ));

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
    //
    float *dX, *dY, *d_tmp, *dZ, *dStaticY; 
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    CHECK_CUDA(cudaMalloc((void**)&dX, pm * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dY, pm * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dZ, pm * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dStaticY, pm * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_tmp, pm * sizeof(float)));
    //cudaMalloc((void**)&d_eps, sizeof(float));

    // create a initial Y vector
    float *static_y = (float*) calloc(pm, sizeof(float));
    
    // copy the vector from host memory to device memory
    CHECK_CUDA(cudaMemcpy(dX, init_value, pm * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dY, y, pm * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dStaticY, y, pm * sizeof(float), cudaMemcpyHostToDevice));

    // create a dense vector on device memory
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, pn, dX, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, pm, dY, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descrP, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    
    // --------------REWARDS MATRIX MULTIPLICATION SETUP---------------

    float alphaR = 1.0;
    float betaR = 1.0;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecW, vecRMv;
    float *dRw, *dRMv, *dRstaticMx;
    void* dBufferR = NULL;
    size_t bufferSizeR = 0;

    //float *rmv = (float*) calloc(rm, sizeof(float));

    CHECK_CUDA(cudaMalloc((void**)&dRw, rn * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dRMv, rm * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dRstaticMx, rm * sizeof(float)));

    // copy the vector from host memory to device memory
    CHECK_CUDA(cudaMemcpy(dRw, w, rn * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRMv, rmv, rm  * sizeof(float), cudaMemcpyHostToDevice));
    //cudaMemcpy(dRstaticMx, rmv, rm * sizeof(float), cudaMemcpyHostToDevice);

    // create a dense vector on device memory
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecW, rn, dRw, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecRMv, rm, dRMv, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alphaR, descrR, vecW, &betaR, vecRMv, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSizeR));
    CHECK_CUDA(cudaMalloc(&dBufferR, bufferSizeR));

    // ONE OFF REWARDS COMPUTATION

    CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &alphaR, descrR, vecW, &betaR, vecRMv, CUDA_R_32F, 
        CUSPARSE_MV_ALG_DEFAULT, dBufferR));
    
    // ALGORITHM LOOP - POLICY GENERATION
    float maxeps;
    maxeps = 0.0f;

    for (int algo_i = 0; algo_i < MAX_ITERATIONS; algo_i ++) {

        CHECK_CUDA(cudaMemcpy(y, dY, pm * sizeof(float), cudaMemcpyDeviceToHost));
       
        /*
        printf("BEFORE M.v\n");
        for (int k = 0; k<pm; k++) {
            if (k == 0) {
                printf("Y: %.2f, ", y[k]);
            } else {
                printf("%.2f, ", y[k]);
            }
        }
        printf("\n");
        */

        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
            &alpha, descrP, vecX, &beta, vecY, CUDA_R_32F, 
            CUSPARSE_MV_ALG_DEFAULT, dBuffer));

        // ---------------------SUM DENSE VECTORS-------------------------

        /* 
        i.e. we are summing dY + dRMv
        */
        
        CHECK_CUBLAS(cublasSaxpy(blashandle, pm, &alpha, dRMv, 1, dY, 1));
        // ---------------------COMPUTE EPSILON---------------------------

        // what is the difference between dY and dX

        // EPSILON COMPUTATION
        
        max_value_launcher(dY, dX, d_tmp, dZ, PI, PI_new, block_size, nact);
        //
        CHECK_CUBLAS(cublasIsamax(blashandle, pm, dZ, 1, &iepsilon));
        thrust::device_ptr<float> dev_ptr(dZ);
        maxeps = *thrust::max_element(thrust::device, dev_ptr, dev_ptr + pm);
        //
        // compute the max value 
        // reset y to zero

        CHECK_CUDA(cudaMemcpy(y, dZ, pm * sizeof(float), cudaMemcpyDeviceToHost));
        
        CHECK_CUBLAS(cublasScopy(blashandle, pm, dStaticY, 1, dY, 1));
        CHECK_CUBLAS(cublasScopy(blashandle, pm, d_tmp, 1, dX, 1));
        // std::cout << "EPS_TEST " << dev_ptr[iepsilon - 1] << "THRUST "<< maxeps << std::endl;
        if (maxeps < eps) {
            printf("EPS TOL REACHED in %i ITERATIONS\n", algo_i);
            break;
        }
    }

    CHECK_CUDA(cudaMemcpy(y, dX, pm * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(Pi, PI, block_size * sizeof(int), cudaMemcpyDeviceToHost));
    

    // MEMORY MANAGEMENT
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
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(d_tmp);
    cudaFree(dStaticY);
    cudaFree(dZ);
    cudaFree(dRw);
    cudaFree(dRMv);
    cudaFree(dRstaticMx);
    cudaFree(dBuffer);
    cudaFree(dBufferR);
    cudaFree(PI);
    cudaFree(PI_new);
    free(epsilon);
}

int multi_objective_values(
    float * R, // A block_size * nobjs vector of argmax rewards under pi
    int *pi, // argmax multi objective transition matrix csr
    int *pj, // ..
    float *px, // ..
    int pm, // row size
    int pn, // col size
    int pnz, // number of non zero elements in P
    float eps,
    float *x // output
) {
    /*
    This function is the final part of the value iteration algorithm
    */
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
    CHECK_CUDA(cudaMalloc((void **)&dPCsrValPtr, sizeof(float) * pnz));
    CHECK_CUDA(cudaMalloc((void **)&dPCsrRowPtr, sizeof(int) * (pm + 1)));
    CHECK_CUDA(cudaMalloc((void **)&dPCsrColPtr, sizeof(int) * pnz));

    CHECK_CUDA(cudaMemcpy(dPCsrValPtr, px, sizeof(float) * pnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dPCsrColPtr, pj, sizeof(int) * pnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dPCsrRowPtr, pi, sizeof(int) * (pm + 1), cudaMemcpyHostToDevice));
    
    // create the sparse CSR matrix in device memory
    CHECK_CUSPARSE(cusparseCreateCsr(
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
    ));

    // ----------------------------------------------------------------
    //                       Rewards Vector
    // ----------------------------------------------------------------
    float *dRValPtr;
    CHECK_CUDA(cudaMalloc((void **)&dRValPtr, sizeof(float) * pm));
    CHECK_CUDA(cudaMemcpy(dRValPtr, R, pm * sizeof(float), cudaMemcpyHostToDevice));

    // --------------TRANSITION MATRIX MULTIPLICATION SETUP------------
    float alpha = 1.0;
    float beta = 1.0;
    float *epsilon = (float*) malloc(pm * sizeof(float));
    int iepsilon;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecX, vecY;
    //
    float *dX, *dY, *d_tmp, *dZ, *dStaticY; 
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    CHECK_CUDA(cudaMalloc((void**)&dX, pm * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dY, pm * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dZ, pm * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dStaticY, pm * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_tmp, pm * sizeof(float)));
    //cudaMalloc((void**)&d_eps, sizeof(float));

    // create a initial Y vector
    float *static_y = (float*) calloc(pm, sizeof(float));
    
    // copy the vector from host memory to device memory
    CHECK_CUDA(cudaMemcpy(dX, x, pn * sizeof(float), cudaMemcpyHostToDevice));
    //cudaMemcpy(dStaticY, static_y, pm * sizeof(float), cudaMemcpyHostToDevice);

    // create a dense vector on device memory
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, pn, dX, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, pm, dY, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descrP, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    float maxeps;
    maxeps = 0.0f;

    for (int algo_i = 0; algo_i < 1; algo_i ++) {

        CHECK_CUDA(cudaMemcpy(x, dY, pm *sizeof(float), cudaMemcpyDeviceToHost));

        printf("BEFORE M.v\n");
        for (int k = 0; k<pm; k++) {
            if (k == 0) {
                printf("Y: %.2f, ", x[k]);
            } else {
                printf("%.2f, ", x[k]);
            }
        }
        printf("\n");

        CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &alpha, descrP, vecX, &beta, vecY, CUDA_R_32F, 
        CUSPARSE_MV_ALG_DEFAULT, dBuffer));

        // ---------------------SUM DENSE VECTORS-------------------------

        CHECK_CUBLAS(cublasSaxpy(blashandle, pm, &alpha, dRValPtr, 1, dY, 1));

        CHECK_CUDA(cudaMemcpy(x, dY, pm *sizeof(float), cudaMemcpyDeviceToHost));

        printf("AFTER R + P.v\n");
        for (int k = 0; k<pm; k++) {
            if (k == 0) {
                printf("Y: %.2f, ", x[k]);
            } else {
                printf("%.2f, ", x[k]);
            }
        }
        printf("\n");

        abs_diff_launcher(dX, dY, dZ, pm);
        //CHECK_CUBLAS(cublasIsamax(blashandle, pm, dZ, 1, &iepsilon));

        thrust::device_ptr<float> dev_ptr(dZ);
        maxeps = *thrust::max_element(thrust::device, dev_ptr, dev_ptr + pm);

        CHECK_CUBLAS(cublasScopy(blashandle, pm, dY, 1, dX, 1));
        // RESET Y
        CHECK_CUBLAS(cublasScopy(blashandle, pm, dStaticY, 1, dY, 1));
        //printf("\nepsilon: %f\n", dev_ptr[iepsilon - 1]);
        if (maxeps < eps) {
            printf("EPS TOL REACHED in %i ITERATIONS\n", algo_i);
            break;
        }
    }
    
    CHECK_CUDA(cudaMemcpy(x, dX, pm *sizeof(float), cudaMemcpyDeviceToHost));

    //destroy the vector descriptors
    cusparseDestroySpMat(descrP);
    cusparseDestroySpMat(descrR);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    cublasDestroy(blashandle);

    // Free the device memory
    cudaFree(dPCsrColPtr);
    cudaFree(dPCsrRowPtr);
    cudaFree(dPCsrValPtr);
    //cudaFree(d_eps);
    cudaFree(dRValPtr);
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dStaticY);
    cudaFree(dZ);
    cudaFree(dBuffer);
    free(epsilon);

}


}