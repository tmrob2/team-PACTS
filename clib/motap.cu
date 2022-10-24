#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct COOMatrix {
    int nzmax;
    int m;
    int n; 
    int *p;
    int *i;
    int *x;
    int nz;
};

/*void value_iteration(struct CSRMatrix x[], size_t len) {
    // convert each of the CSR matrices into a sparse matrix
    // TODO need a pointer for the returns from value iteration
}*/


/*extern "C" {

void test_csr_to_cuda_spcsr(struct CSRMatrix x[], size_t len) {
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t descrA = NULL;

    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
}

}*/

// we want the input to be our dense matrix
// and we also want the indices that we need to gather
extern "C" {

void gather_policy(int *pi, int *output, int nc, int nr, int prod_block_size) {
    int rows[prod_block_size];
    for (int i = 0; i < prod_block_size; i++) {
        rows[i] = i + prod_block_size * pi[i];
    }

    int idx = 0;
    for (int r = 0; r < prod_block_size; r++) {
        for (int c=0; c < nc; c ++) {
            output[idx] = c * nr + rows[r];
            idx++;            
        }
    }
}

void test_thrust_gather(int *output) {
    // mark even indices with a 1; odd indices with a 0
    int values[10] = {1, 0, 0, 0, 1, 0, 1, 0, 1, 0};
    thrust::device_vector<int> d_values(values, values + 10);
    thrust::host_vector<int> output_(10);
    // gather all even indices into the first half of the range
    // and odd indices to the last half of the range
    int map[10]   = {0, 2, 4, 6, 8}; //, 1, 3, 5, 7, 9};
    thrust::device_vector<int> d_map(map, map + 5);//, map + 10);
    thrust::device_vector<int> d_output(5);
    thrust::gather(d_map.begin(), d_map.end(),
                d_values.begin(),
                d_output.begin());

    thrust::copy(d_output.begin(), d_output.end(), output_.begin());
    output = thrust::raw_pointer_cast(output_.data());
    printf("[");

    for (int i = 0; i < 10; i++) {
        if (i < 9) {
            printf("%i,", output[i]);
        } else {
            printf("%i]\n", output[i]);
        }
    }
}

void test_initial_policy_value(
    int *policy,
    int max_state_space,
    int block_size, 
    int total_rows,
    float *value, 
    int *trans_output,
    int *reward_output, 
    int *trans_i,
    int *trans_j,
    float * trans_x,
    int nnz,
    float* rewards, 
    int num_objectives
    ) {
    /* 
    this test is to understand moving data onto CUDA so that
    a spmv can be performed with cublas, cusparse
    then a resulting sum ax + by

    First thing, convert the policy to the indices required
    for both Thrust, and cusparse gather. 

    Then get the gather indices onto the device
    Get the value vector onto the device

    Get the COO matrix into sparsescoo fmt and then convert this 
    format to CSR

    Then multiply the CSR by the initial value vector

    Multiply the dense rewards vector by the weight vector
    Finally sum the result

    This should happen in a loop until convergence

    I also want to do some wall timing to see some statistics on 
    the GPU 
    */
    //int *trans_output, *reward_output;
    //trans_output = (int *)malloc(block_size * max_state_space * sizeof(int));
    //reward_output = (int *)malloc(block_size * num_objectives * sizeof(int));

    gather_policy(policy, trans_output, max_state_space, total_rows, block_size);

    // this output is the sparse matrix output

    // we need another gather array for the rewards matrix
    gather_policy(policy, reward_output, num_objectives, total_rows, block_size);

    // lets build the sparse transition matrix first

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

    cudaMemcpy(dCOOValPtr, trans_i, sizeof(float) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dCOOColPtr, trans_j, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dCOORowPtr, trans_x, sizeof(int) * nnz, cudaMemcpyHostToDevice);

    status = cusparseCreateCoo(
        &descrP,
        total_rows,
        max_state_space,
        nnz,
        dCOORowPtr,
        dCOOColPtr,
        dCOOValPtr,
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
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

}