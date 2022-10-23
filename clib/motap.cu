#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

struct CSRMatrix {
    int nzmax;
    int m;
    int n; 
    int *p;
    int *i;
    int *x;
    int nz;
};

void value_iteration(struct CSRMatrix x[], size_t len) {
    // convert each of the CSR matrices into a sparse matrix
    // TODO need a pointer for the returns from value iteration
}


extern "C" {

void test_csr_to_cuda_spcsr(struct CSRMatrix x[], size_t len) {
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t descrA = NULL;

    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
}

}