#include "linalg/linalg.hpp"
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cstdio>

// We will only work on quantum mechanical systems == hermitian matrices
void diagonalize(double* H_dev, double* eigenvalues_dev, int N) {

    // --- 1. Create a cuSOLVER handle (like opening a connection to the library)
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // --- 2. Tell cuSOLVER what we want:
    //    CUSOLVER_EIG_MODE_VECTOR = compute eigenvalues AND eigenvectors
    //    CUBLAS_FILL_MODE_LOWER   = we filled the lower triangle of H
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // --- 3. Query how much workspace cuSOLVER needs
    int workspace_size = 0;
    cusolverDnDsyevd_bufferSize(handle, jobz, uplo, N,
                                 H_dev, N, eigenvalues_dev,
                                 &workspace_size);

    // --- 4. Allocate workspace and info flag on GPU
    double* workspace_dev = nullptr;
    int* info_dev = nullptr;
    cudaMalloc(&workspace_dev, workspace_size * sizeof(double));
    cudaMalloc(&info_dev, sizeof(int));

    // --- 5. Run the diagonalization
    //    After this call:
    //      H_dev is OVERWRITTEN with eigenvectors (one per column)
    //      eigenvalues_dev contains the eigenvalues in ascending order
    cusolverDnDsyevd(handle, jobz, uplo, N,
                      H_dev, N, eigenvalues_dev,
                      workspace_dev, workspace_size, info_dev);

    // --- 6. Check if it worked
    int info = 0;
    cudaMemcpy(&info, info_dev, sizeof(int), cudaMemcpyDeviceToHost);
    if (info != 0) {
        printf("diagonalize failed: info = %d\n", info);
    }

    // --- 7. Clean up
    cudaFree(workspace_dev);
    cudaFree(info_dev);
    cusolverDnDestroy(handle);
}