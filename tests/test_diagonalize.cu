#include "linalg/linalg.hpp"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

// Test diagonalization using two checks:
//
// 1. Eigenpair check:  H * v_n = E_n * v_n  for each n
//
// 2. Cayley-Hamilton:  The characteristic polynomial of a matrix
//    evaluated at the matrix itself gives zero.
//    For eigenvalues E_0, E_1, ..., E_{N-1}:
//
//    p(H) = (H - E_0 I)(H - E_1 I)...(H - E_{N-1} I) = 0
//
//    For a small matrix (say 4x4), we can test this directly.

int main() {
    // --- A known 4x4 real symmetric matrix ---
    // Using a simple spin-like matrix so results are physically meaningful
    //
    //     | 1   0.5  0    0  |
    //     | 0.5  -1  0.5  0  |
    //     | 0   0.5  1   0.5 |
    //     | 0    0   0.5 -1  |
    //
    const int N = 4;
    double H_host[N * N] = {
         1.0, 0.5, 0.0, 0.0,
         0.5,-1.0, 0.5, 0.0,
         0.0, 0.5, 1.0, 0.5,
         0.0, 0.0, 0.5,-1.0
    };

    // Keep a copy since diagonalize overwrites H
    double H_copy[N * N];
    for (int i = 0; i < N * N; i++) H_copy[i] = H_host[i];

    // Allocate on GPU
    double* H_dev = nullptr;
    double* E_dev = nullptr;
    cudaMalloc(&H_dev, N * N * sizeof(double));
    cudaMalloc(&E_dev, N * sizeof(double));

    // Copy H to GPU
    cudaMemcpy(H_dev, H_host, N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Diagonalize
    diagonalize(H_dev, E_dev, N);

    // Copy results back to CPU
    // H_dev now contains eigenvectors (columns), E_dev contains eigenvalues
    double V[N * N];  // eigenvectors
    double E[N];      // eigenvalues
    cudaMemcpy(V, H_dev, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(E, E_dev, N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("=== Eigenvalues ===\n");
    for (int n = 0; n < N; n++) {
        printf("  E[%d] = %f\n", n, E[n]);
    }

    // ============================================================
    // TEST 1: Eigenpair check    H * v_n  =  E_n * v_n
    // ============================================================
    printf("\n=== Test 1: Eigenpair check (H*v = E*v) ===\n");
    double max_error_1 = 0.0;

    for (int n = 0; n < N; n++) {
        // Extract eigenvector n (column n of V, stored column-major)
        double v[N];
        for (int i = 0; i < N; i++) v[i] = V[n * N + i];

        // Compute H_copy * v
        double Hv[N] = {0};
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                Hv[i] += H_copy[j * N + i] * v[j];  // column-major

        // Compare Hv with E[n] * v
        double err = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = Hv[i] - E[n] * v[i];
            err += diff * diff;
        }
        err = sqrt(err);
        if (err > max_error_1) max_error_1 = err;
        printf("  eigenpair %d: |H*v - E*v| = %.2e %s\n",
               n, err, err < 1e-10 ? "PASS" : "FAIL");
    }

    // ============================================================
    // TEST 2: Cayley-Hamilton
    //
    // p(H) = (H - E_0 I)(H - E_1 I)(H - E_2 I)(H - E_3 I) = 0
    //
    // We build this product one factor at a time.
    // Start with result = I, then repeatedly:
    //   result = (H - E_n I) * result
    // ============================================================
    printf("\n=== Test 2: Cayley-Hamilton p(H) = 0 ===\n");

    // Start with result = identity
    double result[N * N] = {0};
    for (int i = 0; i < N; i++) result[i * N + i] = 1.0;

    // Multiply by each factor (H - E_n * I)
    for (int n = 0; n < N; n++) {
        // Build factor = H_copy - E[n] * I
        double factor[N * N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                factor[i * N + j] = H_copy[i * N + j] - (i == j ? E[n] : 0.0);

        // new_result = factor * result
        double new_result[N * N] = {0};
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    new_result[i * N + j] += factor[i * N + k] * result[k * N + j];

        for (int i = 0; i < N * N; i++) result[i] = new_result[i];
    }

    // Check that result is the zero matrix
    double max_error_2 = 0.0;
    for (int i = 0; i < N * N; i++) {
        if (fabs(result[i]) > max_error_2) max_error_2 = fabs(result[i]);
    }
    printf("  max |p(H)_ij| = %.2e %s\n",
           max_error_2, max_error_2 < 1e-8 ? "PASS" : "FAIL");

    // Clean up
    cudaFree(H_dev);
    cudaFree(E_dev);

    printf("\n=== Summary ===\n");
    printf("  Eigenpair max error:       %.2e\n", max_error_1);
    printf("  Cayley-Hamilton max error: %.2e\n", max_error_2);

    return 0;
}
