#pragma once

#include <cuda_runtime.h>
#include <cusolverDn.h>

// Diagonalize a real symmetric matrix (Hermitian / quantum mechanical)
// H_dev - on input: the matrix (N x N). On output: eigenvectors as columns.
// eigenvalues_dev - output eigenvalues (N)
// N - dimension
void diagonalize(double* H_dev, double* eigenvalues_dev, int N);

// Matrix-vector product: y = A * x
void matvec(double* A_dev, double* x_dev, double* y_dev, int D);

// Dot product: result = x . y
double dot(double* x_dev, double* y_dev, int D);