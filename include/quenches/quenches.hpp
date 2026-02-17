#pragma once
#include <cusolverDn.h>

// Step 2
void diagonalize(double* H_dev, double* E_dev, int D);

// Step 3
void project_state(double* eigenvectors_dev, double* psi0_dev, 
                    double* coeffs_dev, int D);

// Step 4
void time_evolve(double* eigenvalues_dev, double* coeffs_dev,
                 double* psi_t_dev, double t, int D);

// Step 5
double measure(double* psi_t_dev, double* observable_dev, int D);