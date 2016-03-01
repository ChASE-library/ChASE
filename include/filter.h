#ifndef OMP_FILTER
#define OMP_FILTER

#include <iostream>
using namespace std;

#include <complex>
#define MKL_Complex16 complex<double>

#include <mkl.h>
#include <omp.h>

void filter(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m,
            int deg, double lambda_1, double lower, double upper,
            MKL_Complex16 *y);

void filterModified(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m, int nev,
                    int M, int *deg, double lambda_1, double lower, double upper,
                    MKL_Complex16 *y, int block);

#endif // OMP_FILTER
