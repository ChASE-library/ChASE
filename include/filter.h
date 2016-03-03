#ifndef CHASE_FILTER_H
#define CHASE_FILTER_H

#include <complex>

#ifndef MKL_Complex16
#define MKL_Complex16 std::complex<double>
#endif

#include <mkl_cblas.h>

void filter(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m,
            int deg, double lambda_1, double lower, double upper,
            MKL_Complex16 *y);

void filterModified(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m, int nev,
                    int M, int *deg, double lambda_1, double lower, double upper,
                    MKL_Complex16 *y, int block);

#endif // CHASE_FILTER_H
