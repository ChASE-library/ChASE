#ifndef CHASE_FILTER_H
#define CHASE_FILTER_H

#include <complex>

#ifndef MKL_Complex16
#define MKL_Complex16 std::complex<double>
#endif

#include <mkl_cblas.h>

int filter(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m,
                    int M, int *deg, double lambda_1, double lower, double upper,
                    MKL_Complex16 *y);

#ifdef CHASE_BUILD_CUDA
int cuda_filter( MKL_Complex16 *H, MKL_Complex16 *V, int n, int unprocessed,
            int deg, int *degrees, double lambda_1, double lower, double upper,
                 MKL_Complex16 *W );
#endif

#endif // CHASE_FILTER_H
