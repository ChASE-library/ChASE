#ifndef CHASE_FILTER_H
#define CHASE_FILTER_H

#include <complex>

#ifndef MKL_Complex16
#define MKL_Complex16 std::complex<double>
#endif

#include <mkl_cblas.h>

std::size_t filter(MKL_Complex16 *A, MKL_Complex16 *x, std::size_t n, std::size_t m,
                    std::size_t M, std::size_t *deg, double lambda_1, double lower, double upper,
                    MKL_Complex16 *y);

#ifdef CHASE_BUILD_CUDA
std::size_t cuda_filter( MKL_Complex16 *H, MKL_Complex16 *V, std::size_t n, std::size_t unprocessed,
            std::size_t deg, std::size_t *degrees, double lambda_1, double lower, double upper,
                 MKL_Complex16 *W );
#endif

#endif // CHASE_FILTER_H
