#ifndef CHASE_FILTER_H
#define CHASE_FILTER_H

#include <complex>

#ifndef MKL_Complex16
#define MKL_Complex16 std::complex<double>
#endif

#include <mkl_cblas.h>
#include "cuda_util.h"

int cuda_filter(
  cuDoubleComplex *H, cuDoubleComplex *dV_, int n, int unprocessed,
  int deg, int *degrees, double lambda_1, double lower, double upper,
  cuDoubleComplex *dW_, cudaStream_t *stream, cublasHandle_t *handle
  );


#endif // CHASE_FILTER_H
