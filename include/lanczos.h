#ifndef CHASE_LANCZOS_H
#define CHASE_LANCZOS_H

#include <complex>
#define MKL_Complex16 std::complex<double>

#include <mkl.h>
#include <cstring> // memcpy
#include <iostream> // memcpy

void lanczos(MKL_Complex16 *B, MKL_Complex16 *v, int n, int blk, int m,
             double tol, int ctrl, double *ritzv, double *bound);

#endif // CHASE_LANCZOS_H
