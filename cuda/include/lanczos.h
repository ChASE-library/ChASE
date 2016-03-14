#ifndef CHASE_LANCZOS_H
#define CHASE_LANCZOS_H

#include <iostream>

#include <cstring> // memcpy
#include <functional>
#include <complex>
#include <random>
#include <algorithm>

#define MKL_Complex16 std::complex<double>

#include <mkl_cblas.h>
#include <mkl_lapacke.h>

#define CHASE_LANCZOS_FULLV true

int lanczos(const MKL_Complex16 *H, int N, int numvec, int m, int nevex, double *upperb,
             bool mode, double *ritzv_, MKL_Complex16 *V_);

void lanczosM( const MKL_Complex16 *H, int n, int m, double *bound,
               bool ctrl, MKL_Complex16 *V, double *ritzv,  double *ritzV );

#endif // CHASE_LANCZOS_H
