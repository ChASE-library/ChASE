#ifndef OMP_LANCZOS
#define OMP_LANCZOS

#include <iostream>
using namespace std;

#include <complex>
#define MKL_Complex16 std::complex<double>

#include <mkl.h>

void lanczos(MKL_Complex16 *B, MKL_Complex16 *v, int n, int blk, int m, double tol, int ctrl, double *ritzv, double *bound);

#endif // OMP_LANCZOS
