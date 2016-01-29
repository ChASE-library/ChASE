#ifndef OMP_FILTER
#define OMP_FILTER

#include <iostream>
using namespace std;

#include <complex>
#define MKL_Complex16 complex<double>

#include <mkl.h>
#include <omp.h>

#include "cuda_util.h"

void filter(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m,
            int deg, double lambda_1, double lower, double upper,
            MKL_Complex16 *y, gpu_data_t *gpuData);

void cuda_filter(double _Complex *A, double _Complex*x, int n, int m,
            int deg, double lambda_1, double lower, double upper,
            double _Complex*y, gpu_data_t *gpuData);

void filterModified(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m, int nev,
                    int M, int *deg, double lambda_1, double lower, double upper,
                    MKL_Complex16 *y, int block, gpu_data_t *gpuData);

void cuda_filterModified(double _Complex *A, double _Complex *x, int n, int m, int nev,
                    int M, int *deg, double lambda_1, double lower, double upper,
                    double _Complex *y, int block, gpu_data_t *gpuData);

#endif // OMP_FILTER
