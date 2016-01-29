#ifndef OMP_FILTER
#define OMP_FILTER

#include <iostream>
using namespace std;


#include <mkl.h>
#include <omp.h>

void filter(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m,
            int deg, double lambda_1, double lower, double upper,
            MKL_Complex16 *y);
void filter_multiple_phis(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m,
            int deg, double lambda_1, double lower, double upper,
            MKL_Complex16 *y);

void filterModified(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m, int nev,
                    int M, int *deg, double lambda_1, double lower, double upper,
                    MKL_Complex16 *y, int block);

#define REUSE alloc_if(0) free_if(0)
#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)

#endif // OMP_FILTER
