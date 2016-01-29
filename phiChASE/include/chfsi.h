#ifndef OMP_CHFSI
#define OMP_CHFSI

#include <iostream>
using namespace std;


#include <cstdlib>
#include <string>
#include <iomanip>
#include <fstream>
#include <assert.h>
#include <mkl.h>
#include <mkl_vsl.h>
#include <omp.h>
#include <cstring>

#define OMP_RANDOM 1
#define OMP_APPROX 0

#define OMP_NO_OPT       0
#define OMP_OPT          1

#define OMP_CPU      0
#define OMP_GPU      1
#define OMP_XEON_PHI 2

static bool omp_stat = true;
static int  omp_iteration;
static double omp_time[10] = {0.0};
static int  omp_degmax = 50;
static int  omp_degrees_len;
static int* omp_degrees_ell = NULL;
static int  omp_delta = 2;
static int  omp_maxiter = 35;
static int  omp_lanczos = 10;

double minValue(double *v, int N);
double maxValue(double *v, int N);
int sortComp (const void * a, const void * b);
void applyPerm2vect(int** perm, int n, int shift, MKL_Complex16* tmp, int N, MKL_Complex16* V);
void swapEigPair(int i, int j, MKL_Complex16* ztmp, int N, MKL_Complex16* V, double *Lambda, int* P, int int_opt);
void applyPerm2eigPair(int** perm, int n, MKL_Complex16* ztmp, int N, MKL_Complex16* V, double *Lambda);

void get_iteration(int* iteration);
void get_time(double* time);

void chfsi(MKL_Complex16* const H, int N, MKL_Complex16* V, MKL_Complex16* W, double* ritzv, int nev, const int nex,  const int deg, const double tol, const int mode, const int opt, const int arch);

#define BGN_TOTAL   0
#define BGN_LANCZOS 1
#define BGN_FILTER  2
#define BGN_RR      3
#define BGN_CONV    4

#define END_TOTAL   5
#define END_LANCZOS 6
#define END_FILTER  7
#define END_RR      8
#define END_CONV    9

#define MINDEG  0
#define MAXDEG  1
#define SORTED  1
#define ALLOWED 2
#define PINV    0
#define WHERE   3
#define PERM    4

#define RHO 0
#define SP  1

#endif  // OMP_CHFSI
