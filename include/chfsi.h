#ifndef CHASE_CHASE_H
#define CHASE_CHASE_H

#include <iostream>
using namespace std;

#include <complex>
#define MKL_Complex16 complex<double>

#include <cstdlib>
#include <string>
#include <iomanip>
#include <fstream>
#include <assert.h>
#include <omp.h>
#include <cstring>
#include <random>

//#include <mkl_cblas.h>
//#include <mkl_lapacke.h>
#include <mkl.h>


#define CHASE_MODE_TYPE char
#define CHASE_MODE_RANDOM CHASE_MODE_TYPE('R')
#define CHASE_MODE_APPROX CHASE_MODE_TYPE('A')

#define CHASE_OPT_TYPE char
#define CHASE_OPT_NONE CHASE_OPT_TYPE('N')
#define CHASE_OPT_SINGLE CHASE_OPT_TYPE('S')

static bool omp_stat = true;
static int  omp_iteration = 0;
static int  chase_filteredVecs = 0;
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
void swapEigPair(int i, int j, MKL_Complex16* ztmp, int N, MKL_Complex16* V, double *Lambda, int* P, const CHASE_OPT_TYPE int_opt);
void applyPerm2eigPair(int** perm, int n, MKL_Complex16* ztmp, int N, MKL_Complex16* V, double *Lambda);

void get_iteration(int* iteration);
void get_time(double* time);
void get_filteredVecs(int* iterations);

void chfsi(MKL_Complex16* const H, int N, MKL_Complex16* V, MKL_Complex16* W,
           double* ritzv, int nev, const int nex,  const int deg,
           const double tol, const CHASE_MODE_TYPE mode, const CHASE_OPT_TYPE opt);

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

#endif  // CHASE_CHASE_H
