#ifndef CHASE_CHASE_H
#define CHASE_CHASE_H

#include <iostream>
#include <complex>
#include <string>
#include <random>
#include <algorithm>
#include <omp.h>
#include <assert.h>

#ifndef MKL_Complex16
#define MKL_Complex16 std::complex<double>
#endif

#include <mkl_cblas.h>
#include <mkl_lapacke.h>

#include "../include/lanczos.h"
#include "../include/filter.h"
#include "../include/timing.h"

#define CHASE_MODE_TYPE char
#define CHASE_MODE_RANDOM CHASE_MODE_TYPE('R')
#define CHASE_MODE_APPROX CHASE_MODE_TYPE('A')

#define CHASE_OPT_TYPE char
#define CHASE_OPT_NONE CHASE_OPT_TYPE('N')
#define CHASE_OPT_SINGLE CHASE_OPT_TYPE('S')

static int chase_max_deg = 50;
static int chase_filtered_vecs = 0;
static int chase_deg_extra = 2;
static int chase_max_iter = 10;
static int chase_lanczos_iter = 10;
static int chase_iteration_count = 0;

int get_iter_count();
int get_filtered_vecs();

void chase(MKL_Complex16* H, int N, MKL_Complex16* V, MKL_Complex16* W,
           double* ritzv, int nev, const int nex,  const int deg, int* const degrees,
           const double tol, const CHASE_MODE_TYPE mode, const CHASE_OPT_TYPE opt);

void ColSwap( MKL_Complex16 *V, int N, int i, int j );

int calc_degrees( int N, int unconverged, int core,  double upperb, double lowerb,
                  double tol, double *ritzv, double *resid, int *degrees,
                  MKL_Complex16 *V, MKL_Complex16 *W);

int locking( int N, int unconverged, double tol,
             double *ritzv, double *resid, int *degrees,
             MKL_Complex16 *V);

void calc_residuals( int N, int unconverged, double tol,
           double *ritzv, double *resid,
           MKL_Complex16 *H, MKL_Complex16 *V, MKL_Complex16 *W);

void QR( int N, int nevex, int converged, MKL_Complex16 *W,
         MKL_Complex16 *tau, MKL_Complex16 *saveW );

void RR( int N, int block, double *Lambda,
         MKL_Complex16 *H, MKL_Complex16 *V, MKL_Complex16 *W );

#endif  // CHASE_CHASE_H
