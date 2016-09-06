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

static std::size_t chase_max_deg = 50;
static std::size_t chase_filtered_vecs = 0;
static std::size_t chase_deg_extra = 2;
static std::size_t chase_max_iter = 10;
static std::size_t chase_lanczos_iter = 10;
static std::size_t chase_iteration_count = 0;

std::size_t get_iter_count();
std::size_t get_filtered_vecs();


extern "C" {
void c_chase_(MKL_Complex16* H, int N, MKL_Complex16* V, MKL_Complex16* W,
              double* ritzv, int* nev, int* nex, int* deg,
              double *tol, char* mode, char* opt);
}

void chase(MKL_Complex16* H, std::size_t N, MKL_Complex16* V, MKL_Complex16* W,
           double* ritzv, std::size_t nev, const std::size_t nex,  const std::size_t deg, std::size_t* const degrees,
           const double tol, const CHASE_MODE_TYPE mode, const CHASE_OPT_TYPE opt);

void ColSwap( MKL_Complex16 *V, std::size_t N, std::size_t i, std::size_t j );

std::size_t calc_degrees( std::size_t N, std::size_t unconverged, std::size_t core,  double upperb, double lowerb,
                  double tol, double *ritzv, double *resid, std::size_t *degrees,
                  MKL_Complex16 *V, MKL_Complex16 *W);

std::size_t locking( std::size_t N, std::size_t unconverged, double tol,
             double *ritzv, double *resid, std::size_t *degrees,
             MKL_Complex16 *V);

void calc_residuals( std::size_t N, std::size_t unconverged, double tol,
           double *ritzv, double *resid,
           MKL_Complex16 *H, MKL_Complex16 *V, MKL_Complex16 *W);

void QR( std::size_t N, std::size_t nevex, std::size_t converged, MKL_Complex16 *W,
         MKL_Complex16 *tau, MKL_Complex16 *saveW );

void RR( std::size_t N, std::size_t block, double *Lambda,
         MKL_Complex16 *H, MKL_Complex16 *V, MKL_Complex16 *W );

#endif  // CHASE_CHASE_H
