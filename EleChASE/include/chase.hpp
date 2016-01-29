/*! \file chfsi.hpp
 *      \brief Header file for the main algorithm
 *      \details This file contains the central function of ChASE, plus some auxiliary functions.
 *  */

/// \cond
#ifndef ELECHFSI_CHASE
#define ELECHFSI_CHASE

#include "El.hpp"
using namespace El;

#include "../include/lanczos.hpp"
#include "../include/filter.hpp"

#include <assert.h>
#include <cstdlib>
#include <algorithm>

#include <iomanip> // std::setfill, std::setw, std::setprecision
#include <sstream> // std::ostringstream

/// \endcond

/// \cond

// TODO: Use enums here?
#define ELECHFSI_RANDOM 1
#define ELECHFSI_APPROX 0

#define BGN_TOTAL   0
#define BGN_LANCZOS 1
#define BGN_FILTER  2
#define BGN_REDUCED 3
#define BGN_CONV    4

#define END_TOTAL   5
#define END_LANCZOS 6
#define END_FILTER  7
#define END_REDUCED 8
#define END_CONV    9
#define END_QR      10
#define BGN_QR      11

#define MINDEG  0
#define MAXDEG  1
#define SORTED  1
#define ALLOWED 2
#define PINV    0
#define WHERE   3
#define PERM    4

#define RHO 0
#define SP  1

#define ELECHFSI_NO_OPT       0
#define ELECHFSI_OPT_SINGLE   1
#define ELECHFSI_OPT_MULTIPLE 2

/// \endcond

/** \fn chase(UpperOrLower uplo, const DistMatrix<F>& H, DistMatrix<F>& V, DistMatrix<F>& W,
              DistMatrix<R, VR, STAR> &Lambda, const int nev, const int nex,
              const int deg, int* const degrees, const R tol, R* const resid,
              const int mode, const int opt)
 * \brief Principal function of the ChASE library.
 * \details The ChASE function computes a relatively small subset of eigenpairs of a dense real symmetric or complex Hermitian
 * matrix H. This function implements most of the functionalities of the Chebyshev Accelerated Subspace iteration Eigensolver.
 * The function arguments can be used by an external driver to call the library. This is the distributed memory version of the
 * ChASE function. This version uses the Elemental library in its pure MPI built.
 *
 * \param uplo          [input] STRING. <br>
 *                      <tt>= LOWER</tt> if lower triangular symmetric (Hermitian) matrix H is inputted. <br>
 *                      <tt>= UPPER</tt> if upper triangular symmetric (Hermitian) matrix H is inputted. <br>
 * \param H             [output] DOUBLE COMPLEX array - dimension (n,n). <br>
 *                      Contains the input hermitian matrix distributed according to the Elemental DistMatrix template of type F. <br>
 *                      If <tt>uplo = UPPER/LOWER</tt> only the leading upper/lower triangular part of H stores the entries of the
 *                      upper/lower entries of the matrix H while the strictly lower/upper triangular part of H is not referenced.
 * \param V             [input/output] DOUBLE COMPLEX array - dimension (n,nex). <br>
 *                      It contains a collection of vectors distributed according to the Elemental DistMatrix template of type F. <br>
 *                      On input, if <tt>mode = ELECHFSI_APPROX</tt>, it is used to provide  approximate eigenvectors which work as a
 *                      preconditioner, speeding up the convergence of required eigenpairs. <br>
 *                      On output, it contains the converged eigenvectors ordered as the corresponding eigenvalues in <tt>Lambda</tt>.
 * \param Lambda        [input/output] DOUBLE array - dimension(1,n). <br>
 *                      It contains and ordered collection of real numbers distributed according to the Elemental DistMatrix
 *                      of type R. <br>
 *                      On input, if <tt>mode = ELECHFSI_APPROX</tt>, it used to provide initial approximate eigenvalues. <br>
 *                      On output contains <tt>nev</tt> converged eigenvalues ordered from the lowest to the highest.
 * \param nev           [input] INTEGER. <br>
 *                      It specifies the number of required eigenvalues. This is usually a fraction of the total eigenspectrum.
 *                      The maximum value of such fraction dependes on the size of the eigenproblem but as a rule of thumb should
 *                      not exceed 20-30% in order to use ChASE efficiently.
 * \param nex           [input] INTEGER. <br>
 *                      It specifies the initial size of the search subspace <i>S</i> such that size(<i>S</i>) = <tt>(nev+nex)</tt>.
 *                      Its correct choice represents a trade-off between extra computations and an enhanced eigenpairs convergence
 *                      ratio. Ideally it should be a fraction of <tt>nev</tt> guaranteeing to include a spectral gap between the
 *                      first <tt>nev</tt> and the next <tt>(nev+nex)</tt> eigenvalues. Example: for <tt>nev = 250</tt>, a value of
 *                      <tt>nex = 50</tt> should suffice unless eigenvalues between index 250 and 300 are quite clustered.
 * \param deg           [input] INTEGER. <br>
 *                      If <tt>opt = ELECHFSI_NO_OPT</tt>, it specifies the constant Chebyshev polynomial degree used in the filter
 *                      to enhance the convergence ration of sought after eigenvectors. Suggested value: <tt>deg = 25</tt>. <br>
 *                      If <tt>opt = ELECHFSI_OPT_SINGLE</tt>, it specifies the Chebyshev polynomial degree used in the first call to
 *                      the filter. Following the first call the optimal polynomial degree is computed on the fly. Suggested value:
 *                      <tt>deg = 10</tt>.
 * \param degrees       [input](optional) INTEGER array. <br>
 *                      Only used when ChASE is utilized to solve a sequence or correlated eigenproblems. Currently under
 *                      implementation.<br>
 *                      It contains <tt>nev</tt> elements, that specify the filter degrees used in the previous sequence eigenproblems.
 * \param tol           [input] DOUBLE. <br>
 *                      It specifies the accuracy of the solutions. <tt>tol</tt> is the minimum tolerance the residuals of the
 *                      required eigenpairs should have to be declared converged.
 * \param resid         [output](optional) DOUBLE array. <br>
 *                      It contains the residuals for the converged eigenpairs ordered as in the <tt>Lambda</tt> array.
 * \param mode          [input] STRING. <br>
 *                      It specifies whether approximate initial vectors are or not provided in input through <tt>V</tt>. <br>
 *                      <tt>= ELECHFSI_APPROX</tt> if <tt>V</tt>  and <tt>Lambda</tt> contain respectively the approximate
 *                      vectors and values on input. <br>
 *                      <tt>= ELECHFSI_RANDOM</tt> if the initial vectors have to be computed by the Lanczos method starting from a
 *                      set of random vectors. <br>
 * \param opt           [input] STRING. <br>
 *                      It specifies whether ChASE computes or not an array of optimal degrees for the filter. <br>
 *                      <tt>= ELECHFSI_NO_OPT</tt> if the same polynomial degree <tt>deg</tt> has to be used to filter all the vectors
 *                      in the seacrh space <i>S</i>. <br>
 *                      <tt>= ELECHFSI_OPT_SINGLE</tt> if the polynomial degree <tt>deg</tt> has to be used only for the first call
 *                      to the filter. Additional calls to the filter use an optimal polynomial degree tailored to each filtered
 *                      vector.
 */
template<typename F>
void chase
( UpperOrLower uplo,
  const DistMatrix<F>& H,
  DistMatrix<F>& V,
  DistMatrix<Base<F>,VR,STAR>& Lambda,
  const int nev,
  const int nex,
  const int deg,
  int* const degrees,
  const Base<F> tol,
  Base<F>* const resid,
  const int mode,
  const int opt );

bool get_stat();
void set_stat(bool stat);
int  get_filtered();
void set_filtered(int);
int  get_iteration();
void set_iteration(int);
void get_times(double*);
double get_time(int);
void set_time(const int, const double);
int  get_degmax(void);
void set_degmax(const int);
void get_degrees(int*, int*);
void init_degrees(int);
void set_degree(int, int);
int  get_degree(int);
int  get_delta();
void set_delta(int);
int  get_maxiter();
void set_maxiter(int);
int  get_lanczos();
void set_lanczos(int);

void swap_perm(int, int, int*, int*);

template <typename T>
void swap_kj(int k, int j, T* array)
{
  T tmp = array[k];
  array[k] = array[j];
  array[j] = tmp;
  return;
}

template<typename F>
void chase
( UpperOrLower uplo,
  const DistMatrix<F>& H,
  DistMatrix<F>& V,
  DistMatrix<Base<F>,VR,STAR>& Lambda,
  const int nev,
  const int nex,
  const int deg,
  int* const degrees,
  const Base<F> tol,
  Base<F>* const resid,
  const int mode,
  const int opt)
{
  typedef Base<F> Real;
  const Int N = V.Height();
  const Grid& grid = H.Grid();
  mpi::Comm comm = grid.Comm();

  // NOTE: Other than this routine currently calling FrobeniusNorm on H,
  //       it is entirely built around this black-box routine
  auto applyH =
    [&]( F alpha, const ElementalMatrix<F>& X, F beta, ElementalMatrix<F>& Y )
    { Hemm( LEFT, uplo, alpha, H, X, beta, Y ); };

  DistMatrix<F> H_reduced(grid), V_reduced(grid);
  DistMatrix<Real,VR,STAR> Lambda_reduced(grid);

  int converged, converged_old;  // Number of converged eigenpairs.
  int block = nev + nex;         // Decreases as the vectors are locked.
  int iteration;                 // Current iteration.

  Real lower, upper, lambda, c, e;  // To translate and scale Chebyshev polynomials.
  Real tmp, norm_1, norm_2, norm_H;         // For the computation of the residuals.
  Real corTol; // tolerance, corrected to use of residual

  DistMatrix<F> W(N, block, grid);

  // Will be used for the residual
  norm_H = TwoNorm( H );
  corTol = tol * max( norm_H, Real(1) );
  /*** Single optimization. Init. ***/
  int index;
  int* pi     = new int[nev];
  int* pi_inv = new int[nev];
  Real* rho = NULL;
  Real lambda_min;

  for (int j = 0; j < nev; ++j)
      pi[j] = pi_inv[j] = j;

  if (opt != ELECHFSI_NO_OPT) rho = new Real[nev];
  else                        rho = NULL;
  if (opt != ELECHFSI_NO_OPT) assert(degrees != NULL);
  if (opt == ELECHFSI_OPT_SINGLE)
      for (int j = 0; j < nev; ++j)
          degrees[j] = deg;
  if (opt == ELECHFSI_OPT_MULTIPLE)
      for (int j = 0; j < nev; ++j)
          if (degrees[j] > deg) degrees[j] = deg;

  // Initializing internal variables.
  if (get_stat() == true)
  {
      set_filtered(0);
      init_degrees(nev);
      for(int i = 0; i < 12 ; ++i)
          set_time(i, 0.0);
  }

  //   if (get_stat() == true || opt == ELECHFSI_OPT_MULTIPLE)
  //     for (int i = 0; i < nev ; ++i)
  //       set_degree(i, (degrees != NULL) ? degrees[i] : deg);

  set_time(BGN_TOTAL, mpi::Time());

  /*** Lanczos. ***/
  set_time(BGN_LANCZOS, mpi::Time());
  auto w0 = W( ALL, IR(0) );
  MakeUniform( w0 );

  if (mode == ELECHFSI_APPROX)
    lanczos
    ( applyH, w0, get_lanczos(),get_lanczos(),
      block, &upper, NULL, NULL );
  if (mode == ELECHFSI_RANDOM)
    lanczos
    ( applyH, w0, get_lanczos(), 2*block/*4*get_lanczos()*/,
      block, &upper, &lower, &lambda);

  mpi::Broadcast( upper, 0, comm );
  if (mode == ELECHFSI_RANDOM)
  {
      mpi::Broadcast( lower,  0, comm );
      mpi::Broadcast( lambda, 0, comm );
  }
  set_time(END_LANCZOS, mpi::Time() - get_time(BGN_LANCZOS));


  /*** Main loop. ***/
  converged = 0;
  iteration = 0;
  while (converged < nev && iteration < get_maxiter())
  {
      auto LambdaBlock = Lambda( IR(converged,converged+block), ALL );

      // Set the parameters lambda and lower for the filter.
      if (iteration != 0 || mode != ELECHFSI_RANDOM)
      {
          lambda = Min( LambdaBlock );
          lower = Max( LambdaBlock );
      }

      /*** Chebyshev filter. ***/
      // The input is assumed to be in V, the output will be in W.
      set_time(BGN_FILTER, mpi::Time());

      /* Single optimization. Ordering vectors according to filter degree. */
      if (opt != ELECHFSI_NO_OPT)
      {
          for (int j = converged; j < nev-1; ++j)
          {
              for (int k = j+1; k < nev; ++k)
              {
                  if (degrees[k] < degrees[j])
                  {
                      swap_kj(k, j, degrees);
                      swap_perm(k, j, pi, pi_inv);
                      ColSwap( V, k, j );
                  }
              }
          }
      }

      int filtered_ell = filter
        ( applyH, V, W, converged, block, deg,
          degrees ? degrees+converged : NULL,
          nev-converged, lambda, lower, upper);

      set_filtered(get_filtered()+filtered_ell);
      set_time(END_FILTER, get_time(END_FILTER) + mpi::Time() - get_time(BGN_FILTER));

      /*** Orthogonalization. ***/
      set_time(BGN_QR, mpi::Time());
      qr::ExplicitUnitary(W);
      set_time(END_QR, get_time(END_QR) + mpi::Time() - get_time(BGN_QR));

      /*** Reduce problem: construct, solve, apply back transformation. ***/
      set_time(BGN_REDUCED, mpi::Time());
      auto VBlock = V( ALL, IR(converged,converged+block) );
      auto WBlock = W( ALL, IR(converged,converged+block) );
      applyH( F(1), WBlock, F(0), VBlock );

      H_reduced.AlignWith(WBlock);
      H_reduced.Resize(block, block);
      Gemm(ADJOINT, NORMAL, F(1), WBlock, VBlock, F(0), H_reduced);

      HermitianEig( uplo, H_reduced, Lambda_reduced, V_reduced, ASCENDING );

      /* Single optimization. Sort the permutation. (The eigenpairs are sorted.) */
      std::sort(pi+converged, pi+nev);
      for (int j = 0; j < nev; ++j)
          pi_inv[pi[j]] = j;

      // Back transformation.
      LambdaBlock = Lambda_reduced;
      H_reduced.AlignWith(WBlock);
      H_reduced = V_reduced;
      Gemm( NORMAL, NORMAL, F(1), WBlock, H_reduced, F(0), VBlock );

      set_time(END_REDUCED, get_time(END_REDUCED) + mpi::Time() - get_time(BGN_REDUCED));

      /** Compute residuals. ***/
      // Copy non converged from V to W.
      set_time(BGN_CONV, mpi::Time());
      WBlock = VBlock;

      // Compute WBlockLeft = HW-WLambda (residuals) and...
      auto VBlockLeft = V( ALL, IR(converged,converged+block-nex) );
      auto WBlockLeft = W( ALL, IR(converged,converged+block-nex) );
      auto LambdaBlockLeft = Lambda( IR(converged,converged+block-nex), ALL );

      DiagonalScale( RIGHT, NORMAL, LambdaBlockLeft, WBlockLeft );
      applyH( F(1), VBlockLeft, F(-1), WBlockLeft );
      for (int i = converged; i < nev; ++i)
      {
          auto wi = W( ALL, IR(i) );
          /*
          tmp = LambdaBlockLeft.Get(i-converged, 0);
          norm_2 = Nrm2(wi);
          resid[i] = norm_2/((upper+Abs(tmp)));
          */

          // || H x - lambda x || / ( max( ||H||, |lambda| ) )
          // norm_2 <- || H x - lambda x ||
          norm_2 = FrobeniusNorm( wi );
          resid[i] = norm_2 / ( max( norm_H, Real(1) ) );
      }

      /*** Check for convergence. ***/
      converged_old = converged;
      for (int j = converged; j < nev; ++j)
      {
          if (resid[j] > corTol) continue;
          if (j != converged)
          {
              if (opt != ELECHFSI_NO_OPT) swap_kj(j, converged, degrees);
              swap_perm(j, converged, pi, pi_inv);
              RowSwap( Lambda, j, converged );
              ColSwap( V, j, converged );
              swap_kj(j, converged, resid);
          }
          converged++;
      }
      // Make sure that perm_inv[0] has the smallest eigpair.
      if (converged != converged_old)
      {
          index = pi_inv[0];
          lambda_min = Lambda.Get(index, 0);

          for (int j = 1; j < converged ; ++j)
          {
              tmp = Lambda.Get(j, 0);
              if (tmp < lambda_min)
              {
                  lambda_min = tmp;
                  index = j;
              }
          }

          if (index != pi_inv[0])
          {
              if (opt != ELECHFSI_NO_OPT) swap_kj(0, index, degrees);
              swap_perm(0, index, pi, pi_inv);
              RowSwap( Lambda, 0, index );
              ColSwap( V, 0, index );
              swap_kj(0, index, resid);
          }
      }

      block -= (converged-converged_old);

      /* Single optimization. Find optimal degrees. */
      if (opt != ELECHFSI_NO_OPT)
      {
          c = (upper+lower)/2;
          e = (upper-lower)/2;

          // Find the minimal degrees, and a maximal one.
          rho[pi_inv[0]] = (Lambda.Get(pi_inv[0], 0)-c)/e;
          tmp  = sqrt(rho[pi_inv[0]]*rho[pi_inv[0]]-1);
          tmp += fabs(rho[pi_inv[0]]);
          rho[pi_inv[0]] = (tmp >= 1) ? tmp : 1/tmp;

          for (int j = converged ; j < nev ; ++j)
          {
              rho[j] = (Lambda.Get(j, 0)-c)/e;
              tmp    = sqrt(rho[j]*rho[j]-1);
              tmp   += fabs(rho[j]);
              rho[j] = (tmp >= 1) ? tmp : 1/tmp;

              degrees[j] = ceil(fabs(log(resid[j]/(corTol))/log(rho[j])));
              degrees[j] = degrees[j] + get_delta();
              if (degrees[j] > get_degmax())
                  degrees[j] = get_degmax();
          }
      }

      // if (get_stat() == true || opt == ELECHFSI_OPT_MULTIPLE)
      //        for (int i = converged ; i < nev ; ++i)
      //          set_degree(pi_inv[i], get_degree(pi_inv[i])
      //                     +(degrees != NULL) ? degrees[i] : deg);

      iteration += 1;

      // If there are changes, copy the views.
      if(converged-converged_old)
      {
          auto VNewConv = V( ALL, IR(converged_old,converged) );
          auto WNewConv = W( ALL, IR(converged_old,converged) );
          WNewConv = VNewConv;
      }

      set_time(END_CONV, get_time(END_CONV) + mpi::Time() - get_time(BGN_CONV));
  }

  set_iteration(iteration);
  V = W;                                 // Need to check if this is really needed.
  herm_eig::Sort( Lambda, V, ASCENDING );

  //   if (opt == ELECHFSI_OPT_MULTIPLE)
  //     for (int i = 0; i < nev; ++i)
  //       degrees[i] = get_degree(i);

  set_time(END_TOTAL, get_time(END_TOTAL) + mpi::Time() - get_time(BGN_TOTAL));

  delete[] pi;
  delete[] pi_inv;
  if (rho != NULL) delete rho;

  return;
}

#endif  // ELECHFSI_CHASE
