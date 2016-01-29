/*! \file filter.hpp
 *      \brief Header file for Chebyshev filter
 *      \details This file contains a single function, an implementation of the Chebyshev polynomial filter.
 *  */

/// \cond

#ifndef ELECHFSI_FILTER
#define ELECHFSI_FILTER

#include "El.hpp"
using namespace El;
using namespace std;

#include <assert.h>

/// \endcond

/** \fn filter(UpperOrLower uplo, const DistMatrix<F>& A, DistMatrix<F>& V, DistMatrix<F>& W,
           int start, int width, const int deg, int const* const degrees,
           const int deglen, const R lambda, const R lower, const R upper)
 * \brief Chebyshev filter
 * \details This function implements the Chebyshev filter. It is a filter of vectors, based on the Chebyshev polynomials (refference?).
 * The concrete implementation here surppresses the vector components that correspond to the eigenvectors of the eigenvalues larger than lambda (input parameter).
 * The result of applying this function on a number of vectors is getting them alligned to a smaller subspace, with the purpose
 * of accelerating the convergence of the process. More explanation on the reasons for using the filter here can be found in (refference).
 *
 * \param applyA        Black-box operator which inputs (alpha,X,beta,Y) and overwrites Y := alpha X + beta Y
 * \param V             Elemental DistMatrix of template type F. Contains the input eigenvectors, that are to be filtered.
 *                      Each vector is a separate column in the matrix V. They have to be sorted according to the degrees parameter.
 * \param start Integer, specifying the index of the starting vector (in the matrix of vectors V), to be filtered.
 * \param width Integer, specifying the (block width) number of vectors after 'start', that are to be filtered.
 * \param deg           Integer, specifying the maximum degree that the filter could use.
 * \param degrees       Pointer to an array of integers of length deglen. Contains the degrees for each vectors that are to be filtered.
 *                      Any value higher than 'deg' is ignored, and 'deg' is used instead. The array should be sorted ascending positive.
 * \param deglen        Integer. Length of the array of degrees.
 * \param lambda        Real. Upper estimate of the eigenvalue problem. Any vector components corresponding to eigenvalues larger
 *                      than lambda are dampered.
 * \param lower Real. Lower bound of the eigenvalue spectrum.
 * \param upper Real. Upper bound of the eigenvalue spectrum
 *
 * \param W           Elemental DistMatrix of template type F. Contains the filtered eigenvectors after the function terminates.
 *
 * \return  int A positive integer, the number of filtered vectors
 */
template<typename F,class ApplyAClass>
int filter
( const ApplyAClass& applyA,
  DistMatrix<F>& V,
  DistMatrix<F>& W,
  int start,
  int width,
  const int deg,
  int const* const degrees,
  const int deglen,
  const Base<F> lambda,
  const Base<F> lower,
  const Base<F> upper)
{
  typedef Base<F> Real;

  Real c = (upper + lower)/2;
  Real e = (upper - lower)/2;

  Real sigma_scale = e/(lambda - c);
  Real sigma = sigma_scale;
  Real sigma_new;
  F alpha;
  F beta;

  int total_vcts_filtered = 0;

  int degmax = deg;
  if (degrees != NULL)
    degmax = (deg >= degrees[deglen-1]) ? deg : degrees[deglen-1];

  int local_vcts_filtered = 0, j = 0;

  if (degmax == 0) return total_vcts_filtered;

  alpha = F(sigma_scale/e);
  beta  = 0;

  /*
  if ( mpi::Rank( mpi::COMM_WORLD ) == 0 ) {
    std::cout << deg << " " << deglen << std::endl;
    if( degrees != NULL ) {
      std::cout << "degs: "
        for ( size_t i = 0; i < deglen; ++i ) {
          std::cout << degrees[i] << " ";
        }
      std::cout << std::endl;
    }
  }
  */

  /*** First filtering step. ***/
  // W := alpha*(A-cI) V + beta W = [alpha A V + beta W] - alpha c V
  {
      auto VBlock = V( ALL, IR(start,start+width) );
      auto WBlock = W( ALL, IR(start,start+width) );
      applyA( alpha, VBlock, beta, WBlock );
      Axpy( -alpha*c, VBlock, WBlock );
  }

  total_vcts_filtered += width;

  // Those that had deglen[j] = 1 are now filtered.
  if (degrees != NULL)
  {
      while (j < deglen && degrees[j] == 1) ++j;
      local_vcts_filtered = j;
      width -= j;
      start += j;
  }

  /*** Remaining filtering steps. ***/
  for (int i = 2 ; i <= degmax ; ++i)
  {
      sigma_new = Real(1)/(Real(2)/sigma_scale - sigma);

      // x = alpha(A-cI)y + beta*x
      alpha = 2*sigma_new / e;
      beta  = -sigma*sigma_new;

      auto VBlock = V( ALL, IR(start,start+width) );
      auto WBlock = W( ALL, IR(start,start+width) );

      // Apply translated matrix (double buffering for performance).
      if (i%2 == 0)
      {
          applyA( alpha, WBlock, beta, VBlock );
          Axpy( -alpha*c, WBlock, VBlock ); 
      }
      else
      {
          applyA( alpha, VBlock, beta, WBlock );
          Axpy( -alpha*c, VBlock, WBlock );
      }

      total_vcts_filtered += width;

      sigma = sigma_new;

      // Those that had degrees[j] = i are now filtered.
      if (degrees != NULL)
      {
          local_vcts_filtered = j;
          while (j < deglen && degrees[j] == i) ++j;
          local_vcts_filtered = j - local_vcts_filtered;
          width -= local_vcts_filtered;

          if (i%2 == 0 && local_vcts_filtered > 0)
          {
              auto VBlockFilt = V( ALL, IR(start,start+local_vcts_filtered) );
              auto WBlockFilt = W( ALL, IR(start,start+local_vcts_filtered) );
              WBlockFilt = VBlockFilt;
          }

          start += local_vcts_filtered;
      }

  }  // for (int i = 2 ; i <= degmax ; ++i)

  // If the filltered vectors are not in W (output), copy them there.
  if (degmax%2 == 0)
  {
      auto VBlock = V( ALL, IR(start,start+width) );
      auto WBlock = W( ALL, IR(start,start+width) );
      WBlock = VBlock;
  }

  return total_vcts_filtered;
}

#endif  // ELECHFSI_FILTER
