#ifndef CHASE_CHASE_H
#define CHASE_CHASE_H

//      ________    ___   _____ ______
//     / ____/ /_  /   | / ___// ____/
//    / /   / __ \/ /| | \__ \/ __/
//   / /___/ / / / ___ |___/ / /___
//   \____/_/ /_/_/  |_/____/_____/

template< class T >
class ChASE_Algorithm {

public:

  static ChASE_PerfData
  solve(
        ChASE< T > *single,
        std::size_t N,
        Base< T > * ritzv,
        std::size_t nev,
        const std::size_t nex
        );

  static std::size_t
  calc_degrees(
               ChASE< T > *single,
               std::size_t N, std::size_t unconverged,
               std::size_t nex,
               Base< T > upperb,
               Base< T > lowerb,
               Base< T > tol,
               Base< T > *ritzv,
               Base< T > *resid,
               std::size_t *degrees,
               std::size_t locked
               );

  static std::size_t
  locking(
          ChASE< T > *single,
          std::size_t N,
          std::size_t unconverged,
          Base< T > tol,
          Base< T > *ritzv,
          Base< T > *resid,
          std::size_t *degrees,
          std::size_t locked
          );

  static std::size_t
  filter(
         ChASE< T > *single,
         std::size_t n,
         std::size_t unprocessed,
         std::size_t deg,
         std::size_t *degrees,
         Base<T> lambda_1,
         Base<T> lower,
         Base<T> upper
         );

  static std::size_t
  lanczos(
          ChASE< T > *single,
          int N,
          int numvec,
          int m,
          int nevex,
          Base<T> *upperb,
          bool mode,
          Base<T> *ritzv_
          );
};


#include "ChASE_Algorithm_impl.hpp"

#endif  // CHASE_CHASE_H
