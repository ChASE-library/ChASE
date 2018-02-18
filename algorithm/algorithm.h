/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */
#ifndef CHASE_CHASE_H
#define CHASE_CHASE_H

//      ________    ___   _____ ______
//     / ____/ /_  /   | / ___// ____/
//    / /   / __ \/ /| | \__ \/ __/
//   / /___/ / / / ___ |___/ / /___
//   \____/_/ /_/_/  |_/____/_____/

#include <assert.h>
#include <algorithm>
#include <iomanip>
#include <random>

#include "algorithm/interface.h"
#include "algorithm/performance.h"

namespace chase {

template <class T>
class Algorithm {
 public:
  static ChasePerfData solve(Chase<T>* single, std::size_t N, Base<T>* ritzv,
                             std::size_t nev, const std::size_t nex, Base<T>* resid_);

  static std::size_t calc_degrees(Chase<T>* single, std::size_t N,
                                  std::size_t unconverged, std::size_t nex,
                                  Base<T> upperb, Base<T> lowerb, Base<T> tol,
                                  Base<T>* ritzv, Base<T>* resid, Base<T>* residLast,
                                  std::size_t* degrees, std::size_t locked);

  static std::size_t locking(Chase<T>* single, std::size_t N,
                             std::size_t unconverged, Base<T> tol,
                             Base<T>* ritzv, Base<T>* resid, Base<T>* residLast,
                             std::size_t* degrees, std::size_t locked);

  static std::size_t filter(Chase<T>* single, std::size_t n,
                            std::size_t unprocessed, std::size_t deg,
                            std::size_t* degrees, Base<T> lambda_1,
                            Base<T> lower, Base<T> upper);

  static std::size_t lanczos(Chase<T>* single, int N, int numvec, int m,
                             int nevex, Base<T>* upperb, bool mode,
                             Base<T>* ritzv_);
};
}

#include "algorithm.inc"

#endif  // CHASE_CHASE_H
