/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#ifndef CHASE_ALGORITHM_ALGORITHM_HPP
#define CHASE_ALGORITHM_ALGORITHM_HPP

#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <random>

#include "interface.hpp"

namespace chase
{

template <class T>
class Algorithm
{
public:
    static void solve(Chase<T>* single);

    static std::size_t calc_degrees(Chase<T>* kernel, std::size_t N,
                                    std::size_t unconverged, std::size_t nex,
                                    Base<T> upperb, Base<T> lowerb, Base<T> tol,
                                    Base<T>* ritzv, Base<T>* resid,
                                    Base<T>* residLast, std::size_t* degrees,
                                    std::size_t locked);

    static std::size_t locking(Chase<T>* kernel, std::size_t N,
                               std::size_t unconverged, Base<T> tol,
                               Base<T>* ritzv, Base<T>* resid,
                               Base<T>* residLast, std::size_t* degrees,
                               std::size_t locked);

    static std::size_t filter(Chase<T>* kernel, std::size_t n,
                              std::size_t unprocessed, std::size_t deg,
                              std::size_t* degrees, Base<T> lambda_1,
                              Base<T> lower, Base<T> upper);

    static std::size_t lanczos(Chase<T>* kernel, int N, int numvec, int m,
                               int nevex, Base<T>* upperb, bool mode,
                               Base<T>* ritzv_);
};

template <typename T>
void Solve(Chase<T>* single)
{
    Algorithm<T>::solve(single);
}

} // namespace chase

#include "algorithm.inc"

#endif
