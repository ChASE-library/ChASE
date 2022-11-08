/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#ifndef CHASE_ALGORITHM_TYPES_HPP
#define CHASE_ALGORITHM_TYPES_HPP

#include <mpi.h>
#include <complex>
#include <functional>
#include <fstream>
#include <sys/stat.h>

namespace chase {

// Base< std::complex< Base<T> > > -> Base<T>
// Base<               Base<T>   > -> Base<T>
template <class Q>
struct Base_Class {
  typedef Q type;
};

template <class Q>
struct Base_Class<std::complex<Q>> {
  typedef Q type;
};

template <typename Q>
using Base = typename Base_Class<Q>::type;
}  // namespace chase

bool isPathExist(const std::string &s)
{
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}

template <typename T>
T getRandomT(std::function<double(void)> f);

template <>
double getRandomT(std::function<double(void)> f) {
  return double(f());
}

template <>
float getRandomT(std::function<double(void)> f) {
  return float(f());
}

template <>
std::complex<double> getRandomT(std::function<double(void)> f) {
  return std::complex<double>(f(), f());
}

template <>
std::complex<float> getRandomT(std::function<double(void)> f) {
  return std::complex<float>(f(), f());
}

#endif
