/* -*- Mode: C++; -*- */
#pragma once
#include <complex>

#define CHASE_VERSION_MAJOR 0
#define CHASE_VERSION_MINOR 9

namespace chase {
typedef int CHASE_INT;
typedef int MPI_Int;
typedef int CHASE_MPIINT;

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

#ifdef HAS_ELEMENTAL
/*
// Elemental has its own El::Complex type
template <class Q>
struct Base_Class<El::Complex<Q>> {
  typedef Q type;
};
*/
#endif

template <typename Q>
using Base = typename Base_Class<Q>::type;
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
