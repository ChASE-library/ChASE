/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */
#pragma once

#include <complex>
#include <cstring>  //memcpy
#include <random>

namespace chase {

namespace ChASE_Config_Helper {
template <typename T>
std::size_t initMaxDeg(bool approx, bool optimization);

template <typename T>
std::size_t initDeg(bool approx, bool optimization);

template <typename T>
std::size_t initLanczosIter(bool approx, bool optimization);

template <typename T>
double initTolerance(bool approx, bool optimization);

// TODO: we require even degree!
// matrixFree/blasSkewed requires it

template <>
std::size_t initMaxDeg<double>(bool approx, bool optimization) {
  return 36;
};

template <>
std::size_t initDeg<double>(bool approx, bool optimization) {
  //  return 20;
  return 10;
};

template <>
std::size_t initLanczosIter<double>(bool approx, bool optimization) {
  return 25;
};

template <>
double initTolerance<double>(bool approx, bool optimization) {
  return 1e-10;
};

template <>
double initTolerance<float>(bool approx, bool optimization) {
  return 1e-5;
}

template <>
std::size_t initMaxDeg<std::complex<double> >(bool approx, bool optimization) {
  return initMaxDeg<double>(approx, optimization);
};

template <>
std::size_t initMaxDeg<float>(bool approx, bool optimization) {
  return initMaxDeg<double>(approx, optimization) / 2;
};

template <>
std::size_t initMaxDeg<std::complex<float> >(bool approx, bool optimization) {
  return 12;
  return initMaxDeg<double>(approx, optimization) / 2;
};

template <>
std::size_t initDeg<std::complex<double> >(bool approx, bool optimization) {
  return initDeg<double>(approx, optimization);
};

template <>
std::size_t initDeg<float>(bool approx, bool optimization) {
  return initDeg<double>(approx, optimization) / 2;
};

template <>
std::size_t initDeg<std::complex<float> >(bool approx, bool optimization) {
  return initDeg<double>(approx, optimization) / 2;
};

template <>
std::size_t initLanczosIter<std::complex<double> >(bool approx,
                                                   bool optimization) {
  return initLanczosIter<double>(approx, optimization);
};

template <>
std::size_t initLanczosIter<float>(bool approx, bool optimization) {
  return initLanczosIter<double>(approx, optimization) / 2;
};

template <>
std::size_t initLanczosIter<std::complex<float> >(bool approx,
                                                  bool optimization) {
  return initLanczosIter<double>(approx, optimization) / 2;
};

template <>
double initTolerance<std::complex<double> >(bool approx, bool optimization) {
  return initTolerance<double>(approx, optimization);
}

template <>
double initTolerance<std::complex<float> >(bool approx, bool optimization) {
  return initTolerance<float>(approx, optimization);
}
}  // ChASE_Config_Helper

template <class T>
class ChaseConfig {
 public:
  ChaseConfig(std::size_t _N, std::size_t _nev, std::size_t _nex)
      : N(_N),
        nev(_nev),
        nex(_nex),
        optimization(false),
        approx(false),
        mMaxIter(2000),
        mDegExtra(2) {
    mMaxDeg = ChASE_Config_Helper::initMaxDeg<T>(approx, optimization);
    deg = ChASE_Config_Helper::initDeg<T>(approx, optimization);
    mLanczosIter =
        ChASE_Config_Helper::initLanczosIter<T>(approx, optimization);
    tol = ChASE_Config_Helper::initTolerance<T>(approx, optimization);
  }

  bool use_approx() { return approx; }
  void setApprox(bool flag) { approx = flag; }

  bool do_optimization() { return optimization; }
  void setOpt(bool flag) { optimization = flag; }

  std::size_t getDeg() { return deg; }
  void setDeg(std::size_t _deg) { deg = _deg; }

  double getTol() { return tol; }
  void setTol(double _tol) { tol = _tol; }

  std::size_t getMaxDeg() { return mMaxDeg; }
  void setMaxDeg(std::size_t maxDeg_) { mMaxDeg = maxDeg_; }

  std::size_t getDegExtra() { return mDegExtra; }
  void setDegExtra(std::size_t degExtra) { mDegExtra = degExtra; }

  std::size_t getMaxIter() { return mMaxIter; }
  void setMaxIter(std::size_t maxIter) { mMaxIter = maxIter; }

  std::size_t getLanczosIter() { return mLanczosIter; }
  void setLanczosIter(std::size_t aLanczosIter) { mLanczosIter = aLanczosIter; }

  std::size_t getN() { return N; }

  std::size_t getNev() { return nev; }

  std::size_t getNex() { return nex; }

 private:
  bool optimization;
  bool approx;
  std::size_t deg;

  std::size_t mDegExtra;
  std::size_t mMaxIter;
  std::size_t mLanczosIter;
  std::size_t mMaxDeg;

  std::size_t N, nev, nex;

  // not sure about this, would we ever need more?
  double tol;
};
}
