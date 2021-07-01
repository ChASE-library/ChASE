/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#ifndef CHASE_ALGORITHM_INTERFACE_HPP
#define CHASE_ALGORITHM_INTERFACE_HPP

#include "configuration.hpp"
#include "types.hpp"

namespace chase {

template <class T>
class Chase {
 public:
  virtual void Shift(T c, bool isunshift = false) = 0;
  virtual void HEMM(std::size_t nev, T alpha, T beta, std::size_t offset) = 0;
  virtual void QR(std::size_t fixednev) = 0;
  virtual void RR(Base<T> *ritzv, std::size_t block) = 0;
  virtual void Resd(Base<T> *ritzv, Base<T> *resd, std::size_t fixednev) = 0;
  virtual void Lanczos(std::size_t m, Base<T> *upperb) = 0;
  virtual void Lanczos(std::size_t M, std::size_t idx, Base<T> *upperb,
                       Base<T> *ritzv, Base<T> *Tau, Base<T> *ritzV) = 0;
  virtual void LanczosDos(std::size_t idx, std::size_t m, T *ritzVc) = 0;

  virtual void Swap(std::size_t i, std::size_t j) = 0;
  virtual void Lock(std::size_t new_converged) = 0;
  virtual void Start() = 0;
  virtual void End() = 0;

  virtual std::size_t GetN() const = 0;
  virtual std::size_t GetNev() = 0;
  virtual std::size_t GetNex() = 0;
  virtual Base<T> *GetRitzv() = 0;
  virtual Base<T> *GetResid() = 0;
  virtual ChaseConfig<T> &GetConfig() = 0;

#ifdef CHASE_OUTPUT
  virtual void Output(std::string str) = 0;
#endif
};
}  // namespace chase

#endif
