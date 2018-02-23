/* -*- Mode: C++; -*- */
#pragma once

#include "algorithm/configuration.h"

namespace chase {

template <class T>
class Chase {
 public:
  virtual void Shift(T c, bool isunshift = false) = 0;
  virtual void HEMM(CHASE_INT nev, T alpha, T beta, CHASE_INT offset) = 0;
  virtual void QR(CHASE_INT fixednev) = 0;
  virtual void RR(Base<T> *ritzv, CHASE_INT block) = 0;
  virtual void Resd(Base<T> *ritzv, Base<T> *resd, CHASE_INT fixednev) = 0;
  virtual void Lanczos(CHASE_INT m, Base<T> *upperb) = 0;
  virtual void Lanczos(CHASE_INT M, CHASE_INT idx, Base<T> *upperb,
                       Base<T> *ritzv, Base<T> *Tau, Base<T> *ritzV) = 0;
  virtual void LanczosDos(CHASE_INT idx, CHASE_INT m, T *ritzVc) = 0;

  virtual void Swap(CHASE_INT i, CHASE_INT j) = 0;
  virtual void Lock(CHASE_INT new_converged) = 0;
  virtual void Reset() = 0;

  virtual std::size_t GetN() const = 0;
  virtual CHASE_INT GetNev() = 0;
  virtual CHASE_INT GetNex() = 0;
  virtual Base<T> *GetRitzv() = 0;
  virtual Base<T> *GetResid() = 0;
  virtual ChaseConfig<T> &GetConfig() = 0;

#ifdef OUTPUT
  virtual void Output(std::string str) = 0;
#endif
};
}  // namespace chase
