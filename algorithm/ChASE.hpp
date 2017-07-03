/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */
#pragma once

#include <complex>
#include <cstring>  //memcpy
#include <iostream>
#include <iostream>
#include <random>

typedef int CHASE_INT;

// Base< std::complex< Base<T> > > -> Base<T>
// Base<               Base<T>   > -> Base<T>
template <class Q>
struct Base_Class {
  typedef Q type;
};

template <class Q>
struct Base_Class<std::complex<Q> > {
  typedef Q type;
};

template <typename Q>
using Base = typename Base_Class<Q>::type;

#include "ChASE_Config.hpp"
#include "ChASE_Perf.hpp"

template <class T>
class ChASE {
 public:
  virtual void shift(T c, bool isunshift = false) = 0;
  virtual void threeTerms(CHASE_INT nev, T alpha, T beta, CHASE_INT offset) = 0;
  virtual void QR(CHASE_INT fixednev) = 0;
  virtual void RR(Base<T>* ritzv, CHASE_INT block) = 0;
  virtual void resd(Base<T>* ritzv, Base<T>* resd, CHASE_INT fixednev) = 0;
  virtual void swap(CHASE_INT i, CHASE_INT j) = 0;
  virtual void lanczos(CHASE_INT m, Base<T>* upperb) = 0;
  virtual void lanczos(CHASE_INT M, CHASE_INT idx, Base<T>* upperb,
                       Base<T>* ritzv, Base<T>* Tau, Base<T>* ritzV) = 0;
  virtual void lanczosDoS(CHASE_INT idx, CHASE_INT m, T* ritzVc) = 0;
  virtual void lock(CHASE_INT new_converged) = 0;
  //    virtual void cpy(CHASE_INT new_converged) = 0;
  virtual Base<T> getNorm() = 0;

  virtual std::size_t getN() = 0;
  virtual CHASE_INT getNev() = 0;
  virtual CHASE_INT getNex() = 0;
  virtual Base<T>* getRitzv() = 0;

  virtual ChASE_Config getConfig() = 0;
  virtual void solve() = 0;

  virtual void get_off(CHASE_INT* xoff, CHASE_INT* yoff, CHASE_INT* xlen,
                       CHASE_INT* ylen) = 0;

#ifdef OUTPUT
  virtual void output(std::string str) = 0;
#endif
};

#include "ChASE_Algorithm.hpp"
