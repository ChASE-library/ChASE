/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <assert.h>
#include <memory>

#include "scalapack_fortran.hpp"
#include "fortran_mangle.h"

namespace chase {
namespace mpi {

void blacs_pinfo(int *mypnum, int *nprocs){
  BlasInt mypnum_, nprocs_;
  FC_GLOBAL(blacs_pinfo, BLACS_PINFO)(&mypnum_, &nprocs_);
  *mypnum = mypnum_;
  *nprocs = nprocs_;
}

void blacs_get(int *icontxt, const int *what, int *val ){
  BlasInt val_;
  BlasInt icontxt_;
  BlasInt what_ = *what;
  FC_GLOBAL(blacs_get, BLACS_GET)(&icontxt_, &what_, &val_);
  *val = val_; 
  *icontxt = icontxt_; 
}

void blacs_gridinit(int *icontxt, const char layout, const int *nprow, const int *npcol){
  BlasInt icontxt_;
  FC_GLOBAL(blacs_gridinit, BLACS_GRIDINIT)(&icontxt_, &layout, nprow, npcol);  
  *icontxt = icontxt_; 
}

void blacs_gridinfo(int *icontxt, int *nprow, int *npcol, int *myprow, int *mypcol){
  BlasInt nprow_, npcol_, myprow_, mypcol_, icontxt_;
  icontxt_ = *icontxt;
  FC_GLOBAL(blacs_gridinfo, BLACS_GRIDINFO)( &icontxt_, &nprow_, &npcol_, &myprow_, &mypcol_);
  *nprow = nprow_;
  *npcol = npcol_;
  *myprow = myprow_;
  *mypcol = mypcol_;
}

std::size_t numroc(std::size_t *n, std::size_t *nb, int *iproc, const int *isrcproc, int *nprocs){
  std::size_t val;
  BlasInt val_, n_, nb_;
  n_ = static_cast<int>(*n);
  nb_ = static_cast<int>(*nb);
  val_ = FC_GLOBAL(numroc, NUMROC)(&n_, &nb_, iproc, isrcproc, nprocs);
  val = static_cast<std::size_t>(val_);
  return val;
}

void descinit(std::size_t *desc, std::size_t *m, std::size_t *n, std::size_t *mb, std::size_t *nb, 
	const int *irsrc, const int *icsrc, int *ictxt, std::size_t *lld, int *info)
{
  BlasInt *desc_ = new BlasInt[9];
  BlasInt m_ = *m;
  BlasInt n_ = *n;
  BlasInt mb_ = *mb;
  BlasInt nb_ = *nb;
  const BlasInt irsrc_ = *irsrc;
  const BlasInt icsrc_ = *icsrc;
  BlasInt ictxt_ = *ictxt;
  BlasInt lld_ = *lld;
  BlasInt info_ = *info;
  FC_GLOBAL(descinit, DESCINIT)(desc_, &m_, &n_, &mb_, &nb_, &irsrc_, &icsrc_, &ictxt_, &lld_, &info_);
  *info = info_;

  for(int i = 0; i < 9; i++){
      desc[i] = static_cast<std::size_t>(desc_[i]);
  }

}

template <>
void t_geadd(const char trans, const std::size_t m, std::size_t n, float alpha, float *a, const std::size_t ia,
                const std::size_t ja, std::size_t *desc_a, float beta, float *c, const std::size_t ic,
                const std::size_t jc, std::size_t *desc_c)
{
  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt ic_ = ic;
  BlasInt jc_ = jc;
  BlasInt *desc_c_ = new BlasInt[9];

  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
      desc_c_[i] = static_cast<BlasInt>(desc_c[i]);
  }
  
  FC_GLOBAL(psgeadd, PSGEADD)
  (&trans, &m_, &n_, &alpha, a, &ia_, &ja_, desc_a_, &beta, c, &ic_, &jc_, desc_c_);

}

template <>
void t_geadd(const char trans, const std::size_t m, std::size_t n, double alpha, double *a, const std::size_t ia,	
                const std::size_t ja, std::size_t *desc_a, double beta, double *c, const std::size_t ic,
                const std::size_t jc, std::size_t *desc_c)
{
  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt ic_ = ic;
  BlasInt jc_ = jc;
  BlasInt *desc_c_ = new BlasInt[9];

  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
      desc_c_[i] = static_cast<BlasInt>(desc_c[i]);
  }

  FC_GLOBAL(pdgeadd, PDGEADD)
  (&trans, &m_, &n_, &alpha, a, &ia_, &ja_, desc_a_, &beta, c, &ic_, &jc_, desc_c_);

}

template <>
void t_geadd(const char trans, const std::size_t m, std::size_t n, std::complex<float> alpha, std::complex<float> *a, const std::size_t ia,
                const std::size_t ja, std::size_t *desc_a, std::complex<float> beta, std::complex<float> *c, const std::size_t ic,
                const std::size_t jc, std::size_t *desc_c)
{
  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt ic_ = ic;
  BlasInt jc_ = jc;
  BlasInt *desc_c_ = new BlasInt[9];
  char trans_;

  if(trans == 'T'){
      trans_ = 'C';
  }else{
      trans_ = trans;
  }

  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
      desc_c_[i] = static_cast<BlasInt>(desc_c[i]);
  }


  FC_GLOBAL(pcgeadd, PCGEADD)
  (&trans_, &m_, &n_, &alpha, a, &ia_, &ja_, desc_a_, &beta, c, &ic_, &jc_, desc_c_);

}


template <>
void t_geadd(const char trans, const std::size_t m, std::size_t n, std::complex<double> alpha, std::complex<double> *a, const std::size_t ia,
                const std::size_t ja, std::size_t *desc_a, std::complex<double> beta, std::complex<double> *c, const std::size_t ic,		
                const std::size_t jc, std::size_t *desc_c)
{
  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt ic_ = ic;
  BlasInt jc_ = jc;
  BlasInt *desc_c_ = new BlasInt[9];
  char trans_;

  if(trans == 'T'){
      trans_ = 'C';
  }else{
      trans_ = trans;
  }

  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
      desc_c_[i] = static_cast<BlasInt>(desc_c[i]);
  }


  FC_GLOBAL(pzgeadd, PZGEADD)
  (&trans_, &m_, &n_, &alpha, a, &ia_, &ja_, desc_a_, &beta, c, &ic_, &jc_, desc_c_);

}

template <>
void t_ppotrf(const char uplo, const std::size_t n, float *a, const std::size_t ia,
              const std::size_t ja, std::size_t *desc_a)
{
  BlasInt n_ = n;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt info_ = 0;
  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
  }

  FC_GLOBAL(pspotrf, PSPOTRF)
  (&uplo, &n_, a, &ia_, &ja_, desc_a_, &info_);
  assert(info_ == 0);

}

template <>
void t_ppotrf(const char uplo, const std::size_t n, double *a, const std::size_t ia,
              const std::size_t ja, std::size_t *desc_a)
{
  BlasInt n_ = n;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt info_ = 0;
  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
  }

  FC_GLOBAL(pdpotrf, PDPOTRF)
  (&uplo, &n_, a, &ia_, &ja_, desc_a_, &info_);
  assert(info_ == 0);

}


template <>
void t_ppotrf(const char uplo, const std::size_t n, std::complex<float> *a, const std::size_t ia,
              const std::size_t ja, std::size_t *desc_a)
{
  BlasInt n_ = n;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt info_ = 0;
  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
  }

  FC_GLOBAL(pcpotrf, PCPOTRF)
  (&uplo, &n_, a, &ia_, &ja_, desc_a_, &info_);
  assert(info_ == 0);

}

template <>
void t_ppotrf(const char uplo, const std::size_t n, std::complex<double> *a, const std::size_t ia,
              const std::size_t ja, std::size_t *desc_a)
{
  BlasInt n_ = n;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt info_ = 0;
  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
  }

  FC_GLOBAL(pzpotrf, PZPOTRF)
  (&uplo, &n_, a, &ia_, &ja_, desc_a_, &info_);
  assert(info_ == 0);

}


template <>
void t_psyhegst(const int ibtype, const char uplo,const std::size_t n, float *a, const std::size_t ia,
                const std::size_t ja, std::size_t *desc_a, const float *b, const std::size_t ib,
                const std::size_t jb, std::size_t *desc_b, Base<float> *scale)
{
  BlasInt ibtype_ = ibtype;
  BlasInt n_ = n;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt ib_ = ib;
  BlasInt jb_ = jb;
  BlasInt *desc_b_ = new BlasInt[9];

  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
      desc_b_[i] = static_cast<BlasInt>(desc_b[i]);
  }

  BlasInt info_ = 0;

  FC_GLOBAL(pssygst, PSSYGST)
  (&ibtype_, &uplo, &n_, a, &ia_, &ja_, desc_a_, b, &ib_, &jb_, desc_b_, scale, &info_);

  assert(info_ == 0);
}

template <>
void t_psyhegst(const int ibtype, const char uplo,const std::size_t n, double *a, const std::size_t ia,
                const std::size_t ja, std::size_t *desc_a, const double *b, const std::size_t ib,
                const std::size_t jb, std::size_t *desc_b, Base<double> *scale)
{
  BlasInt ibtype_ = ibtype;
  BlasInt n_ = n;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt ib_ = ib;
  BlasInt jb_ = jb;
  BlasInt *desc_b_ = new BlasInt[9];

  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
      desc_b_[i] = static_cast<BlasInt>(desc_b[i]);
  }

  BlasInt info_ = 0;

  FC_GLOBAL(pdsygst, PDSYGST)
  (&ibtype_, &uplo, &n_, a, &ia_, &ja_, desc_a_, b, &ib_, &jb_, desc_b_, scale, &info_);

  assert(info_ == 0);
}

template <>
void t_psyhegst(const int ibtype, const char uplo,const std::size_t n, std::complex<float> *a, const std::size_t ia,
                const std::size_t ja, std::size_t *desc_a, const std::complex<float> *b, const std::size_t ib,
                const std::size_t jb, std::size_t *desc_b, Base<std::complex<float>> *scale)
{
  BlasInt ibtype_ = ibtype;
  BlasInt n_ = n;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt ib_ = ib;
  BlasInt jb_ = jb;
  BlasInt *desc_b_ = new BlasInt[9];

  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
      desc_b_[i] = static_cast<BlasInt>(desc_b[i]);
  }

  BlasInt info_ = 0;

  FC_GLOBAL(pchegst, PCHEGST)
  (&ibtype_, &uplo, &n_, a, &ia_, &ja_, desc_a_, b, &ib_, &jb_, desc_b_, scale, &info_);

  assert(info_ == 0);
}

template <>
void t_psyhegst(const int ibtype, const char uplo,const std::size_t n, std::complex<double> *a, const std::size_t ia,
                const std::size_t ja, std::size_t *desc_a, const std::complex<double> *b, const std::size_t ib,
                const std::size_t jb, std::size_t *desc_b, Base<std::complex<double>> *scale)
{
  BlasInt ibtype_ = ibtype;
  BlasInt n_ = n;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt ib_ = ib;
  BlasInt jb_ = jb;
  BlasInt *desc_b_ = new BlasInt[9];

  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
      desc_b_[i] = static_cast<BlasInt>(desc_b[i]);
  }

  BlasInt info_ = 0;

  FC_GLOBAL(pzhegst, PZHEGST)
  (&ibtype_, &uplo, &n_, a, &ia_, &ja_, desc_a_, b, &ib_, &jb_, desc_b_, scale, &info_);

  assert(info_ == 0);
}

template <>
void t_ptrtrs(const char uplo, const char trans, const char diag, const std::size_t n,
              const std::size_t nhs, float *a,  const std::size_t ia, const std::size_t ja,
              std::size_t *desc_a,  float *b, const std::size_t ib, const std::size_t jb,
              std::size_t *desc_b)
{
  BlasInt n_ = n;
  BlasInt nhs_ = nhs;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt ib_ = ib;
  BlasInt jb_ = jb;
  BlasInt *desc_b_ = new BlasInt[9];

  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
      desc_b_[i] = static_cast<BlasInt>(desc_b[i]);
  }

  BlasInt info_ = 0;

  FC_GLOBAL(pstrtrs, PSTRTRS)
  (&uplo, &trans, &diag, &n_, &nhs_, a, &ia_, &ja_, desc_a_, b, &ib_, &jb_, desc_b_, &info_);

  assert(info_ == 0);

}

template <>
void t_ptrtrs(const char uplo, const char trans, const char diag, const std::size_t n,
              const std::size_t nhs, double *a,  const std::size_t ia, const std::size_t ja,
              std::size_t *desc_a,  double *b, const std::size_t ib, const std::size_t jb,
              std::size_t *desc_b)
{
  BlasInt n_ = n;
  BlasInt nhs_ = nhs;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt ib_ = ib;
  BlasInt jb_ = jb;
  BlasInt *desc_b_ = new BlasInt[9];

  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
      desc_b_[i] = static_cast<BlasInt>(desc_b[i]);
  }

  BlasInt info_ = 0;

  FC_GLOBAL(pdtrtrs, PDTRTRS)
  (&uplo, &trans, &diag, &n_, &nhs_, a, &ia_, &ja_, desc_a_, b, &ib_, &jb_, desc_b_, &info_);

  assert(info_ == 0);

}

template <>
void t_ptrtrs(const char uplo, const char trans, const char diag, const std::size_t n,
              const std::size_t nhs, std::complex<float> *a,  const std::size_t ia, const std::size_t ja,
              std::size_t *desc_a,  std::complex<float> *b, const std::size_t ib, const std::size_t jb,
              std::size_t *desc_b)
{
  BlasInt n_ = n;
  BlasInt nhs_ = nhs;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt ib_ = ib;
  BlasInt jb_ = jb;
  BlasInt *desc_b_ = new BlasInt[9];

  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
      desc_b_[i] = static_cast<BlasInt>(desc_b[i]);
  }

  BlasInt info_ = 0;

  FC_GLOBAL(pctrtrs, PCTRTRS)
  (&uplo, &trans, &diag, &n_, &nhs_, a, &ia_, &ja_, desc_a_, b, &ib_, &jb_, desc_b_, &info_);

  assert(info_ == 0);

}

template <>
void t_ptrtrs(const char uplo, const char trans, const char diag, const std::size_t n,
              const std::size_t nhs, std::complex<double> *a,  const std::size_t ia, const std::size_t ja,
              std::size_t *desc_a,  std::complex<double> *b, const std::size_t ib, const std::size_t jb,
              std::size_t *desc_b)
{
  BlasInt n_ = n;
  BlasInt nhs_ = nhs;
  BlasInt ia_ = ia;
  BlasInt ja_ = ja;
  BlasInt *desc_a_ = new BlasInt[9];
  BlasInt ib_ = ib;
  BlasInt jb_ = jb;
  BlasInt *desc_b_ = new BlasInt[9];

  for(int i = 0; i < 9; i++){
      desc_a_[i] = static_cast<BlasInt>(desc_a[i]);
      desc_b_[i] = static_cast<BlasInt>(desc_b[i]);
  }

  BlasInt info_ = 0;

  FC_GLOBAL(pztrtrs, PZTRTRS)
  (&uplo, &trans, &diag, &n_, &nhs_, a, &ia_, &ja_, desc_a_, b, &ib_, &jb_, desc_b_, &info_);

  assert(info_ == 0);

}



}
}
