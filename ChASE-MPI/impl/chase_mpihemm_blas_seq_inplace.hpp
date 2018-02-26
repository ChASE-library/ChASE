/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2018, Simulation Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany
// and
// Copyright (c) 2016-2018, Aachen Institute for Advanced Study in Computational
//   Engineering Science, RWTH Aachen University, Germany All rights reserved.
// License is 3-clause BSD:
// https://github.com/SimLabQuantumMaterials/ChASE/

#pragma once

#include <cstring>
#include <memory>

#include "chase_mpihemm_interface.hpp"

namespace chase {
namespace mpi {

template <class T>
class ChaseMpiHemmBlasSeqInplace : public ChaseMpiHemmInterface<T> {
 public:
  ChaseMpiHemmBlasSeqInplace(ChaseMpiMatrices<T>& matrices, std::size_t n,
                        std::size_t maxBlock)
      : N_(n),
        V1_(matrices.get_V1()),
        V2_(matrices.get_V2()),
        H_(matrices.get_H()) {}

  ~ChaseMpiHemmBlasSeqInplace() {}

  void preApplication(T* V, std::size_t locked, std::size_t block) {
    locked_ = locked;

    if (V != V1_) std::swap(V1_, V2_);
    assert(V == V1_);
  }

  void preApplication(T* V1, T* V2, std::size_t locked, std::size_t block) {
    this->preApplication(V1, locked, block);
  }

  void apply(T const alpha, T const beta, std::size_t const offset,
             std::size_t const block) {
    assert(V2_ != V1_);
    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           N_, block, N_,                              //
           &alpha,                                     // V2_ <-
           H_, N_,                                     //   H * V1_
           V1_ + (locked_ + offset) * N_, N_,          //   + V2_
           &beta,                                      //
           V2_ + (locked_ + offset) * N_, N_);         //

    std::swap(V1_, V2_);
  }

  bool postApplication(T* V, std::size_t block) {
    // this is somewhat a hack, but causes the approxV in the next
    // preApplication to be the same pointer content as V1_
    // std::swap(V1_, V2_);

    assert(V == V1_);

    return false;
  }

  void shiftMatrix(T const c) override {
    for (std::size_t i = 0; i < N_; ++i) {
      H_[i + i * N_] += c;
    }
  }

  void applyVec(T* B, T* C) override {
    T alpha = T(1.0);
    T beta = T(0.0);

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           N_, 1, N_,                                  //
           &alpha,                                     //
           H_, N_,                                     //
           B, N_,                                      //
           &beta,                                      //
           C, N_);                                     //
  }

  T* get_H() const override { return H_; }

  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) const override {
    *xoff = 0;
    *yoff = 0;
    *xlen = N_;
    *ylen = N_;
  }

 private:
  std::size_t N_;
  std::size_t locked_;

  T* H_;
  T* V1_;
  T* V2_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiHemmBlasSeqInplace<T>> {
  static const bool value = false;
};

}  // namespace matrixfree
}  // namespace chase
