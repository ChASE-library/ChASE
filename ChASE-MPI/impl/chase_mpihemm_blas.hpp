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

#include "ChASE-MPI/blas_templates.hpp"
#include "ChASE-MPI/chase_mpihemm_interface.hpp"
#include "ChASE-MPI/chase_mpi_properties.hpp"

namespace chase {
namespace mpi {
//
//  This Class is meant to be used with MatrixFreeMPI
//
template <class T>
class ChaseMpiHemmBlas : public ChaseMpiHemmInterface<T> {
 public:
  ChaseMpiHemmBlas(ChaseMpiProperties<T>* matrix_properties) {
    // TODO
    // ldc_ = matrix_properties->get_ldc();
    // ldb_ = matrix_properties->get_ldb();

    n_ = matrix_properties->get_n();
    m_ = matrix_properties->get_m();
    N_ = matrix_properties->get_N();

    H_ = matrix_properties->get_H();
    B_ = matrix_properties->get_B();
    C_ = matrix_properties->get_C();
    IMT_ = matrix_properties->get_IMT();

    matrix_properties_ = matrix_properties;
  }

  ~ChaseMpiHemmBlas() {}

  void preApplication(T* V, std::size_t locked, std::size_t block) {
    next_ = NextOp::bAc;
    // std::memcpy(C_, V + locked_ * N_, N_ * block * sizeof(T));
  }

  void preApplication(T* V1, T* V2, std::size_t locked, std::size_t block) {
    // std::memcpy(B_, V2 + locked * N_, N_ * block * sizeof(T));
    this->preApplication(V1, locked, block);
  }

  void apply(T alpha, T beta, std::size_t offset, std::size_t block) {
    if (next_ == NextOp::bAc) {
      t_gemm<T>(CblasColMajor, CblasConjTrans, CblasNoTrans, n_,
                static_cast<std::size_t>(block), m_, &alpha, H_, m_,
                C_ + offset * m_, m_, &beta, IMT_ + offset * n_, n_);
      next_ = NextOp::cAb;
    } else {
      t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_,
             static_cast<std::size_t>(block), n_, &alpha, H_, m_,
             B_ + offset * n_, n_, &beta, IMT_ + offset * m_, m_);
      next_ = NextOp::bAc;
    }
  }

  // deg is always even so we know that we return C?
  bool postApplication(T* V, std::size_t block) {
    T* buff;
    if (next_ == NextOp::bAc) {
      buff = C_;
    } else {
      buff = B_;
    }

    // std::memcpy(V + locked_ * N_, buff, N_ * block * sizeof(T));
    return false;
  }

  void shiftMatrix(T c) {
    // for (std::size_t i = 0; i < n_; i++) {
    //     H_[i * m_ + i] += c;
    // }
  }

  void applyVec(T* B, T* C) {
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

  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) const override {
    *xoff = 0;
    *yoff = 0;
    *xlen = static_cast<std::size_t>(N_);
    *ylen = static_cast<std::size_t>(N_);
  }

  T* get_H() const override { return matrix_properties_->get_H(); }

 private:
  enum NextOp { cAb, bAc };

  NextOp next_;
  std::size_t N_;

  std::size_t n_;
  std::size_t m_;

  T* H_;
  T* B_;
  T* C_;
  T* IMT_;

  ChaseMpiProperties<T>* matrix_properties_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiHemmBlas<T>> {
  static const bool value = true;
};

}  // namespace matrixfree
}  // namespace chase
