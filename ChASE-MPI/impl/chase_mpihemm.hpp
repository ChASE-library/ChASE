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

#include <mpi.h>
#include <iterator>

#include "ChASE-MPI/chase_mpi_properties.hpp"
#include "ChASE-MPI/chase_mpihemm_interface.hpp"

namespace chase {
namespace mpi {

template <class T>
class ChaseMpiHemm : public ChaseMpiHemmInterface<T> {
 public:
  ChaseMpiHemm(ChaseMpiProperties<T>* matrix_properties,
               ChaseMpiHemmInterface<T>* gemm)
      : gemm_(gemm) {
    ldc_ = matrix_properties->get_ldc();
    ldb_ = matrix_properties->get_ldb();

    N_ = matrix_properties->get_N();
    n_ = matrix_properties->get_n();
    m_ = matrix_properties->get_m();

    H_ = matrix_properties->get_H();
    B_ = matrix_properties->get_B();
    C_ = matrix_properties->get_C();
    IMT_ = matrix_properties->get_IMT();

    matrix_properties_ = matrix_properties;

    row_comm_ = matrix_properties->get_row_comm();
    col_comm_ = matrix_properties->get_col_comm();

    dims_ = matrix_properties->get_dims();
    coord_ = matrix_properties->get_coord();
    off_ = matrix_properties->get_off();
  }

  ~ChaseMpiHemm() {}

  void preApplication(T* V, std::size_t locked, std::size_t block) {
    next_ = NextOp::bAc;
    locked_ = locked;

    for (auto j = 0; j < block; j++) {
      std::memcpy(C_ + j * m_, V + j * N_ + off_[0] + locked * N_,
                  m_ * sizeof(T));
    }

    gemm_->preApplication(V, locked, block);
  }

  void preApplication(T* V1, T* V2, std::size_t locked, std::size_t block) {
    for (auto j = 0; j < block; j++) {
      std::memcpy(B_ + j * n_, V2 + j * N_ + off_[1] + locked * N_,
                  n_ * sizeof(T));
    }

    gemm_->preApplication(V1, V2, locked, block);

    this->preApplication(V1, locked, block);
  }

  void apply(T alpha, T beta, std::size_t offset, std::size_t block) {
    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t dim;
    if (next_ == NextOp::bAc) {
      dim = n_ * block;
      gemm_->apply(One, Zero, offset, block);

      MPI_Allreduce(MPI_IN_PLACE, IMT_ + offset * n_, dim, getMPI_Type<T>(),
                    MPI_SUM, col_comm_);

      t_scal(dim, &beta, B_ + offset * n_, 1);
      t_axpy(dim, &alpha, IMT_ + offset * n_, 1, B_ + offset * n_, 1);

      next_ = NextOp::cAb;
    } else {  // cAb

      dim = m_ * block;
      gemm_->apply(One, Zero, offset, block);

      MPI_Allreduce(MPI_IN_PLACE, IMT_ + offset * m_, dim, getMPI_Type<T>(),
                    MPI_SUM, row_comm_);

      t_scal(dim, &beta, C_ + offset * m_, 1);
      t_axpy(dim, &alpha, IMT_ + offset * m_, 1, C_ + offset * m_, 1);

      next_ = NextOp::bAc;
    }
  }
  bool postApplication(T* V, std::size_t block) {
    gemm_->postApplication(V, block);

    std::size_t N = N_;
    std::size_t dimsIdx;
    std::size_t subsize;
    T* buff;
    MPI_Comm comm;

    T* targetBuf = V + locked_ * N;

    if (next_ == NextOp::bAc) {
      subsize = m_;
      buff = C_;
      comm = col_comm_;
      dimsIdx = 0;
    } else {
      subsize = n_;
      buff = B_;
      comm = row_comm_;
      dimsIdx = 1;
    }

    int gsize, rank;
    MPI_Comm_size(comm, &gsize);
    MPI_Comm_rank(comm, &rank);

    auto& recvcounts = matrix_properties_->get_recvcounts()[dimsIdx];
    auto& displs = matrix_properties_->get_displs()[dimsIdx];

    std::vector<MPI_Request> reqs(gsize);
    std::vector<MPI_Datatype> newType(gsize);

    // Set up the datatype for the recv
    for (auto i = 0; i < gsize; ++i) {
      int array_of_sizes[2] = {static_cast<int>(N_), 1};
      int array_of_subsizes[2] = {recvcounts[i], 1};
      int array_of_starts[2] = {displs[i], 0};

      MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                               array_of_starts, MPI_ORDER_FORTRAN,
                               getMPI_Type<T>(), &(newType[i]));

      MPI_Type_commit(&(newType[i]));
    }

    for (auto i = 0; i < gsize; ++i) {
      if (rank == i) {
        // The sender sends from the appropriate buffer
        MPI_Ibcast(buff, recvcounts[i] * block, getMPI_Type<T>(), i, comm,
                   &reqs[i]);
      } else {
        // The recv goes right unto the correct buffer
        MPI_Ibcast(targetBuf, block, newType[i], i, comm, &reqs[i]);
      }
    }

    int i = rank;
    // we copy the sender into the target Buffer directly
    for (auto j = 0; j < block; ++j) {
      std::memcpy(targetBuf + j * N + displs[i], buff + recvcounts[i] * j,
                  recvcounts[i] * sizeof(T));
    }

    MPI_Waitall(gsize, reqs.data(), MPI_STATUSES_IGNORE);

    for (auto i = 0; i < gsize; ++i) {
      MPI_Type_free(&newType[i]);
    }
	return true;
  }

  void shiftMatrix(T c, bool isunshift = false) {
    for (std::size_t i = 0; i < n_; i++) {
      for (std::size_t j = 0; j < m_; j++) {
        if (off_[0] + j == (i + off_[1])) {
          H_[i * m_ + j] += c;
        }
      }
    }
    gemm_->shiftMatrix(c);
  }

  void applyVec(T* B, T* C) {
    // TODO
    T One = T(1.0);
    T Zero = T(0.0);

    this->preApplication(B, 0, 1);
    this->apply(One, Zero, 0, 1);
    this->postApplication(C, 1);

    // gemm_->applyVec(B, C);
  }

  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) const override {
    *xoff = off_[0];
    *yoff = off_[1];
    *xlen = m_;
    *ylen = n_;
  }

  T* get_H() const override { return matrix_properties_->get_H(); }
  void Start() override { gemm_->Start(); }

 private:
  enum NextOp { cAb, bAc };

  std::size_t locked_;
  std::size_t ldc_;
  std::size_t ldb_;

  std::size_t n_;
  std::size_t m_;
  std::size_t N_;

  T* H_;
  T* B_;
  T* C_;
  T* IMT_;

  NextOp next_;
  MPI_Comm row_comm_, col_comm_;
  int* dims_;
  int* coord_;
  std::size_t* off_;

  std::unique_ptr<ChaseMpiHemmInterface<T>> gemm_;
  ChaseMpiProperties<T>* matrix_properties_;
};
}  // namespace mpi
}  // namespace chase
