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
#include <memory>
#include <vector>

#include "algorithm/types.hpp"
#include "chase_mpi_matrices.hpp"

namespace chase {
namespace mpi {

/*
 * Logic for an MPI parallel HEMM
 */

template <typename T>
MPI_Datatype getMPI_Type();

template <>
MPI_Datatype getMPI_Type<float>() {
  return MPI_FLOAT;
}

template <>
MPI_Datatype getMPI_Type<double>() {
  return MPI_DOUBLE;
}

template <>
MPI_Datatype getMPI_Type<std::complex<float> >() {
  return MPI_COMPLEX;
}

template <>
MPI_Datatype getMPI_Type<std::complex<double> >() {
  return MPI_DOUBLE_COMPLEX;
}

template <class T>
class ChaseMpiProperties {
 public:
  ChaseMpiProperties(std::size_t N, std::size_t nev, std::size_t nex,
                     MPI_Comm comm = MPI_COMM_WORLD)
      : N_(N), nev_(nev), nex_(nex), max_block_(nev + nex), comm_(comm) {
    int periodic[] = {0, 0};
    int reorder = 0;
    int free_coords[2];
    MPI_Comm cartComm;

    // create cartesian communicator
    MPI_Comm_size(comm, &nprocs_);
    dims_[0] = dims_[1] = 0;
    MPI_Dims_create(nprocs_, 2, dims_);
    MPI_Cart_create(comm, 2, dims_, periodic, reorder, &cartComm);
    MPI_Comm_size(cartComm, &nprocs_);
    MPI_Comm_rank(cartComm, &rank_);
    MPI_Cart_coords(cartComm, rank_, 2, coord_);

    if (nprocs_ > N_) throw std::exception();

    // row communicator
    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(cartComm, free_coords, &row_comm_);

    // column communicator
    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(cartComm, free_coords, &col_comm_);

    // size of local part of H
    int len;

    len = std::min(N_, N_ / dims_[0] + 1);
    off_[0] = coord_[0] * len;

    if (coord_[0] < dims_[0] - 1) {
      m_ = len;
    } else {
      m_ = N_ - (dims_[0] - 1) * len;
    }

    len = std::min(N_, N_ / dims_[1] + 1);
    off_[1] = coord_[1] * len;

    if (coord_[1] < dims_[1] - 1) {
      n_ = len;
    } else {
      n_ = N_ - (dims_[1] - 1) * len;
    }

    H_.reset(new T[n_ * m_]());
    B_.reset(new T[n_ * max_block_]());
    C_.reset(new T[m_ * max_block_]());
    IMT_.reset(new T[std::max(n_, m_) * max_block_]());

    {
      recvcounts_.resize(2);
      displs_.resize(2);
      newType_.resize(2);

      MPI_Comm_rank(col_comm_, &rank_cart_[0]);
      MPI_Comm_rank(row_comm_, &rank_cart_[1]);

      for (std::size_t dim_idx = 0; dim_idx < 2; dim_idx++) {
        recvcounts_[dim_idx].resize(dims_[dim_idx]);
        displs_[dim_idx].resize(dims_[dim_idx]);
        newType_[dim_idx].resize(dims_[dim_idx]);

        for (auto i = 0; i < dims_[dim_idx]; ++i) {
          len = std::min(N_, N_ / dims_[dim_idx] + 1);
          recvcounts_[dim_idx][i] = len;
          displs_[dim_idx][i] = i * recvcounts_[dim_idx][0];
        }
        recvcounts_[dim_idx][dims_[dim_idx] - 1] =
            N_ - (dims_[dim_idx] - 1) * len;

        // TODO //
        // we reduce over rows or cols and not all of them
        // -> nprocs_ -> dims_[0]

        for (auto i = 0; i < dims_[dim_idx]; ++i) {
          // int array_of_sizes[2] = { N, max_block_ };
          // int array_of_subsizes[2] = { recvcounts_[dim_idx][i], max_block_ };
          int array_of_sizes[2] = {static_cast<int>(N_), 1};
          int array_of_subsizes[2] = {recvcounts_[dim_idx][i], 1};
          int array_of_starts[2] = {displs_[dim_idx][i], 0};

          MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                                   array_of_starts, MPI_ORDER_FORTRAN,
                                   getMPI_Type<T>(), &(newType_[dim_idx][i]));

          MPI_Type_commit(&(newType_[dim_idx][i]));
        }
      }
    }
  }

  // destruktor
  /*
  for (auto i = 0; i < gsize; ++i) {
      MPI_Type_free(&newType[i]);
  }
*/

  std::size_t get_N() { return N_; };
  std::size_t get_n() { return n_; };
  std::size_t get_m() { return m_; };
  std::size_t get_max_block() { return max_block_; };
  std::size_t GetNev() { return nev_; };
  std::size_t GetNex() { return nex_; };

  MPI_Comm get_row_comm() { return row_comm_; }
  MPI_Comm get_col_comm() { return col_comm_; }

  // dimensions of cartesian communicator grid
  int* get_dims() { return dims_; }
  // offsets of this rank
  std::size_t* get_off() { return off_; }
  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) {
    *xoff = off_[0];
    *yoff = off_[1];
    *xlen = m_;
    *ylen = n_;
  }
  // coordinates in the cartesian communicator grid
  int* get_coord() { return coord_; }

  // some nice utility functions
  std::size_t get_ldb() { return n_; };
  std::size_t get_ldc() { return m_; };

  T* get_H() { return H_.get(); }
  T* get_B() { return B_.get(); }
  T* get_C() { return C_.get(); }
  T* get_IMT() { return IMT_.get(); }

  const std::vector<std::vector<MPI_Datatype> >& get_new_type() {
    return newType_;
  }
  const std::vector<std::vector<int> >& get_recvcounts() { return recvcounts_; }
  const std::vector<std::vector<int> >& get_displs() { return displs_; }

  int get_nprocs() { return nprocs_; }
  // TODO this should take a dimIdx and return the rank of the col or row
  // communicator
  int get_my_rank() { return rank_; }

  int get_my_rank(std::size_t dim_idx) { return rank_cart_[dim_idx]; }

  ChaseMpiMatrices<T> create_matrices(T* V1 = nullptr, Base<T>* ritzv = nullptr,
                                      T* V2 = nullptr,
                                      Base<T>* resid = nullptr) const {
    return ChaseMpiMatrices<T>(comm_, N_, max_block_, V1, ritzv, V2, resid);
  }

 private:
  std::size_t N_;
  std::size_t nev_;
  std::size_t nex_;
  std::size_t max_block_;

  std::size_t n_;
  std::size_t m_;

  // TODO this should be std::array<std::vector<>,2>
  std::vector<std::vector<int> > recvcounts_;
  std::vector<std::vector<int> > displs_;
  std::vector<std::vector<MPI_Datatype> > newType_;

  MPI_Comm comm_;

  int nprocs_;
  int rank_;
  int rank_cart_[2];

  std::unique_ptr<T[]> H_;
  std::unique_ptr<T[]> B_;
  std::unique_ptr<T[]> C_;
  std::unique_ptr<T[]> IMT_;

  MPI_Comm row_comm_, col_comm_;
  int dims_[2];
  int coord_[2];
  std::size_t off_[2];
};
}  // namespace mpi
}  // namespace chase
