/* -*- Mode: C++; -*- */
#pragma once

#include <mpi.h>
#include <vector>

#include "algorithm/types.h"

namespace chase {

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
class SkewedMatrixProperties {
 public:
  SkewedMatrixProperties(std::size_t N, std::size_t max_block,
                         MPI_Comm comm = MPI_COMM_WORLD)
      : N_(N), max_block_(max_block), comm_(comm) {
    MPI_Int periodic[] = {0, 0};
    MPI_Int reorder = 0;
    MPI_Int free_coords[2];
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

    /*
    //Calculating ranks which will consist wog and col group and later row and
    col communicators
    MPI_Cart_coords(cartComm, rank_, 2, coord_);
    CHASE_MPIINT* ranks_row = (CHASE_MPIINT*)malloc(dims_[1] *
    sizeof(CHASE_MPIINT));
    CHASE_MPIINT* ranks_col = (CHASE_MPIINT*)malloc(dims_[0] *
    sizeof(CHASE_MPIINT));
    int r;
    MPI_Group origGroup, ROW, COL;

    for (auto i = 0; i < dims_[0]; i++) {
        coord_[0] = i;
        coord_[1] = coord_[1];
        MPI_Cart_rank(cartComm, coord_, &r);
        ranks_col[i] = r;
    }

    for (auto j = 0; j < dims_[1]; j++) {
        coord_[1] = j;
        coord_[0] = coord_[0];
        MPI_Cart_rank(cartComm, coord_, &r);
        ranks_row[j] = r;
        MPI_Cart_rank(cartComm, coord_, &r);
        ranks_row[j] = r;
    }
    MPI_Comm_group(cartComm, &(origGroup));
    MPI_Group_incl(origGroup, dims_[1], ranks_row, &(ROW));
    MPI_Group_incl(origGroup, dims_[0], ranks_col, &(COL));
    MPI_Comm_create(comm, ROW, &(row_comm_));
    MPI_Comm_create(comm, COL, &(col_comm_));
    MPI_Cart_coords(cartComm, rank_, 2, coord_);
    */

    // offsets
    off_[0] = coord_[0] * (N_ / dims_[0]);
    off_[1] = coord_[1] * (N_ / dims_[1]);

    // size of local part of H
    if (coord_[0] < dims_[0] - 1) {
      m_ = N_ / dims_[0];
    } else {
      m_ = N_ - (dims_[0] - 1) * (N_ / dims_[0]);
    }
    if (coord_[1] < dims_[1] - 1) {
      n_ = N_ / dims_[1];
    } else {
      n_ = N_ - (dims_[1] - 1) * (N_ / dims_[1]);
    }

    // TODO unique_ptr
    H_ = new T[n_ * m_]();
    B_ = new T[n_ * max_block_]();
    C_ = new T[m_ * max_block_]();
    IMT_ = new T[std::max(n_, m_) * max_block_]();

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
          recvcounts_[dim_idx][i] = N_ / dims_[dim_idx];
          displs_[dim_idx][i] = i * recvcounts_[dim_idx][0];
        }
        recvcounts_[dim_idx][dims_[dim_idx] - 1] =
            N_ - (dims_[dim_idx] - 1) * (N_ / dims_[dim_idx]);

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

  MPI_Comm get_row_comm() { return row_comm_; }
  MPI_Comm get_col_comm() { return col_comm_; }

  // dimensions of cartesian communicator grid
  MPI_Int* get_dims() { return dims_; }
  // offsets of this rank
  MPI_Int* get_off() { return off_; }
  // coordinates in the cartesian communicator grid
  MPI_Int* get_coord() { return coord_; }

  // some nice utility functions
  std::size_t get_ldb() { return n_; };
  std::size_t get_ldc() { return m_; };

  T* get_H() { return H_; }
  T* get_B() { return B_; }
  T* get_C() { return C_; }
  T* get_IMT() { return IMT_; }

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

 private:
  std::size_t N_;
  std::size_t max_block_;

  std::size_t n_;
  std::size_t m_;

  std::vector<std::vector<int> > recvcounts_;
  std::vector<std::vector<int> > displs_;
  std::vector<std::vector<MPI_Datatype> > newType_;

  MPI_Comm comm_;

  MPI_Int nprocs_;
  MPI_Int rank_;
  MPI_Int rank_cart_[2];

  T* H_;
  T* B_;
  T* C_;
  T* IMT_;

  MPI_Comm row_comm_, col_comm_;
  MPI_Int dims_[2];
  MPI_Int coord_[2];
  CHASE_INT off_[2];
};
}
