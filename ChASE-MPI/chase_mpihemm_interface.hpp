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

#include <cstdlib>

#include "algorithm/types.hpp"
#include "chase_mpi_properties.hpp"
//
//  MatrixFreeInterface provides the Matrix-Matrix multiplication for use in
//  ChASE. The core functionality is apply(), which performs a HEMM.

// ASSUMPTION: MatrixFreeInterface assumes that apply() is called an even number
//             of times. This is always the case when used with ChASE_Algorithm.

// ASSUMPTION: shiftMatrix must be called at least once before apply() is called
//             (Some implementations may use shift() to transfer the matrix to
//              the GPU)

// The manner is which the Matrix H is loaded into the class is defined by the
// subclass. Further, the size of the vectors V1 and V2 must be known to any
// subclass.

namespace chase {
namespace mpi {

// Type Trait
template <typename T>
struct is_skewed_matrixfree {
  static const bool value = false;
};

template <class T>
class ChaseMpiHemmInterface {
 public:
  typedef T value_type;

  virtual ~ChaseMpiHemmInterface() {};

  // After a call to shiftMatrix(T c) all subsequent calls to apply() and
  //   applyVec() perform the multiplication with a shifted H
  virtual void shiftMatrix(T c,bool isunshift = false) = 0;

  // preApplication prepares internal state to perform the GEMM:
  // V2 <- alpha * H*V1 + beta*V2
  // The first locked number of vectors of V1 are not used in the GEMM.
  //     In ChASE these are the locked vectors.
  // Starting from locked, V1 contains block number of vectors.
  virtual void preApplication(T* V1, T* V2, std::size_t locked,
                              std::size_t block) = 0;

  // This function only populates V1.
  // After a call to this function the state of V2 is undefined.
  virtual void preApplication(T* V1, std::size_t locked, std::size_t block) = 0;

  // Performs V2<- alpha*V1*H + beta*V2 and swap(V1,V2).
  // The first offset vectors of V1 and V2 are not part of the GEMM.
  // The GEMM uses block vectors.
  // block must be smaller or equal to the block passed in preApplication.
  // Including the locked vectors passed into apply we perform, in MATLAB
  // notation
  // V2[:,locked+offset, locked+offset+block]
  //       <- alpha* V1[:,locked+offset,locked+offset+block]*H
  //          + beta*V2[:,locked+offset,locked+offset+block]
  virtual void apply(T alpha, T beta, std::size_t offset,
                     std::size_t block) = 0;

  // Copies V2, the result of one or more results of apply() to V.
  // block number of vectors are copied.
  // The first locked ( as supplied to preApplication() ) vectors are skipped
  virtual bool postApplication(T* V, std::size_t block) = 0;

  // Performs a GEMV with alpha=1.0 and beta=0.0
  // Equivalent to calling:
  //
  // preApplication( V1, 0, 1 );
  // apply( T(1.0), T(0.0), 0, 1 );
  // postApplication( C, 1 );
  virtual void applyVec(T* B, T* C) = 0;

  // The offsets and sizes of the block of the matrix H that the class uses.
  // In a subclass that uses MPI parallelization of the HEMM offsets may be
  // non-zero for some processes.
  virtual void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
                       std::size_t* ylen) const = 0;

  // Returns ptr to H, which may be used to populate H.
  virtual T* get_H() const = 0;
  virtual void Start() = 0;
};
}  // namespace matrixfree
}  // namespace chase
